#if !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

/*
    mnist_max_m4.c

    A purpose-built, single-file MNIST CNN trainer. On macOS it uses:
      - Apple Accelerate SGEMM/SGEMV/SDOT for tuned, multithreaded linear algebra
      - Grand Central Dispatch for non-BLAS parallel kernels
      - ARM NEON for fused elementwise and AdamW kernels on Apple Silicon
      - mmap for zero-copy access to the four raw float32 data files

    Maximum-performance Apple Silicon build:
      clang -std=c11 -O3 -ffast-math -flto -mcpu=native -DNDEBUG mnist_max_m4.c \
            -framework Accelerate -o mnist_max

    Expected files in the working directory:
      train_images.mat  60000 x 784 float32, normalized to [0,1]
      train_labels.mat  60000 x 1   float32, integer-valued labels 0..9
      test_images.mat   10000 x 784 float32, normalized to [0,1]
      test_labels.mat   10000 x 1   float32, integer-valued labels 0..9

    Non-Apple builds use a correctness-oriented scalar GEMM fallback. The fast
    path is intentionally specialized for macOS/Apple Silicon.
*/

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#define ML_APPLE_ACCELERATE 1
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define ML_ARM_NEON 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ML_INLINE static inline __attribute__((always_inline))
#define ML_RESTRICT __restrict__
#define ML_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ML_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ML_INLINE static inline
#define ML_RESTRICT restrict
#define ML_LIKELY(x)   (x)
#define ML_UNLIKELY(x) (x)
#endif

#ifndef MNIST_TRAIN_COUNT
#define MNIST_TRAIN_COUNT 60000u
#endif
#ifndef MNIST_TEST_COUNT
#define MNIST_TEST_COUNT 10000u
#endif
#ifndef TRAIN_BATCH_SIZE
#define TRAIN_BATCH_SIZE 256u
#endif
#ifndef TRAIN_EPOCHS
#define TRAIN_EPOCHS 20u
#endif
#ifndef CONV1_CHANNELS
#define CONV1_CHANNELS 16u
#endif
#ifndef CONV2_CHANNELS
#define CONV2_CHANNELS 32u
#endif
#ifndef HIDDEN_UNITS
#define HIDDEN_UNITS 128u
#endif

#define IMAGE_H 28u
#define IMAGE_W 28u
#define IMAGE_PIXELS (IMAGE_H * IMAGE_W)
#define CLASS_COUNT 10u
#define KERNEL 3u
#define KERNEL_AREA 9u
#define POOL1_H 14u
#define POOL1_W 14u
#define POOL2_H 7u
#define POOL2_W 7u
#define FLAT_FEATURES (POOL2_H * POOL2_W * CONV2_CHANNELS)
#define ALIGNMENT 64u

#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPSILON 1.0e-8f
#define GRAD_CLIP 5.0f
#define PI_F 3.14159265358979323846f
#define SOFTMAX_FLOOR 1.0e-12f

_Static_assert(IMAGE_H % 4u == 0u && IMAGE_W % 4u == 0u,
               "The two 2x2 pooling stages require dimensions divisible by four");

typedef struct {
    void *address;
    size_t bytes;
    int fd;
} mapped_file;

typedef void (*range_function)(size_t begin, size_t end, void *context);

typedef struct {
    size_t count;
    size_t grain;
    size_t chunks;
    range_function function;
    void *function_context;
} parallel_job;

typedef struct {
    size_t count;
    float *value;
    float *grad;
    float *moment1;
    float *moment2;
    float *ema;
    bool decay;
    const char *name;
} parameter;

typedef struct {
    parameter conv1_w; /* [3*3*1, C1] */
    parameter conv1_b; /* [C1] */
    parameter conv2_w; /* [3*3*C1, C2] */
    parameter conv2_b; /* [C2] */
    parameter dense1_w; /* [7*7*C2, H] */
    parameter dense1_b; /* [H] */
    parameter dense2_w; /* [H, 10] */
    parameter dense2_b; /* [10] */

    parameter *all[8];
    uint32_t step;
    float beta1_power;
    float beta2_power;
} network;

typedef struct {
    size_t max_batch;

    float *input;       /* [B, 28, 28, 1] */
    uint8_t *labels;    /* [B] */

    float *col1;        /* [B*28*28, 3*3] */
    float *act1;        /* [B*28*28, C1] */
    float *pool1;       /* [B*14*14, C1] */
    uint8_t *pool1_mask;

    float *col2;        /* [B*14*14, 3*3*C1] */
    float *act2;        /* [B*14*14, C2] */
    float *pool2;       /* [B, 7*7*C2] */
    uint8_t *pool2_mask;

    float *hidden;      /* [B, H] */
    float *logits;      /* [B, 10], overwritten by probabilities/dlogits */

    float *dhidden;     /* [B, H] */
    float *dpool2;      /* [B, 7*7*C2] */
    float *dact2;       /* [B*14*14, C2] */
    float *dcol2;       /* [B*14*14, 3*3*C1] */
    float *dpool1;      /* [B*14*14, C1] */
    float *dact1;       /* [B*28*28, C1] */

    float *ones;        /* bias-reduction vector, length B*28*28 */
    uint32_t *order;    /* [train count] */
} workspace;

typedef struct {
    double loss;
    uint32_t correct;
} batch_metrics;

typedef struct {
    double loss;
    double accuracy;
} eval_metrics;

static uint64_t global_rng_state = UINT64_C(0x853c49e6748fea9b);

static void fatal(const char *message) {
    fprintf(stderr, "Fatal: %s\n", message);
    exit(EXIT_FAILURE);
}

static void fatal_errno(const char *operation, const char *path) {
    fprintf(stderr, "Fatal: %s '%s': %s\n", operation, path, strerror(errno));
    exit(EXIT_FAILURE);
}

static double now_seconds(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) {
        fatal("clock_gettime failed");
    }
    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static void *aligned_zeroed(size_t count, size_t element_size) {
    if (count != 0u && element_size > SIZE_MAX / count) {
        fatal("allocation size overflow");
    }
    size_t bytes = count * element_size;
    if (bytes == 0u) bytes = 1u;

    void *memory = NULL;
    const int result = posix_memalign(&memory, ALIGNMENT, bytes);
    if (result != 0 || memory == NULL) {
        fprintf(stderr, "Fatal: aligned allocation of %zu bytes failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    memset(memory, 0, bytes);
    return memory;
}

ML_INLINE uint32_t pcg32(void) {
    const uint64_t old_state = global_rng_state;
    global_rng_state = old_state * UINT64_C(6364136223846793005)
                     + UINT64_C(1442695040888963407);
    const uint32_t xorshifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    const uint32_t rotation = (uint32_t)(old_state >> 59u);
    return (xorshifted >> rotation)
         | (xorshifted << ((0u - rotation) & 31u));
}

ML_INLINE float random_uniform(void) {
    return (float)(pcg32() >> 8u) * (1.0f / 16777216.0f);
}

ML_INLINE uint32_t random_bounded(uint32_t bound) {
    return (uint32_t)(((uint64_t)pcg32() * bound) >> 32u);
}

ML_INLINE uint32_t hash32(uint32_t x) {
    x ^= x >> 16u;
    x *= UINT32_C(0x7feb352d);
    x ^= x >> 15u;
    x *= UINT32_C(0x846ca68b);
    x ^= x >> 16u;
    return x;
}

#if defined(ML_APPLE_ACCELERATE)
static void dispatch_range_worker(void *opaque, size_t chunk) {
    parallel_job *job = (parallel_job *)opaque;
    const size_t begin = chunk * job->grain;
    size_t end = begin + job->grain;
    if (end > job->count) end = job->count;
    job->function(begin, end, job->function_context);
}
#endif

static void parallel_for(
    size_t count,
    size_t grain,
    range_function function,
    void *context
) {
    if (count == 0u) return;
    if (grain == 0u) grain = 1u;
    const size_t chunks = (count + grain - 1u) / grain;

#if defined(ML_APPLE_ACCELERATE)
    if (chunks > 1u) {
        parallel_job job = { count, grain, chunks, function, context };
        dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0u);
        dispatch_apply_f(chunks, queue, &job, dispatch_range_worker);
        return;
    }
#endif

    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t begin = chunk * grain;
        size_t end = begin + grain;
        if (end > count) end = count;
        function(begin, end, context);
    }
}

static mapped_file map_file_exact(const char *path, size_t expected_bytes) {
    mapped_file file = {0};
    file.fd = open(path, O_RDONLY);
    if (file.fd < 0) fatal_errno("open", path);

    struct stat status;
    if (fstat(file.fd, &status) != 0) fatal_errno("fstat", path);
    if ((uint64_t)status.st_size != (uint64_t)expected_bytes) {
        fprintf(stderr,
                "Fatal: '%s' is %lld bytes; expected %zu bytes\n",
                path, (long long)status.st_size, expected_bytes);
        exit(EXIT_FAILURE);
    }

    file.address = mmap(NULL, expected_bytes, PROT_READ, MAP_PRIVATE, file.fd, 0);
    if (file.address == MAP_FAILED) fatal_errno("mmap", path);
    file.bytes = expected_bytes;

#if defined(MADV_WILLNEED)
    (void)madvise(file.address, file.bytes, MADV_WILLNEED);
#endif
    return file;
}

static void unmap_file(mapped_file *file) {
    if (file->address != NULL && file->address != MAP_FAILED) {
        (void)munmap(file->address, file->bytes);
    }
    if (file->fd >= 0) (void)close(file->fd);
    file->address = NULL;
    file->bytes = 0u;
    file->fd = -1;
}

/* Row-major C[M,N] = alpha*op(A)[M,K]*op(B)[K,N] + beta*C. */
static void sgemm_fast(
    bool transpose_a,
    bool transpose_b,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    const float *a,
    size_t lda,
    const float *b,
    size_t ldb,
    float beta,
    float *c,
    size_t ldc
) {
#if defined(ML_APPLE_ACCELERATE)
    cblas_sgemm(
        CblasRowMajor,
        transpose_a ? CblasTrans : CblasNoTrans,
        transpose_b ? CblasTrans : CblasNoTrans,
        (int)m, (int)n, (int)k,
        alpha, a, (int)lda, b, (int)ldb, beta, c, (int)ldc
    );
#else
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                const float av = transpose_a ? a[p * lda + i] : a[i * lda + p];
                const float bv = transpose_b ? b[j * ldb + p] : b[p * ldb + j];
                sum += (double)av * bv;
            }
            c[i * ldc + j] = alpha * (float)sum + beta * c[i * ldc + j];
        }
    }
#endif
}

static void bias_gradient(
    size_t rows,
    size_t cols,
    const float *matrix,
    const float *ones,
    float *output
) {
#if defined(ML_APPLE_ACCELERATE)
    cblas_sgemv(
        CblasRowMajor, CblasTrans,
        (int)rows, (int)cols,
        1.0f, matrix, (int)cols,
        ones, 1,
        0.0f, output, 1
    );
#else
    (void)ones;
    memset(output, 0, sizeof(float) * cols);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            output[col] += matrix[row * cols + col];
        }
    }
#endif
}

static double squared_l2(size_t count, const float *values) {
#if defined(ML_APPLE_ACCELERATE)
    return (double)cblas_sdot((int)count, values, 1, values, 1);
#else
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) sum += (double)values[i] * values[i];
    return sum;
#endif
}

static parameter parameter_create(size_t count, bool decay, const char *name) {
    parameter result;
    result.count = count;
    result.value = (float *)aligned_zeroed(count, sizeof(float));
    result.grad = (float *)aligned_zeroed(count, sizeof(float));
    result.moment1 = (float *)aligned_zeroed(count, sizeof(float));
    result.moment2 = (float *)aligned_zeroed(count, sizeof(float));
    result.ema = (float *)aligned_zeroed(count, sizeof(float));
    result.decay = decay;
    result.name = name;
    return result;
}

static void parameter_destroy(parameter *parameter_value) {
    free(parameter_value->value);
    free(parameter_value->grad);
    free(parameter_value->moment1);
    free(parameter_value->moment2);
    free(parameter_value->ema);
    memset(parameter_value, 0, sizeof(*parameter_value));
}

static void initialize_he(parameter *weight, size_t fan_in) {
    const float bound = sqrtf(6.0f / (float)fan_in);
    for (size_t i = 0; i < weight->count; ++i) {
        weight->value[i] = (2.0f * random_uniform() - 1.0f) * bound;
    }
    memcpy(weight->ema, weight->value, sizeof(float) * weight->count);
}

static network network_create(void) {
    network model;
    memset(&model, 0, sizeof(model));

    model.conv1_w = parameter_create(KERNEL_AREA * CONV1_CHANNELS, true, "conv1.weight");
    model.conv1_b = parameter_create(CONV1_CHANNELS, false, "conv1.bias");
    model.conv2_w = parameter_create(
        KERNEL_AREA * CONV1_CHANNELS * CONV2_CHANNELS, true, "conv2.weight");
    model.conv2_b = parameter_create(CONV2_CHANNELS, false, "conv2.bias");
    model.dense1_w = parameter_create(FLAT_FEATURES * HIDDEN_UNITS, true, "dense1.weight");
    model.dense1_b = parameter_create(HIDDEN_UNITS, false, "dense1.bias");
    model.dense2_w = parameter_create(HIDDEN_UNITS * CLASS_COUNT, true, "dense2.weight");
    model.dense2_b = parameter_create(CLASS_COUNT, false, "dense2.bias");

    model.all[0] = &model.conv1_w;
    model.all[1] = &model.conv1_b;
    model.all[2] = &model.conv2_w;
    model.all[3] = &model.conv2_b;
    model.all[4] = &model.dense1_w;
    model.all[5] = &model.dense1_b;
    model.all[6] = &model.dense2_w;
    model.all[7] = &model.dense2_b;

    initialize_he(&model.conv1_w, KERNEL_AREA);
    initialize_he(&model.conv2_w, KERNEL_AREA * CONV1_CHANNELS);
    initialize_he(&model.dense1_w, FLAT_FEATURES);
    initialize_he(&model.dense2_w, HIDDEN_UNITS);
    memcpy(model.conv1_b.ema, model.conv1_b.value, sizeof(float) * model.conv1_b.count);
    memcpy(model.conv2_b.ema, model.conv2_b.value, sizeof(float) * model.conv2_b.count);
    memcpy(model.dense1_b.ema, model.dense1_b.value, sizeof(float) * model.dense1_b.count);
    memcpy(model.dense2_b.ema, model.dense2_b.value, sizeof(float) * model.dense2_b.count);

    model.beta1_power = 1.0f;
    model.beta2_power = 1.0f;
    return model;
}

static void network_destroy(network *model) {
    for (size_t i = 0; i < 8u; ++i) parameter_destroy(model->all[i]);
}

static size_t network_parameter_count(const network *model) {
    size_t count = 0u;
    for (size_t i = 0; i < 8u; ++i) count += model->all[i]->count;
    return count;
}

static void network_swap_ema(network *model) {
    for (size_t i = 0; i < 8u; ++i) {
        parameter *p = model->all[i];
        float *temporary = p->value;
        p->value = p->ema;
        p->ema = temporary;
    }
}

static workspace workspace_create(size_t batch_size, size_t training_count) {
    workspace work;
    memset(&work, 0, sizeof(work));
    work.max_batch = batch_size;

    const size_t rows1 = batch_size * IMAGE_H * IMAGE_W;
    const size_t rows2 = batch_size * POOL1_H * POOL1_W;
    const size_t pool1_count = rows2 * CONV1_CHANNELS;
    const size_t pool2_count = batch_size * FLAT_FEATURES;

    work.input = (float *)aligned_zeroed(batch_size * IMAGE_PIXELS, sizeof(float));
    work.labels = (uint8_t *)aligned_zeroed(batch_size, sizeof(uint8_t));
    work.col1 = (float *)aligned_zeroed(rows1 * KERNEL_AREA, sizeof(float));
    work.act1 = (float *)aligned_zeroed(rows1 * CONV1_CHANNELS, sizeof(float));
    work.pool1 = (float *)aligned_zeroed(pool1_count, sizeof(float));
    work.pool1_mask = (uint8_t *)aligned_zeroed(pool1_count, sizeof(uint8_t));
    work.col2 = (float *)aligned_zeroed(
        rows2 * KERNEL_AREA * CONV1_CHANNELS, sizeof(float));
    work.act2 = (float *)aligned_zeroed(rows2 * CONV2_CHANNELS, sizeof(float));
    work.pool2 = (float *)aligned_zeroed(pool2_count, sizeof(float));
    work.pool2_mask = (uint8_t *)aligned_zeroed(pool2_count, sizeof(uint8_t));
    work.hidden = (float *)aligned_zeroed(batch_size * HIDDEN_UNITS, sizeof(float));
    work.logits = (float *)aligned_zeroed(batch_size * CLASS_COUNT, sizeof(float));

    work.dhidden = (float *)aligned_zeroed(batch_size * HIDDEN_UNITS, sizeof(float));
    work.dpool2 = (float *)aligned_zeroed(pool2_count, sizeof(float));
    work.dact2 = (float *)aligned_zeroed(rows2 * CONV2_CHANNELS, sizeof(float));
    work.dcol2 = (float *)aligned_zeroed(
        rows2 * KERNEL_AREA * CONV1_CHANNELS, sizeof(float));
    work.dpool1 = (float *)aligned_zeroed(pool1_count, sizeof(float));
    work.dact1 = (float *)aligned_zeroed(rows1 * CONV1_CHANNELS, sizeof(float));
    work.ones = (float *)aligned_zeroed(rows1, sizeof(float));
    work.order = (uint32_t *)aligned_zeroed(training_count, sizeof(uint32_t));

    for (size_t i = 0; i < rows1; ++i) work.ones[i] = 1.0f;
    for (size_t i = 0; i < training_count; ++i) work.order[i] = (uint32_t)i;
    return work;
}

static void workspace_destroy(workspace *work) {
    free(work->input);
    free(work->labels);
    free(work->col1);
    free(work->act1);
    free(work->pool1);
    free(work->pool1_mask);
    free(work->col2);
    free(work->act2);
    free(work->pool2);
    free(work->pool2_mask);
    free(work->hidden);
    free(work->logits);
    free(work->dhidden);
    free(work->dpool2);
    free(work->dact2);
    free(work->dcol2);
    free(work->dpool1);
    free(work->dact1);
    free(work->ones);
    free(work->order);
    memset(work, 0, sizeof(*work));
}

static void network_bind_parameters(network *model) {
    model->all[0] = &model->conv1_w;
    model->all[1] = &model->conv1_b;
    model->all[2] = &model->conv2_w;
    model->all[3] = &model->conv2_b;
    model->all[4] = &model->dense1_w;
    model->all[5] = &model->dense1_b;
    model->all[6] = &model->dense2_w;
    model->all[7] = &model->dense2_b;
}

typedef struct {
    float *destination;
    uint8_t *batch_labels;
    const float *images;
    const float *labels;
    const uint32_t *order;
    size_t start;
    uint32_t augmentation_seed;
    bool augment;
} load_batch_context;

static void load_batch_worker(size_t begin, size_t end, void *opaque) {
    load_batch_context *context = (load_batch_context *)opaque;

    for (size_t row = begin; row < end; ++row) {
        const size_t source_index = context->order != NULL
                                  ? context->order[context->start + row]
                                  : context->start + row;
        const float label_value = context->labels[source_index];
        const int label = (int)label_value;
        if (ML_UNLIKELY(label < 0 || label >= (int)CLASS_COUNT
                        || fabsf(label_value - (float)label) > 0.001f)) {
            fprintf(stderr, "Fatal: invalid label %.4f at index %zu\n",
                    label_value, source_index);
            exit(EXIT_FAILURE);
        }
        context->batch_labels[row] = (uint8_t)label;

        const float *source = context->images + source_index * IMAGE_PIXELS;
        float *destination = context->destination + row * IMAGE_PIXELS;
        int dx = 0;
        int dy = 0;

        if (context->augment) {
            const uint32_t bits = hash32(
                context->augmentation_seed ^ (uint32_t)source_index * UINT32_C(0x9e3779b9));
            /* 75% translated; shifts are uniformly selected from [-2,2]^2. */
            if ((bits & 0xffffu) < 49152u) {
                dx = (int)((bits >> 16u) % 5u) - 2;
                dy = (int)((bits >> 24u) % 5u) - 2;
            }
        }

        if (dx == 0 && dy == 0) {
            memcpy(destination, source, sizeof(float) * IMAGE_PIXELS);
            continue;
        }

        for (int y = 0; y < (int)IMAGE_H; ++y) {
            const int source_y = y - dy;
            for (int x = 0; x < (int)IMAGE_W; ++x) {
                const int source_x = x - dx;
                destination[(size_t)y * IMAGE_W + (size_t)x] =
                    source_y >= 0 && source_y < (int)IMAGE_H
                    && source_x >= 0 && source_x < (int)IMAGE_W
                    ? source[(size_t)source_y * IMAGE_W + (size_t)source_x]
                    : 0.0f;
            }
        }
    }
}

static void load_batch(
    workspace *work,
    const float *images,
    const float *labels,
    const uint32_t *order,
    size_t start,
    size_t count,
    uint32_t augmentation_seed,
    bool augment
) {
    load_batch_context context = {
        work->input, work->labels, images, labels, order, start,
        augmentation_seed, augment
    };
    parallel_for(count, 8u, load_batch_worker, &context);
}

typedef struct {
    const float *input;
    float *columns;
    size_t height;
    size_t width;
    size_t channels;
} im2col_context;

/* Padding-one, stride-one, 3x3 im2col for NHWC tensors. */
static void im2col_worker(size_t begin, size_t end, void *opaque) {
    im2col_context *context = (im2col_context *)opaque;
    const size_t spatial = context->height * context->width;
    const size_t patch_size = KERNEL_AREA * context->channels;

    for (size_t row = begin; row < end; ++row) {
        const size_t batch = row / spatial;
        const size_t position = row - batch * spatial;
        const int output_y = (int)(position / context->width);
        const int output_x = (int)(position % context->width);
        float *patch = context->columns + row * patch_size;

        for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
            const int input_y = output_y + kernel_y - 1;
            for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                const int input_x = output_x + kernel_x - 1;
                float *slot = patch
                    + (size_t)(kernel_y * 3 + kernel_x) * context->channels;

                if (input_y >= 0 && input_y < (int)context->height
                    && input_x >= 0 && input_x < (int)context->width) {
                    const float *pixel = context->input
                        + ((batch * context->height + (size_t)input_y) * context->width
                           + (size_t)input_x) * context->channels;
                    memcpy(slot, pixel, sizeof(float) * context->channels);
                } else {
                    memset(slot, 0, sizeof(float) * context->channels);
                }
            }
        }
    }
}

static void im2col_same3x3(
    const float *input,
    size_t batch,
    size_t height,
    size_t width,
    size_t channels,
    float *columns
) {
    im2col_context context = { input, columns, height, width, channels };
    parallel_for(batch * height * width, 512u, im2col_worker, &context);
}

typedef struct {
    float *matrix;
    const float *bias;
    size_t cols;
    bool relu;
} bias_context;

static void bias_worker(size_t begin, size_t end, void *opaque) {
    bias_context *context = (bias_context *)opaque;
    const size_t cols = context->cols;

    for (size_t row = begin; row < end; ++row) {
        float *values = context->matrix + row * cols;
        size_t col = 0u;

#if defined(ML_ARM_NEON)
        const float32x4_t zero = vdupq_n_f32(0.0f);
        for (; col + 4u <= cols; col += 4u) {
            float32x4_t result = vaddq_f32(
                vld1q_f32(values + col), vld1q_f32(context->bias + col));
            if (context->relu) result = vmaxq_f32(result, zero);
            vst1q_f32(values + col, result);
        }
#endif
        for (; col < cols; ++col) {
            float result = values[col] + context->bias[col];
            values[col] = context->relu && result < 0.0f ? 0.0f : result;
        }
    }
}

static void add_bias_inplace(
    float *matrix,
    size_t rows,
    size_t cols,
    const float *bias,
    bool relu
) {
    bias_context context = { matrix, bias, cols, relu };
    parallel_for(rows, 512u, bias_worker, &context);
}

typedef struct {
    const float *input;
    float *output;
    uint8_t *mask;
    size_t height;
    size_t width;
    size_t channels;
} pool_context;

static void maxpool_worker(size_t begin, size_t end, void *opaque) {
    pool_context *context = (pool_context *)opaque;
    const size_t output_h = context->height / 2u;
    const size_t output_w = context->width / 2u;
    const size_t output_spatial = output_h * output_w;

    for (size_t index = begin; index < end; ++index) {
        const size_t channel = index % context->channels;
        const size_t tmp = index / context->channels;
        const size_t output_pos = tmp % output_spatial;
        const size_t batch = tmp / output_spatial;
        const size_t output_y = output_pos / output_w;
        const size_t output_x = output_pos % output_w;
        const size_t input_y = output_y * 2u;
        const size_t input_x = output_x * 2u;

        const size_t base = ((batch * context->height + input_y) * context->width
                            + input_x) * context->channels + channel;
        const size_t row_stride = context->width * context->channels;
        const float candidates[4] = {
            context->input[base],
            context->input[base + context->channels],
            context->input[base + row_stride],
            context->input[base + row_stride + context->channels]
        };

        uint8_t winner = 0u;
        if (candidates[1] > candidates[winner]) winner = 1u;
        if (candidates[2] > candidates[winner]) winner = 2u;
        if (candidates[3] > candidates[winner]) winner = 3u;
        context->output[index] = candidates[winner];
        context->mask[index] = winner;
    }
}

static void maxpool2_forward(
    const float *input,
    size_t batch,
    size_t height,
    size_t width,
    size_t channels,
    float *output,
    uint8_t *mask
) {
    pool_context context = { input, output, mask, height, width, channels };
    const size_t output_count = batch * (height / 2u) * (width / 2u) * channels;
    parallel_for(output_count, 4096u, maxpool_worker, &context);
}

typedef struct {
    const float *pool_grad;
    const uint8_t *mask;
    const float *activation;
    float *activation_grad;
    size_t height;
    size_t width;
    size_t channels;
} pool_backward_context;

/* Gather formulation: every input cell reads one non-overlapping pool result, so no atomics. */
static void maxpool_backward_worker(size_t begin, size_t end, void *opaque) {
    pool_backward_context *context = (pool_backward_context *)opaque;
    const size_t sample_stride = context->height * context->width * context->channels;
    const size_t output_w = context->width / 2u;
    const size_t output_spatial = (context->height / 2u) * output_w;

    for (size_t index = begin; index < end; ++index) {
        const size_t batch = index / sample_stride;
        const size_t local = index - batch * sample_stride;
        const size_t channel = local % context->channels;
        const size_t pixel = local / context->channels;
        const size_t y = pixel / context->width;
        const size_t x = pixel % context->width;
        const size_t output_index =
            ((batch * output_spatial + (y / 2u) * output_w + x / 2u)
             * context->channels) + channel;
        const uint8_t local_position = (uint8_t)((y & 1u) * 2u + (x & 1u));

        context->activation_grad[index] =
            context->mask[output_index] == local_position
            && context->activation[index] > 0.0f
            ? context->pool_grad[output_index]
            : 0.0f;
    }
}

static void maxpool2_backward_relu(
    const float *pool_grad,
    const uint8_t *mask,
    const float *activation,
    size_t batch,
    size_t height,
    size_t width,
    size_t channels,
    float *activation_grad
) {
    pool_backward_context context = {
        pool_grad, mask, activation, activation_grad, height, width, channels
    };
    parallel_for(batch * height * width * channels,
                 8192u, maxpool_backward_worker, &context);
}

typedef struct {
    const float *columns;
    float *input_grad;
    size_t height;
    size_t width;
    size_t channels;
} col2im_context;

/* Gather the nine contributing column entries for each NHWC input cell. */
static void col2im_worker(size_t begin, size_t end, void *opaque) {
    col2im_context *context = (col2im_context *)opaque;
    const size_t sample_stride = context->height * context->width * context->channels;
    const size_t patch_size = KERNEL_AREA * context->channels;

    for (size_t index = begin; index < end; ++index) {
        const size_t batch = index / sample_stride;
        const size_t local = index - batch * sample_stride;
        const size_t channel = local % context->channels;
        const size_t pixel = local / context->channels;
        const int input_y = (int)(pixel / context->width);
        const int input_x = (int)(pixel % context->width);
        float sum = 0.0f;

        for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
            const int output_y = input_y - kernel_y + 1;
            if (output_y < 0 || output_y >= (int)context->height) continue;
            for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                const int output_x = input_x - kernel_x + 1;
                if (output_x < 0 || output_x >= (int)context->width) continue;

                const size_t row = (batch * context->height + (size_t)output_y)
                                 * context->width + (size_t)output_x;
                const size_t column = (size_t)(kernel_y * 3 + kernel_x)
                                    * context->channels + channel;
                sum += context->columns[row * patch_size + column];
            }
        }
        context->input_grad[index] = sum;
    }
}

static void col2im_same3x3(
    const float *columns,
    size_t batch,
    size_t height,
    size_t width,
    size_t channels,
    float *input_grad
) {
    col2im_context context = { columns, input_grad, height, width, channels };
    parallel_for(batch * height * width * channels, 8192u, col2im_worker, &context);
}

typedef struct {
    float *gradient;
    const float *activation;
} relu_backward_context;

static void relu_backward_worker(size_t begin, size_t end, void *opaque) {
    relu_backward_context *context = (relu_backward_context *)opaque;
    size_t i = begin;

#if defined(ML_ARM_NEON)
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4u <= end; i += 4u) {
        const float32x4_t activation = vld1q_f32(context->activation + i);
        const uint32x4_t positive = vcgtq_f32(activation, zero);
        const float32x4_t gradient = vld1q_f32(context->gradient + i);
        vst1q_f32(context->gradient + i,
                  vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(gradient), positive)));
    }
#endif
    for (; i < end; ++i) {
        if (context->activation[i] <= 0.0f) context->gradient[i] = 0.0f;
    }
}

static void relu_backward_inplace(float *gradient, const float *activation, size_t count) {
    relu_backward_context context = { gradient, activation };
    parallel_for(count, 16384u, relu_backward_worker, &context);
}

typedef struct {
    float *logits;
    size_t classes;
} softmax_context;

static void softmax_worker(size_t begin, size_t end, void *opaque) {
    softmax_context *context = (softmax_context *)opaque;
    for (size_t row = begin; row < end; ++row) {
        float *values = context->logits + row * context->classes;
        float maximum = values[0];
        for (size_t col = 1; col < context->classes; ++col) {
            if (values[col] > maximum) maximum = values[col];
        }
        float sum = 0.0f;
        for (size_t col = 0; col < context->classes; ++col) {
            values[col] = expf(values[col] - maximum);
            sum += values[col];
        }
        const float inverse = 1.0f / sum;
        for (size_t col = 0; col < context->classes; ++col) values[col] *= inverse;
    }
}

static void softmax_inplace(float *logits, size_t rows) {
    softmax_context context = { logits, CLASS_COUNT };
    parallel_for(rows, 64u, softmax_worker, &context);
}

static batch_metrics calculate_metrics(
    const float *probabilities,
    const uint8_t *labels,
    size_t rows,
    float label_smoothing
) {
    batch_metrics result = {0.0, 0u};
    const float smooth = label_smoothing / (float)CLASS_COUNT;

    for (size_t row = 0; row < rows; ++row) {
        const float *probability = probabilities + row * CLASS_COUNT;
        uint32_t best = 0u;
        double log_sum = 0.0;
        for (uint32_t col = 0; col < CLASS_COUNT; ++col) {
            const float p = probability[col] > SOFTMAX_FLOOR
                          ? probability[col] : SOFTMAX_FLOOR;
            log_sum += log((double)p);
            if (probability[col] > probability[best]) best = col;
        }
        const float true_probability = probability[labels[row]] > SOFTMAX_FLOOR
                                     ? probability[labels[row]] : SOFTMAX_FLOOR;
        result.loss += -(double)(1.0f - label_smoothing) * log((double)true_probability)
                       -(double)smooth * log_sum;
        result.correct += best == labels[row];
    }
    return result;
}

typedef struct {
    float *probability_and_gradient;
    const uint8_t *labels;
    float smoothing;
    float inverse_batch;
} xent_backward_context;

static void xent_backward_worker(size_t begin, size_t end, void *opaque) {
    xent_backward_context *context = (xent_backward_context *)opaque;
    const float off_target = context->smoothing / (float)CLASS_COUNT;
    const float true_extra = 1.0f - context->smoothing;

    for (size_t row = begin; row < end; ++row) {
        float *gradient = context->probability_and_gradient + row * CLASS_COUNT;
        const uint32_t label = context->labels[row];
        for (uint32_t col = 0; col < CLASS_COUNT; ++col) {
            const float target = off_target + (col == label ? true_extra : 0.0f);
            gradient[col] = (gradient[col] - target) * context->inverse_batch;
        }
    }
}

static void softmax_cross_entropy_backward_inplace(
    float *probabilities,
    const uint8_t *labels,
    size_t rows,
    float smoothing
) {
    xent_backward_context context = {
        probabilities, labels, smoothing, 1.0f / (float)rows
    };
    parallel_for(rows, 64u, xent_backward_worker, &context);
}

static batch_metrics network_forward(
    network *model,
    workspace *work,
    size_t batch,
    float label_smoothing
) {
    const size_t rows1 = batch * IMAGE_H * IMAGE_W;
    const size_t rows2 = batch * POOL1_H * POOL1_W;
    const size_t kernel2 = KERNEL_AREA * CONV1_CHANNELS;

    im2col_same3x3(work->input, batch, IMAGE_H, IMAGE_W, 1u, work->col1);
    sgemm_fast(false, false,
               rows1, CONV1_CHANNELS, KERNEL_AREA,
               1.0f, work->col1, KERNEL_AREA,
               model->conv1_w.value, CONV1_CHANNELS,
               0.0f, work->act1, CONV1_CHANNELS);
    add_bias_inplace(work->act1, rows1, CONV1_CHANNELS, model->conv1_b.value, true);

    maxpool2_forward(work->act1, batch, IMAGE_H, IMAGE_W, CONV1_CHANNELS,
                     work->pool1, work->pool1_mask);

    im2col_same3x3(work->pool1, batch, POOL1_H, POOL1_W,
                   CONV1_CHANNELS, work->col2);
    sgemm_fast(false, false,
               rows2, CONV2_CHANNELS, kernel2,
               1.0f, work->col2, kernel2,
               model->conv2_w.value, CONV2_CHANNELS,
               0.0f, work->act2, CONV2_CHANNELS);
    add_bias_inplace(work->act2, rows2, CONV2_CHANNELS, model->conv2_b.value, true);

    maxpool2_forward(work->act2, batch, POOL1_H, POOL1_W, CONV2_CHANNELS,
                     work->pool2, work->pool2_mask);

    sgemm_fast(false, false,
               batch, HIDDEN_UNITS, FLAT_FEATURES,
               1.0f, work->pool2, FLAT_FEATURES,
               model->dense1_w.value, HIDDEN_UNITS,
               0.0f, work->hidden, HIDDEN_UNITS);
    add_bias_inplace(work->hidden, batch, HIDDEN_UNITS, model->dense1_b.value, true);

    sgemm_fast(false, false,
               batch, CLASS_COUNT, HIDDEN_UNITS,
               1.0f, work->hidden, HIDDEN_UNITS,
               model->dense2_w.value, CLASS_COUNT,
               0.0f, work->logits, CLASS_COUNT);
    add_bias_inplace(work->logits, batch, CLASS_COUNT, model->dense2_b.value, false);
    softmax_inplace(work->logits, batch);
    return calculate_metrics(work->logits, work->labels, batch, label_smoothing);
}

static void network_backward(
    network *model,
    workspace *work,
    size_t batch,
    float label_smoothing
) {
    const size_t rows1 = batch * IMAGE_H * IMAGE_W;
    const size_t rows2 = batch * POOL1_H * POOL1_W;
    const size_t kernel2 = KERNEL_AREA * CONV1_CHANNELS;

    /* logits currently contains probabilities; overwrite it with dL/dlogits. */
    softmax_cross_entropy_backward_inplace(
        work->logits, work->labels, batch, label_smoothing);

    sgemm_fast(true, false,
               HIDDEN_UNITS, CLASS_COUNT, batch,
               1.0f, work->hidden, HIDDEN_UNITS,
               work->logits, CLASS_COUNT,
               0.0f, model->dense2_w.grad, CLASS_COUNT);
    bias_gradient(batch, CLASS_COUNT, work->logits, work->ones, model->dense2_b.grad);

    sgemm_fast(false, true,
               batch, HIDDEN_UNITS, CLASS_COUNT,
               1.0f, work->logits, CLASS_COUNT,
               model->dense2_w.value, CLASS_COUNT,
               0.0f, work->dhidden, HIDDEN_UNITS);
    relu_backward_inplace(work->dhidden, work->hidden, batch * HIDDEN_UNITS);

    sgemm_fast(true, false,
               FLAT_FEATURES, HIDDEN_UNITS, batch,
               1.0f, work->pool2, FLAT_FEATURES,
               work->dhidden, HIDDEN_UNITS,
               0.0f, model->dense1_w.grad, HIDDEN_UNITS);
    bias_gradient(batch, HIDDEN_UNITS, work->dhidden, work->ones, model->dense1_b.grad);

    sgemm_fast(false, true,
               batch, FLAT_FEATURES, HIDDEN_UNITS,
               1.0f, work->dhidden, HIDDEN_UNITS,
               model->dense1_w.value, HIDDEN_UNITS,
               0.0f, work->dpool2, FLAT_FEATURES);

    maxpool2_backward_relu(
        work->dpool2, work->pool2_mask, work->act2,
        batch, POOL1_H, POOL1_W, CONV2_CHANNELS, work->dact2);

    sgemm_fast(true, false,
               kernel2, CONV2_CHANNELS, rows2,
               1.0f, work->col2, kernel2,
               work->dact2, CONV2_CHANNELS,
               0.0f, model->conv2_w.grad, CONV2_CHANNELS);
    bias_gradient(rows2, CONV2_CHANNELS,
                  work->dact2, work->ones, model->conv2_b.grad);

    sgemm_fast(false, true,
               rows2, kernel2, CONV2_CHANNELS,
               1.0f, work->dact2, CONV2_CHANNELS,
               model->conv2_w.value, CONV2_CHANNELS,
               0.0f, work->dcol2, kernel2);
    col2im_same3x3(work->dcol2, batch, POOL1_H, POOL1_W,
                   CONV1_CHANNELS, work->dpool1);

    maxpool2_backward_relu(
        work->dpool1, work->pool1_mask, work->act1,
        batch, IMAGE_H, IMAGE_W, CONV1_CHANNELS, work->dact1);

    sgemm_fast(true, false,
               KERNEL_AREA, CONV1_CHANNELS, rows1,
               1.0f, work->col1, KERNEL_AREA,
               work->dact1, CONV1_CHANNELS,
               0.0f, model->conv1_w.grad, CONV1_CHANNELS);
    bias_gradient(rows1, CONV1_CHANNELS,
                  work->dact1, work->ones, model->conv1_b.grad);

    /* No d(input): it cannot affect trainable parameters and would waste a GEMM+col2im. */
}

typedef struct {
    parameter *parameter_value;
    float learning_rate;
    float clip_scale;
    float inverse_bias1;
    float inverse_bias2;
    float weight_decay;
    float ema_decay;
} adam_context;

static void adam_worker(size_t begin, size_t end, void *opaque) {
    adam_context *context = (adam_context *)opaque;
    parameter *p = context->parameter_value;
    size_t i = begin;

#if defined(ML_ARM_NEON)
    const float32x4_t beta1 = vdupq_n_f32(ADAM_BETA1);
    const float32x4_t beta2 = vdupq_n_f32(ADAM_BETA2);
    const float32x4_t one_minus_beta1 = vdupq_n_f32(1.0f - ADAM_BETA1);
    const float32x4_t one_minus_beta2 = vdupq_n_f32(1.0f - ADAM_BETA2);
    const float32x4_t clip = vdupq_n_f32(context->clip_scale);
    const float32x4_t inverse_bias1 = vdupq_n_f32(context->inverse_bias1);
    const float32x4_t inverse_bias2 = vdupq_n_f32(context->inverse_bias2);
    const float32x4_t epsilon = vdupq_n_f32(ADAM_EPSILON);
    const float32x4_t learning_rate = vdupq_n_f32(context->learning_rate);
    const float32x4_t weight_decay = vdupq_n_f32(
        p->decay ? context->weight_decay : 0.0f);
    const float32x4_t ema_decay = vdupq_n_f32(context->ema_decay);
    const float32x4_t one_minus_ema = vdupq_n_f32(1.0f - context->ema_decay);

    for (; i + 4u <= end; i += 4u) {
        const float32x4_t gradient = vmulq_f32(vld1q_f32(p->grad + i), clip);
        float32x4_t first = vld1q_f32(p->moment1 + i);
        float32x4_t second = vld1q_f32(p->moment2 + i);
        float32x4_t value = vld1q_f32(p->value + i);
        float32x4_t ema = vld1q_f32(p->ema + i);

        first = vaddq_f32(vmulq_f32(beta1, first),
                          vmulq_f32(one_minus_beta1, gradient));
        second = vaddq_f32(vmulq_f32(beta2, second),
                           vmulq_f32(one_minus_beta2, vmulq_f32(gradient, gradient)));

        const float32x4_t m_hat = vmulq_f32(first, inverse_bias1);
        const float32x4_t v_hat = vaddq_f32(vmulq_f32(second, inverse_bias2), epsilon);
        float32x4_t inverse_sqrt = vrsqrteq_f32(v_hat);
        inverse_sqrt = vmulq_f32(
            inverse_sqrt,
            vrsqrtsq_f32(vmulq_f32(v_hat, inverse_sqrt), inverse_sqrt));
        inverse_sqrt = vmulq_f32(
            inverse_sqrt,
            vrsqrtsq_f32(vmulq_f32(v_hat, inverse_sqrt), inverse_sqrt));

        const float32x4_t update = vaddq_f32(
            vmulq_f32(m_hat, inverse_sqrt), vmulq_f32(weight_decay, value));
        value = vsubq_f32(value, vmulq_f32(learning_rate, update));
        ema = vaddq_f32(vmulq_f32(ema_decay, ema),
                        vmulq_f32(one_minus_ema, value));

        vst1q_f32(p->moment1 + i, first);
        vst1q_f32(p->moment2 + i, second);
        vst1q_f32(p->value + i, value);
        vst1q_f32(p->ema + i, ema);
    }
#endif

    for (; i < end; ++i) {
        const float gradient = p->grad[i] * context->clip_scale;
        p->moment1[i] = ADAM_BETA1 * p->moment1[i]
                      + (1.0f - ADAM_BETA1) * gradient;
        p->moment2[i] = ADAM_BETA2 * p->moment2[i]
                      + (1.0f - ADAM_BETA2) * gradient * gradient;
        const float m_hat = p->moment1[i] * context->inverse_bias1;
        const float v_hat = p->moment2[i] * context->inverse_bias2;
        const float decay = p->decay ? context->weight_decay * p->value[i] : 0.0f;
        p->value[i] -= context->learning_rate
                     * (m_hat / sqrtf(v_hat + ADAM_EPSILON) + decay);
        p->ema[i] = context->ema_decay * p->ema[i]
                  + (1.0f - context->ema_decay) * p->value[i];
    }
}

static void network_adamw_step(
    network *model,
    float learning_rate,
    float weight_decay
) {
    double squared_norm = 0.0;
    for (size_t i = 0; i < 8u; ++i) {
        squared_norm += squared_l2(model->all[i]->count, model->all[i]->grad);
    }
    const float norm = (float)sqrt(squared_norm);
    const float clip_scale = norm > GRAD_CLIP ? GRAD_CLIP / norm : 1.0f;

    ++model->step;
    model->beta1_power *= ADAM_BETA1;
    model->beta2_power *= ADAM_BETA2;
    const float inverse_bias1 = 1.0f / (1.0f - model->beta1_power);
    const float inverse_bias2 = 1.0f / (1.0f - model->beta2_power);
    const float warm_ema = 1.0f - 1.0f / (float)(model->step + 10u);
    const float ema_decay = warm_ema < 0.999f ? warm_ema : 0.999f;

    for (size_t i = 0; i < 8u; ++i) {
        parameter *p = model->all[i];
        adam_context context = {
            p, learning_rate, clip_scale, inverse_bias1, inverse_bias2,
            weight_decay, ema_decay
        };
        const size_t grain = p->count >= 65536u ? 16384u : p->count;
        parallel_for(p->count, grain, adam_worker, &context);
    }
}

static float cosine_learning_rate(
    float maximum,
    float minimum,
    uint32_t step,
    uint32_t warmup_steps,
    uint32_t total_steps
) {
    if (warmup_steps > 0u && step <= warmup_steps) {
        return maximum * (float)step / (float)warmup_steps;
    }
    const uint32_t decay_steps = total_steps > warmup_steps
                               ? total_steps - warmup_steps : 1u;
    float progress = (float)(step - warmup_steps) / (float)decay_steps;
    if (progress > 1.0f) progress = 1.0f;
    const float cosine = 0.5f * (1.0f + cosf(PI_F * progress));
    return minimum + (maximum - minimum) * cosine;
}

static void shuffle_order(uint32_t *order, size_t count) {
    for (size_t i = count - 1u; i > 0u; --i) {
        const size_t j = random_bounded((uint32_t)(i + 1u));
        const uint32_t temporary = order[i];
        order[i] = order[j];
        order[j] = temporary;
    }
}

static eval_metrics evaluate(
    network *model,
    workspace *work,
    const float *images,
    const float *labels,
    size_t count
) {
    double total_loss = 0.0;
    uint64_t total_correct = 0u;

    for (size_t start = 0; start < count; start += work->max_batch) {
        const size_t remaining = count - start;
        const size_t batch = remaining < work->max_batch ? remaining : work->max_batch;
        load_batch(work, images, labels, NULL, start, batch, 0u, false);
        const batch_metrics metrics = network_forward(model, work, batch, 0.0f);
        total_loss += metrics.loss;
        total_correct += metrics.correct;
    }

    eval_metrics result;
    result.loss = total_loss / (double)count;
    result.accuracy = (double)total_correct / (double)count;
    return result;
}

static void train(
    network *model,
    workspace *work,
    const float *train_images,
    const float *train_labels,
    const float *test_images,
    const float *test_labels
) {
    const size_t train_count = MNIST_TRAIN_COUNT;
    const size_t batch_size = work->max_batch;
    const uint32_t batches_per_epoch = (uint32_t)((train_count + batch_size - 1u) / batch_size);
    const uint32_t total_steps = TRAIN_EPOCHS * batches_per_epoch;
    const uint32_t warmup_steps = batches_per_epoch / 2u;
    const float maximum_lr = 1.5e-3f;
    const float minimum_lr = 2.0e-5f;
    const float weight_decay = 1.0e-4f;
    const float label_smoothing = 0.03f;

    printf("\nTraining %u epochs, batch %zu, %u updates/epoch\n",
           TRAIN_EPOCHS, batch_size, batches_per_epoch);
    printf("AdamW + global clipping + cosine decay + label smoothing + EMA\n\n");

    for (uint32_t epoch = 0; epoch < TRAIN_EPOCHS; ++epoch) {
        const double epoch_start = now_seconds();
        double train_loss = 0.0;
        uint64_t train_correct = 0u;
        float learning_rate = maximum_lr;
        shuffle_order(work->order, train_count);

        uint32_t batch_number = 0u;
        for (size_t start = 0; start < train_count; start += batch_size) {
            const size_t remaining = train_count - start;
            const size_t batch = remaining < batch_size ? remaining : batch_size;
            const uint32_t augmentation_seed = hash32(
                (epoch + 1u) * UINT32_C(0x9e3779b9) ^ batch_number);

            load_batch(work, train_images, train_labels, work->order,
                       start, batch, augmentation_seed, true);
            const batch_metrics metrics = network_forward(
                model, work, batch, label_smoothing);
            train_loss += metrics.loss;
            train_correct += metrics.correct;

            network_backward(model, work, batch, label_smoothing);
            learning_rate = cosine_learning_rate(
                maximum_lr, minimum_lr, model->step + 1u,
                warmup_steps, total_steps);
            network_adamw_step(model, learning_rate, weight_decay);

            ++batch_number;
            const uint32_t report_interval = batches_per_epoch >= 10u
                                           ? batches_per_epoch / 10u : 1u;
            if (batch_number % report_interval == 0u
                || batch_number == batches_per_epoch) {
                printf("Epoch %2u/%2u | batch %3u/%3u | loss %.4f | lr %.6f\r",
                       epoch + 1u, TRAIN_EPOCHS, batch_number, batches_per_epoch,
                       train_loss / (double)(start + batch), learning_rate);
                fflush(stdout);
            }
        }

        network_swap_ema(model);
        const eval_metrics test = evaluate(
            model, work, test_images, test_labels, MNIST_TEST_COUNT);
        network_swap_ema(model);

        const double elapsed = now_seconds() - epoch_start;
        printf("\n  train %.2f%%, loss %.4f | test EMA %.2f%%, loss %.4f"
               " | %.2fs | %.0f images/s\n",
               100.0 * (double)train_correct / (double)train_count,
               train_loss / (double)train_count,
               100.0 * test.accuracy, test.loss,
               elapsed, (double)(train_count + MNIST_TEST_COUNT) / elapsed);
    }

    /* The final model uses the smoothed weights permanently. */
    network_swap_ema(model);
}

static void configure_apple_runtime(void) {
#if defined(ML_APPLE_ACCELERATE)
    /* Both frameworks select their worker counts from the active hardware automatically. */
    printf("Apple Accelerate + GCD fast path enabled\n");
#else
    printf("Portable scalar validation path (Apple Accelerate unavailable)\n");
#endif
}

int main(void) {
    configure_apple_runtime();

    mapped_file train_image_file = map_file_exact(
        "train_images.mat", (size_t)MNIST_TRAIN_COUNT * IMAGE_PIXELS * sizeof(float));
    mapped_file train_label_file = map_file_exact(
        "train_labels.mat", (size_t)MNIST_TRAIN_COUNT * sizeof(float));
    mapped_file test_image_file = map_file_exact(
        "test_images.mat", (size_t)MNIST_TEST_COUNT * IMAGE_PIXELS * sizeof(float));
    mapped_file test_label_file = map_file_exact(
        "test_labels.mat", (size_t)MNIST_TEST_COUNT * sizeof(float));

    network model = network_create();
    /* network_create returns by value, so rebind internal pointers to this final object. */
    network_bind_parameters(&model);
    workspace work = workspace_create(TRAIN_BATCH_SIZE, MNIST_TRAIN_COUNT);

    const size_t parameters = network_parameter_count(&model);
    printf("CNN: 28x28 -> Conv%u -> Pool -> Conv%u -> Pool -> Dense%u -> 10\n",
           CONV1_CHANNELS, CONV2_CHANNELS, HIDDEN_UNITS);
    printf("Trainable parameters: %zu\n", parameters);
    printf("Persistent training workspace: approximately %.1f MiB\n",
           (double)(
               TRAIN_BATCH_SIZE * IMAGE_PIXELS
               + TRAIN_BATCH_SIZE * IMAGE_H * IMAGE_W * KERNEL_AREA
               + 2u * TRAIN_BATCH_SIZE * IMAGE_H * IMAGE_W * CONV1_CHANNELS
               + 3u * TRAIN_BATCH_SIZE * POOL1_H * POOL1_W * CONV1_CHANNELS
               + 2u * TRAIN_BATCH_SIZE * POOL1_H * POOL1_W
                    * KERNEL_AREA * CONV1_CHANNELS
               + 2u * TRAIN_BATCH_SIZE * POOL1_H * POOL1_W * CONV2_CHANNELS
               + 3u * TRAIN_BATCH_SIZE * FLAT_FEATURES
               + 2u * TRAIN_BATCH_SIZE * HIDDEN_UNITS
               + TRAIN_BATCH_SIZE * CLASS_COUNT
           ) * sizeof(float) / (1024.0 * 1024.0));

    train(
        &model, &work,
        (const float *)train_image_file.address,
        (const float *)train_label_file.address,
        (const float *)test_image_file.address,
        (const float *)test_label_file.address
    );

    const eval_metrics final_test = evaluate(
        &model, &work,
        (const float *)test_image_file.address,
        (const float *)test_label_file.address,
        MNIST_TEST_COUNT
    );
    printf("\nFinal EMA test accuracy: %.2f%% | cross-entropy: %.4f\n",
           100.0 * final_test.accuracy, final_test.loss);

    workspace_destroy(&work);
    network_destroy(&model);
    unmap_file(&train_image_file);
    unmap_file(&train_label_file);
    unmap_file(&test_image_file);
    unmap_file(&test_label_file);
    return 0;
}
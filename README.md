# Neural Network and Autodiff Engine in C

A neural-network framework written from scratch in C and trained on MNIST without external machine-learning libraries.

The project implements the core components normally handled by frameworks such as PyTorch:

- Matrix operations
- Computation graph construction
- Reverse-mode automatic differentiation
- Forward and backward propagation
- Mini-batch stochastic gradient descent
- Xavier weight initialization
- ReLU, softmax, and cross-entropy
- Arena-based memory management
- MNIST training and evaluation

The final model reaches **94.0% test accuracy** after 20 epochs.

## Model Architecture

Each `28 x 28` MNIST image is flattened into a `784 x 1` vector.

```text
h0    = ReLU(W0 * x + b0)
r1    = ReLU(W1 * h0 + b1)
h1    = h0 + r1
y_hat = Softmax(W2 * h1 + b2)
```

| Layer | Weight shape | Output shape |
| --- | ---: | ---: |
| Input | — | `784 x 1` |
| Hidden layer | `16 x 784` | `16 x 1` |
| Residual layer | `16 x 16` | `16 x 1` |
| Output layer | `10 x 16` | `10 x 1` |

The second hidden layer uses a residual connection:

```text
h1 = h0 + ReLU(W1 * h0 + b1)
```

Weights are initialized using Xavier uniform initialization:

```text
limit = sqrt(6 / (fan_in + fan_out))
W[i][j] ~ Uniform(-limit, limit)
```

## Implementation

### Matrix Operations

Matrices are stored as contiguous row-major arrays:

```c
typedef struct {
    u32 rows;
    u32 cols;
    f32 *data;
} matrix;
```

The matrix layer implements:

- Addition and subtraction
- Scalar multiplication
- Matrix multiplication
- Transposed matrix multiplication
- ReLU
- Numerically stable softmax
- Cross-entropy
- Gradient accumulation
- Summation and `argmax`

Four matrix-multiplication kernels support all transpose combinations:

```text
A * B
A * transpose(B)
transpose(A) * B
transpose(A) * transpose(B)
```

These variants are used during both forward propagation and backpropagation.

### Computation Graph

Every value is represented by a `model_var` containing:

- Its matrix value
- Its gradient
- The operation that produced it
- Its input nodes
- Flags identifying parameters, inputs, outputs, and costs

The graph is converted into a topologically ordered program using an iterative depth-first traversal.

Forward execution processes this program from inputs to output. Backpropagation traverses it in reverse and accumulates gradients into every trainable parameter.

For matrix multiplication:

```text
C = A * B
G = dL/dC

dL/dA = G * transpose(B)
dL/dB = transpose(A) * G
```

Gradient accumulation also allows the engine to correctly differentiate through the model's residual connection.

## Training

The model is trained using mini-batch stochastic gradient descent.

| Setting | Value |
| --- | ---: |
| Training examples | 60,000 |
| Test examples | 10,000 |
| Epochs | 20 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Batches per epoch | 937 |
| Optimizer | SGD |

For each batch, the program:

1. Clears the parameter gradients.
2. Runs a forward pass for each example.
3. Computes cross-entropy loss.
4. Runs reverse-mode autodiff.
5. Accumulates gradients over the batch.
6. Applies the averaged SGD update.

```text
parameter = parameter - learning_rate * gradient / batch_size
```

## Training Results

The network improved from effectively random predictions to **94.0% test accuracy**.

| Epoch | Final-batch cost | Test accuracy | Test cost |
| ---: | ---: | ---: | ---: |
| 1 | 0.6355 | 84.4% | 0.5729 |
| 2 | 0.4309 | 89.1% | 0.3837 |
| 3 | 0.2119 | 90.5% | 0.3297 |
| 4 | 0.4791 | 91.1% | 0.3044 |
| 5 | 0.3551 | 91.5% | 0.2889 |
| 6 | 0.3089 | 92.0% | 0.2761 |
| 7 | 0.1696 | 92.4% | 0.2658 |
| 8 | 0.3314 | 92.5% | 0.2563 |
| 9 | 0.4121 | 92.8% | 0.2530 |
| 10 | 0.2205 | 93.0% | 0.2464 |
| 11 | 0.2264 | 93.1% | 0.2389 |
| 12 | 0.1940 | 93.0% | 0.2390 |
| 13 | 0.2686 | 93.3% | 0.2311 |
| 14 | 0.3268 | 93.5% | 0.2256 |
| 15 | 0.1078 | 93.6% | 0.2243 |
| 16 | 0.0415 | 93.5% | 0.2218 |
| 17 | 0.1373 | 93.6% | 0.2170 |
| 18 | 0.2975 | 93.9% | 0.2147 |
| 19 | 0.0569 | **94.0%** | 0.2094 |
| 20 | 0.2338 | **94.0%** | **0.2072** |

The final-batch cost fluctuates because it describes only the last shuffled batch of each epoch. The test cost is the more stable metric and decreased from `0.5729` to `0.2072`.

### Example Prediction

Before training, the model assigned scattered probabilities to the first test image:

```text
0.06 0.23 0.01 0.12 0.06 0.01 0.04 0.38 0.03 0.06
```

After training:

```text
0.000016 0.000000 0.999230 0.000525 0.000006
0.000174 0.000001 0.000000 0.000035 0.000013
```

The trained network assigns approximately **99.923% probability** to class `2`.

## Project Layout

```text
.
├── main.c
├── base.h
├── arena.h
├── arena.c
├── prng.h
├── prng.c
├── train_images.mat
├── train_labels.mat
├── test_images.mat
└── test_labels.mat
```

The `.mat` files are headerless binary arrays of 32-bit floating-point values.

| File | Shape |
| --- | ---: |
| `train_images.mat` | `60000 x 784` |
| `train_labels.mat` | `60000 x 1` |
| `test_images.mat` | `10000 x 784` |
| `test_labels.mat` | `10000 x 1` |

Images should be normalized to `[0, 1]`. Labels are stored as floating-point class indices from `0` to `9` and converted to one-hot vectors by the program.

## Build and Run

On macOS:

```bash
clang -std=c11 -O2 main.c -o mnist -lm
./mnist
```

On Linux:

```bash
gcc -std=c11 -O2 main.c -o mnist -lm
./mnist
```

Run the executable from the directory containing the dataset files.

Because `main.c` directly includes `arena.c` and `prng.c`, do not compile those files separately.

## Limitations

This is an educational implementation rather than a production tensor library.

- Operations are limited to two-dimensional matrices.
- Training processes examples individually and accumulates their gradients.
- Matrix multiplication is single-threaded and CPU-only.
- Softmax backward propagation constructs an explicit Jacobian.
- Cross-entropy does not clamp probabilities away from zero.
- Dataset paths and hyperparameters are hard-coded.
- The final incomplete training batch is ignored.
- Model serialization and checkpointing are not implemented.

Despite these limitations, the project demonstrates the complete training pipeline—from raw matrix operations and graph construction to automatic differentiation and MNIST evaluation—entirely in C.

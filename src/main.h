#include "base.h"
#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"
#include <stdlib.h>
#include <string.h>

// ALL THE FUCKING DECLARATIONS (cleaner way would be to make its own header file)
// but this is the only way I hit 1k lines (need to larp-maxx)


//data structure of a matrix: 
typedef struct{ // iniitalizing matrices
    u32 rows, cols; // these are the number of columns and rows
    f32* data; // one dimensional array for the matrix 
    // so indexing happens via data[i * cols + j]
} matrix;

typedef enum{
    MV_FLAG_NONE = 0,
    MV_FLAG_REQUIRES_GRAD = (1<<0),
    MV_FLAG_PARAMETER = (1<<1),
    MV_FLAG_INPUT = (1<<2),
    MV_FLAG_OUTPUT = (1<<3),
    MV_FLAG_DESIRED_OUTPUT = (1<<4),
    MV_FLAG_COST = (1<<5),
} model_var_flags;

typedef enum{
    MV_OP_NULL = 0,
    MV_OP_CREATE,
    _MV_OP_UNARY_START,
    MV_OP_RELU,
    MV_OP_SOFTMAX,
    _MV_OP_BINARY_START,
    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
    
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) \
    ((op) < _MV_OP_UNARY_START ? 0 : \
    ((op) < _MV_OP_BINARY_START ? 1 : 2))


typedef struct model_var{
    u32 index;
    u32 flags;
    matrix* val;
    matrix*grad;

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct{
    model_var** vars;
    u32 size;

}model_program;


typedef struct{
    u32 num_vars;
    model_var *input;
    model_var *output;
    model_var *desired_output;
    model_var *cost;
    model_program forward_prog; 
    model_program cost_prog;
}model_context;


typedef struct{
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;    

}model_training_desc;

// autodiff functions for creating nodes.
model_var* mv_create(mem_arena* arena, model_context* model, u32 rows, u32 cols, u32 flags);
model_var* mv_relu(mem_arena* arena, model_context* model, model_var* input, u32 flags );
model_var* mv_softmax(mem_arena* arena, model_context* model, model_var* input, u32 flags );
model_var* mv_add(mem_arena* arena, model_context* model, model_var* a, model_var* b ,u32 flags );
model_var* mv_sub(mem_arena* arena, model_context* model, model_var* a, model_var* b ,u32 flags );
model_var* mv_matmul(mem_arena* arena, model_context* model, model_var* a, model_var* b ,u32 flags );
model_var* mv_cross_entropy(mem_arena* arena, model_context* model, model_var* p, model_var* q ,u32 flags );

// training the model functions
model_program model_prog_create(mem_arena* arena, model_context* model, model_var* out_var);
void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
void model_feedforward(model_context* model);
void model_train(model_context* model,const model_training_desc* training_desc);


// functions declarations for ML math:
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
bool mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
u64 mat_argmax(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);


b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad( matrix* out, const matrix* softmax_out, const matrix* grad);
b32 mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad, const matrix* p, const matrix* q, const matrix* grad);

b32 mat_check2(const matrix* a, const matrix* b);
b32 mat_check3(const matrix* out, const matrix* a, const matrix* b);

///////// END OF DECLARTIONS 
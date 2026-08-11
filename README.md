# Neural Network and Autodiff Engine in C

A compact neural-network framework written from scratch in C, with an end-to-end MNIST training example. The project implements its own matrix operations, computation graph, reverse-mode automatic differentiation, parameter updates, and model evaluation without relying on an external machine-learning library.

The included model is a small residual multilayer perceptron trained to classify handwritten digits. Its purpose is educational: to make the mechanics behind tensor operations, forward execution, backpropagation, and mini-batch gradient descent explicit in a relatively small codebase.

## Highlights

- Dense, row-major matrix type backed by contiguous `float` storage
- Matrix addition, subtraction, scaling, reduction, and multiplication
- Four matrix-multiplication kernels supporting transposed operands
- Numerically stable softmax using a maximum-value shift
- ReLU and categorical cross-entropy operations
- Dynamically constructed computation graph
- Reverse-mode automatic differentiation for every supported operation
- Topological program generation through depth-first graph traversal
- Mini-batch stochastic gradient descent with shuffled training examples
- Xavier/Glorot initialization for network weights
- MNIST loading, one-hot encoding, training, and full test-set evaluation
- Arena-based memory management for persistent and temporary allocations

## Model architecture

The MNIST image is flattened from \(28 \times 28\) pixels into a \(784 \times 1\) input vector. The classifier contains two 16-unit hidden transformations, a residual connection, and a 10-class output layer:

\[
\begin{aligned}
h_0 &= \operatorname{ReLU}(W_0x+b_0), \\
r_1 &= \operatorname{ReLU}(W_1h_0+b_1), \\
h_1 &= h_0+r_1, \\
\hat{y} &= \operatorname{softmax}(W_2h_1+b_2).
\end{aligned}
\]

The parameter dimensions are:

| Parameter | Shape | Purpose |
| --- | ---: | --- |
| \(W_0\) | \(16 \times 784\) | Projects the image into the hidden representation |
| \(b_0\) | \(16 \times 1\) | First hidden-layer bias |
| \(W_1\) | \(16 \times 16\) | Residual hidden transformation |
| \(b_1\) | \(16 \times 1\) | Residual-layer bias |
| \(W_2\) | \(10 \times 16\) | Produces the ten class logits |
| \(b_2\) | \(10 \times 1\) | Output-layer bias |

The residual addition \(h_1=h_0+r_1\) gives the second hidden block a direct identity path. The weights are sampled from Xavier uniform initialization,

\[
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right),
\]

while the arena's zero-initialized allocations leave the biases at zero.

## How the engine works

### 1. Matrix layer

`matrix` stores a two-dimensional shape and a pointer to contiguous row-major data. The low-level API supplies the numerical primitives used by both forward and backward execution:

- element-wise addition and subtraction;
- matrix multiplication with optional transposition of either operand;
- ReLU, softmax, and cross-entropy;
- scalar scaling, summation, clearing, copying, random filling, and `argmax`.

`mat_mul` dispatches to one of four loop implementations—NN, NT, TN, or TT—according to whether the input matrices should be interpreted as transposed. This allows the same operation to calculate the forward product and both matrix-product gradients without allocating explicit transposed copies.

Softmax is computed as

\[
\operatorname{softmax}(z)_i =
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad m=\max_j z_j,
\]

which is mathematically equivalent to ordinary softmax but avoids unnecessarily large exponentials.

### 2. Computation graph

Every value in the graph is represented by a `model_var`. A variable records:

- its value matrix;
- an optional gradient matrix;
- the operation that produced it;
- up to two input variables;
- flags identifying parameters, inputs, outputs, targets, and costs.

Calling functions such as `mv_matmul`, `mv_add`, or `mv_relu` creates graph nodes rather than immediately evaluating them. Gradient requirements propagate to downstream nodes automatically.

### 3. Compilation and forward execution

`model_prog_create` performs an iterative depth-first traversal from a requested output and builds a topologically ordered `model_program`. The model is compiled into two programs:

- a forward program ending at the predicted class probabilities;
- a cost program ending at the cross-entropy tensor.

`model_prog_compute` walks the selected program from inputs to output and dispatches each node to the appropriate matrix operation.

### 4. Reverse-mode automatic differentiation

Backpropagation starts by setting the final node's gradient to one and traversing the cost program in reverse topological order. Each operation accumulates its local vector-Jacobian product into the gradients of its inputs.

For example, if

\[
C=AB,
\]

and the upstream gradient is \(G=\partial L/\partial C\), then the engine computes

\[
\frac{\partial L}{\partial A}=GB^\mathsf{T},
\qquad
\frac{\partial L}{\partial B}=A^\mathsf{T}G.
\]

Addition and subtraction propagate the upstream gradient directly, ReLU gates it according to the sign of its input, softmax constructs its Jacobian, and cross-entropy differentiates with respect to its operands. Because gradients are accumulated, the residual branch is handled correctly when a value contributes to more than one later node.

### 5. Training

For each epoch, the implementation randomizes an index array and processes the training set in mini-batches. Within a batch it:

1. clears accumulated parameter gradients;
2. copies one image and one-hot target into the graph;
3. evaluates the cost program;
4. runs reverse-mode autodiff;
5. accumulates gradients over all examples in the batch;
6. applies the SGD update

\[
\theta \leftarrow \theta-eta\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta L_i,
\]

where \(\eta\) is the learning rate and \(B\) is the batch size.

At the end of each epoch, the program evaluates all 10,000 test examples and reports classification accuracy and average cross-entropy cost.

## Project layout

The main translation unit expects the following supporting files in the same directory:

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

`base.h` supplies fixed-width aliases and utility macros, `arena.h`/`arena.c` provide the arena allocator, and `prng.h`/`prng.c` provide random-number generation. Because `main.c` directly includes the two `.c` implementation files, they should not also be passed to the compiler as separate translation units.

## Dataset format

The loader reads headerless binary arrays of 32-bit floating-point values in native byte order. It does **not** parse the original IDX-formatted MNIST files directly.

| File | Logical shape | Contents |
| --- | ---: | --- |
| `train_images.mat` | \(60000 \times 784\) | Flattened training images |
| `train_labels.mat` | \(60000 \times 1\) | Class indices from 0 to 9, stored as `float` |
| `test_images.mat` | \(10000 \times 784\) | Flattened test images |
| `test_labels.mat` | \(10000 \times 1\) | Class indices from 0 to 9, stored as `float` |

Image values should be normalized to the range \([0,1]\). During startup, the scalar labels are converted into ten-dimensional one-hot vectors.

The expected byte sizes are approximately 188 MB for the training images, 31 MB for the test images, 240 KB for the training labels, and 40 KB for the test labels.

## Building and running

### Requirements

- A C compiler with C99 support or newer
- The standard C library and math library
- The four supporting source/header files listed above
- MNIST exported in the raw binary format described above

On macOS with Clang:

```bash
clang -std=c11 -O2 -Wall -Wextra main.c -o mnist -lm
./mnist
```

On Linux with GCC:

```bash
gcc -std=c11 -O2 -Wall -Wextra main.c -o mnist -lm
./mnist
```

Run the executable from the directory containing the four `.mat` files because their paths are currently relative to the process's working directory.

## Default training configuration

The configuration in `main` uses:

| Setting | Value |
| --- | ---: |
| Epochs | 20 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Optimizer | Mini-batch SGD |
| Training examples | 60,000 |
| Test examples | 10,000 |

With 60,000 examples and a batch size of 64, the integer division in the current training loop produces 937 batches per epoch. Consequently, 32 training examples are not used in each epoch. Since the data order is reshuffled, these are not necessarily the same examples every time.

During execution, the program first prints the prediction vector for the first test image before training. It then updates a per-batch progress line, evaluates the test set after every epoch, and finally prints the same example's post-training prediction vector. Output has the general form:

```text
pre-training output: ...
Epoch  1 / 20, Batch  937 /  937, Average Cost: ...
Test Completed. Accuracy:  ... / 10000 (...%), Average Cost: ...
...
post-training output: ...
```

Exact values vary because parameter initialization and training order are randomized.

## Supported operations

| Graph operation | Forward computation | Backward computation |
| --- | --- | --- |
| `MV_OP_ADD` | \(A+B\) | Accumulates \(G\) into both inputs |
| `MV_OP_SUB` | \(A-B\) | Accumulates \(G\) and \(-G\) |
| `MV_OP_MATMUL` | \(AB\) | Computes \(GB^\mathsf{T}\) and \(A^\mathsf{T}G\) |
| `MV_OP_RELU` | \(\max(0,x)\) | Passes \(G\) where \(x>0\) |
| `MV_OP_SOFTMAX` | Normalized exponentials | Multiplies by the softmax Jacobian |
| `MV_OP_CROSS_ENTROPY` | \(-p\log q\), element-wise | Differentiates with respect to \(p\) and/or \(q\) |

The final scalar-like loss used for logging and optimization is the sum of the ten element-wise cross-entropy outputs.

## Memory management

Long-lived model state, parameters, graph nodes, and datasets are allocated from a 1 GiB permanent arena. Short-lived objects—such as traversal buffers, the shuffled index array, and the softmax Jacobian—use scratch arenas and are released together.

This design avoids many small heap allocations and makes object lifetimes straightforward: destroying the permanent arena releases the full model and dataset in one operation. It also means the current program uses a fixed, comparatively large virtual reservation and does not individually free matrices.

## Current limitations

This is a focused learning implementation, not a replacement for a production tensor library. Important limitations include:

- Operations work on individual matrices; there is no general N-dimensional tensor abstraction or vectorized batch dimension.
- Training performs one forward/backward pass per example and only accumulates gradients across a mini-batch, so it is not computationally optimized.
- Matrix multiplication is a straightforward CPU implementation without SIMD, multithreading, BLAS, or GPU acceleration.
- Softmax is global over its input matrix, which is appropriate for the current \(10\times1\) output but not a general batched API.
- The explicit softmax Jacobian requires \(O(n^2)\) temporary work; a fused softmax-cross-entropy gradient would be simpler and faster.
- Cross-entropy does not clamp probabilities before `logf` or division, so an exact zero probability could produce non-finite values.
- The data loader accepts short files and leaves the remaining arena-initialized values unchanged instead of requiring an exact file size.
- Shape failures are returned as false or `NULL`, but graph construction does not provide descriptive error reporting.
- There is no model serialization, checkpointing, inference-only loader, validation split, or early stopping.
- The random seed and reproducibility behavior depend on the accompanying PRNG implementation.

These constraints keep the internals visible and leave clear directions for extension.

## Possible extensions

Natural next steps include:

1. Fuse softmax and cross-entropy to compute the logit gradient directly.
2. Add numerical gradient checking for each differentiable operation.
3. Introduce batched matrix inputs and vectorized bias broadcasting.
4. Add optimizers such as momentum, RMSProp, or Adam.
5. Implement model checkpoint save/load support.
6. Split declarations and implementations into reusable modules.
7. Add unit tests for shapes, forward values, and analytical gradients.
8. Improve matrix multiplication through cache blocking, SIMD, threading, or BLAS.
9. Add layers such as sigmoid, tanh, normalization, and convolution.
10. Replace fixed dataset paths and hyperparameters with command-line options.

## Educational value

The project exposes the parts that high-level frameworks normally hide. A call such as `loss.backward()` conceptually requires a graph of dependencies, a valid execution order, stored intermediate values, local derivatives, reverse traversal, and careful gradient accumulation. Here, each of those mechanisms is represented directly in C.

That makes the code useful for studying:

- how matrix calculus becomes executable loops;
- why reverse-mode autodiff is efficient for scalar-loss models;
- how topological ordering separates graph construction from execution;
- why gradients must accumulate at branches such as residual connections;
- how parameter initialization and numerical stability affect training;
- how memory lifetime choices shape low-level ML system design.

## Acknowledgements

MNIST was created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. This repository uses a separately exported raw representation of the dataset; it does not bundle or parse the original MNIST distribution.

## License

No license has been specified. Add a `LICENSE` file before redistributing the project or accepting external contributions.

# Warp: A Python Multi-GPU Programming Model with CUDA Performance and Flexibility

## Abstract

The rise of artificial intelligence has firmly established Python as the primary programming language for AI, and a similar trend is emerging in scientific applications. However, designing efficient and scalable applications in Python presents unique challenges, driving the development of innovative frameworks. For example, tools like JAX offer a tensor-based programming model inspired by NumPy, which can be mapped to multi-GPU execution. While these abstractions provide significant capabilities, they often obscure low-level details, limiting the user's ability to write optimal code. Tensor-based frameworks often struggle with efficient conditional logic, loops, and memory management.

NVIDIA Warp introduces a new solution: a Python multi-GPU programming model with a CUDA-like interface. This approach empowers developers to write high-performance, fine-grained parallel code in Python with CUDA-level efficiency. Warp leverages just-in-time (JIT) compilation not only to directly generate and compile CUDA code but also to provide a differentiable programming interface, which is essential for optimization and AI workflows.

In this tutorial, we will introduce the Warp multi-GPU programming model and demonstrate how to write high-performance, multi-GPU applications in Python. Beginning with simple Warp kernel development and profiling, we will progress to more advanced features such as automatic differentiability and multi-GPU programming.



## Schedule

### Day One

| Time          | Description                         | Status   |
|---------------|-------------------------------------|----------|
| Documentation | Provides detailed instructions     | ✅ Done   |
| Tutorials     | Step-by-step usage guides          | 🚧 WIP    |
| Support       | Community and professional support | ❌ Not yet |

### Day Two

| Time          | Description                        | Status    |
|---------------|------------------------------------|-----------|
| Documentation | Provides detailed instructions     | ✅ Done   |
| Tutorials     | Step-by-step usage guides          | 🚧 WIP    |
| Support       | Community and professional support | ❌ Not yet |

## Reading list

## Tutorial setup

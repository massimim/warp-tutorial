# Warp: A Python Multi-GPU Programming Model with CUDA Performance and Flexibility

## Abstract

The rise of artificial intelligence has firmly established Python as the primary programming language for AI, and a similar trend is emerging in scientific applications. However, designing efficient and scalable applications in Python presents unique challenges, driving the development of innovative frameworks. For example, tools like JAX offer a tensor-based programming model inspired by NumPy, which can be mapped to multi-GPU execution. While these abstractions provide significant capabilities, they often obscure low-level details, limiting the user's ability to write optimal code. Tensor-based frameworks often struggle with efficient conditional logic, loops, and memory management.

NVIDIA Warp introduces a new solution: a Python multi-GPU programming model with a CUDA-like interface. This approach empowers developers to write high-performance, fine-grained parallel code in Python with CUDA-level efficiency. Warp leverages just-in-time (JIT) compilation not only to directly generate and compile CUDA code but also to provide a differentiable programming interface, which is essential for optimization and AI workflows.

In this tutorial, we will introduce the Warp multi-GPU programming model and demonstrate how to write high-performance, multi-GPU applications in Python. Beginning with simple Warp kernel development and profiling, we will progress to more advanced features such as automatic differentiability and multi-GPU programming.

## Schedule

The tutorial will be divided into two sections, each one starting with a general introduction to Warp and then focusing on different aspects of the programming model.  The following is a tentative schedule for the tutorial that is subject to change. Please check the tutorial website for the most up-to-date information closer to the tutorial date.

### Day One

| Duration  | Title                                          | Topics |
|-----------|-----------------------------------------------|       |
| **1h**    | Introduction to GPU programming in Warp | GPU architectures and Warp programming model|
|           | | Managing GPU-CPU data movement           |
|           | | Warp data types and kernel launches      |
| **1.5h**  | Optimization of Stencil computations on a single GPU |A simple 2D CFD based on LBM|
|           || Profiling Warp and comparison with plain CUDA apps |
|           || Optimizations with `wp.static`          |
| **1.5h**  | Stencil on multi-GPU                   |Managing multiple devices in Warp|
|           || Porting the simple CFD app to multi-GPU with 1D partitioning  |
|           || Overlapping computation and data transfer |

### Day Two

| Duration  | Title                                         | Topics |
|-----------|-----------------------------------------------|---------|
| **1h**    | Introduction to GPU programming in Warp   | GPU architectures and Warp programming model|
|           || Managing GPU-CPU data movement           |
|           || Warp data types and kernel launches      |
| **1.5h**  | Optimization of Stencil computations on a single GPU |Implementing a simple 2D CFD based on LBM|
|           || Profiling Warp and comparison with plain CUDA apps |
|           || Optimizations with `wp.static`          |
| **1.5h**  | Stencil with out-of-core execution       |Unified memory and NVIDIA Grace systems |
|           || Out-of-core capabilities for our CFD app |
|           || Overlapping computation and data transfer |


## Prerequisites

The tutorial is designed for developers with a basic understanding of Python.
Participants should have some experience in writing and executing Python code. Familiarity with GPU architectures or CUDA is beneficial but not required.

<!--
```hidden
## Reading List
To be provided.
```
-->

## Tutorial Setup

Attendees are required to have a laptop with a working Python environment and Nsight Systems installed. Attendees will be provided with credentials to log into an HPC cluster with NVIDIA GPUs.

### Installing Nsight Systems

Instructions to install Nsight Systems can be found [here](https://developer.nvidia.com/nsight-systems).

### How to Access the Cluster

More detailed information will be provided closer to the tutorial date.

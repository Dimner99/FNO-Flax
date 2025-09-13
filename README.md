# Fourier Neural Operator (FNO) Implementation with JAX/Flax

This repository implements 1D Fourier Neural Operator using JAX and Flax, showcasing four different levels of optimization for distributed training.

## What is a Fourier Neural Operator (FNO)?

Fourier Neural Operators are a class of neural networks designed to learn mappings between infinite-dimensional function spaces. Unlike traditional neural networks that operate on discrete grids, FNOs can handle functions defined on continuous domains and are particularly effective for solving partial differential equations (PDEs).

Key advantages of FNOs:

- **Resolution invariant**: Can generalize across different grid resolutions
- **Efficient**: Leverage Fast Fourier Transform (FFT) for computational efficiency
- **Universal approximation**: Can approximate complex function mappings
- **Parameter efficiency**: Fewer parameters compared to traditional CNN approaches for PDE solving

The core insight is to perform convolutions in Fourier space, where they become pointwise multiplications, making them computationally efficient for learning operators on function spaces.

## What is Flax?

Flax is a neural network library built on top of JAX that provides:

- **Functional programming**: Pure functions and immutable data structures
- **Flexibility**: Explicit state management and transformation control
- **Performance**: Automatic differentiation, JIT compilation, and vectorization
- **Scaling**: Easy parallelization across multiple devices

Flax's design philosophy emphasizes composability and explicitness, making it ideal for research and complex model architectures like FNOs.

## Four Levels of Optimization

This repository demonstrates four progressive optimization strategies:

### 1. **Basic Implementation** (`fno.ipynb`)

- Single-device training
- Standard training loop


### 2. **Scan Optimization** (`fno_scan_optimization.ipynb`)

- Single-device training
- Uses `jax.lax.scan` for efficient loops

### 3. **Parallel Training** (`fno_parallel.ipynb`)

- Multi-device data parallelism
- Standard training loop


### 4. **Scan-Optimized Parallel** (`fno_parallel_scan.ipynb`)

- Multi-device data parallelism
- Uses `jax.lax.scan` for efficient loops


Each level builds upon the previous one, demonstrating how to scale FNO training from single-device to highly optimized parallel training.


## SCAN based training.


This approach  does not give significant performance boost when training is performed on cpu, the performance is almost the same. When training on gpu, tpu the performance boost is **bigger** the **smaller** the mini batches are. In general it is a function of the gpu computation power and the computation needed for each mini batch. When we don't use the scan approach and implement the feeding of the data using a for loop there is a time needed for the computation inside the gpu and a time needed for the cpu-gpu communication for each *"batch"* of computations. If the time of the gpu computation is big enough then the scan method will not boost as much the performance because the time overhead for each batch is primarily created for the computation, the communication time needed can be ignored. When the computation time is small enough then the time overhead is being made by the cpu-gpu communication and this is when the scan method can boost performance so much because it feeds the gpu more efficiently.


#### In more details

In pure CPU training, this “scan” strategy yields negligible speed‐up—the overall runtime remains essentially unchanged. On accelerators (GPU/TPU), however, the benefit grows as mini‐batch sizes shrink.

Define:

- $t_{\mathrm{gpu}} =$ time to perform one mini-batch’s computations on the device  
- $t_{\mathrm{sync}} =$ overhead for one CPU↔GPU synchronization (data transfer, kernel launch, etc.)

**Without scan (simple loop):**

Each of the mini-batches incurs both compute and sync costs:

$$
T_{\mathrm{loop}} = N\\,(t_{\mathrm{gpu}} + t_{\mathrm{sync}})
$$

**With scan (fused execution):**

The steps’ computations are batched on the device, so you pay the sync cost only once:

$$
T_{\mathrm{scan}} = N\\,t_{\mathrm{gpu}} + t_{\mathrm{sync}}
$$

**Relative speed-up:**

$$
\text{Speed-up} = \frac{T_{\mathrm{loop}}}{T_{\mathrm{scan}}}
= \frac{N\\,(t_{\mathrm{gpu}} + t_{\mathrm{sync}})}{N\\,t_{\mathrm{gpu}} + t_{\mathrm{sync}}}
$$

- If $t_{\mathrm{gpu}} \\gg t_{\mathrm{sync}}$, then  
  $T_{\mathrm{loop}} \\approx N\\,t_{\mathrm{gpu}}$ and  
  $T_{\mathrm{scan}} \\approx N\\,t_{\mathrm{gpu}}$ → minimal speed-up.  

- If $t_{\mathrm{gpu}} \\ll t_{\mathrm{sync}}$, then  
  $T_{\mathrm{loop}} \\approx N\\,t_{\mathrm{sync}}$ while  
  $T_{\mathrm{scan}} \\approx t_{\mathrm{sync}}$ → nearly $N \\times$ faster execution.


## Reference

**Fourier Neural Operator for Parametric Partial Differential Equations**  
Zongyi Li, Nikola B Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar  
ICLR 2021

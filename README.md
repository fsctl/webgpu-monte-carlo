# WebGPU Monte Carlo Simulation

This repo demonstrates how to use WebGPU for scientific computing.  It calculates the value of π by computing sqrt(1-x<sup>2</sup>) many times for random values of `x` on the interval [0,1) and averaging the results.  (See [Theory of Operation](#theory-of-operation) for why this works to compute π.)

<p align="center">
<img src="https://github.com/fsctl/webgpu-monte-carlo/blob/main/screenshot.png?raw=true" alt="screenshot">
</p>

## Performance

The same algorithm is implemented in [WebGPU](webgpu/) and [CPU-only Rust](rustimpl/).  The paralellism of the GPU improves performance dramatically:

![performance](https://github.com/fsctl/webgpu-monte-carlo/blob/main/time-graph.png?raw=true)

For low iteration counts, both the CPU and GPU implementations are reasonably fast, but the estimates of π are poor (±0.01).  As we increase the iteration count to 10^11, the estimate becomes much better (±0.00001), but the CPU-only version's execution time (blue line) grows rapidly.  The GPU version's execution time (orange line) increases only slightly.

## How to Run

#### WebGPU Version

To run the WebGPU version of the algorithm, you'll need to install a browser with WebGPU support, clone this repo, and then browse to the `webgpu/index.html` page.  Detailed instructions are in the [`webgpu/README.md`](webgpu/README.md) file.

#### Rust Version

For performance comparison purposes, a CPU-only Rust version of the same algorithm is included in the [`rustimpl`](rustimpl/) directory.  See [`rustimpl/README.md`](rustimpl/README.md) for instructions on how to build and run it.

## Theory of Operation

#### Background

Monte Carlo methods are one of the least useful ways of computing the digits of π. Nevertheless, it's a nice way of experimenting with scientific computing on GPUs because these algorithms parallelize easily.  The method used here is from the article [Monte Carlo estimates of pi](https://blogs.sas.com/content/iml/2016/03/14/monte-carlo-estimates-of-pi.html). As that article explains, the curve y=sqrt(1-x<sup>2</sup>) on the interval 0 < x < 1 is a quarter of a unit radius circle, so the area under this curve π/4.

To find the area under that curve, it suffices to calcuate the average value of sqrt(1-x<sup>2</sup>) for a large number of random `x` between 0 and 1 because the average value of any function on the interval `[0,1)` is the same as taking the integral of that function between 0 and 1.  The article explains this in more detail.

Our general strategy, then, is to generate a large number of random `x` values between 0 and 1, compute sqrt(1-x<sup>2</sup>), and average these values to get an estimate of π/4.  We multiply this estimate by 4 to get π.

#### Design

The biggest challenge in creating such an algorithm in WebGPU is the lack of support for a 64-bit `double` floating point type.  Because we want to accumulate and average a large number of individual π estimates, any floating point type smaller than 64 bits is likely to overflow.

To get around this problem, our compute shader will only run 10,000 iterations, and then we'll invoke it in parallel 1,000,000 times with different random seeds.  That's a total of 10^10 (10 billion) total independent estimates of π, all of which will be averaged together in Javascript.  Broken up this way, all of these iterations complete in about 230 ms on an [AMD Radeon Pro 5300M](https://www.techpowerup.com/gpu-specs/radeon-pro-5300m.c3464) laptop GPU with 1,280 cores.

## References

[Monte Carlo estimates of pi.](https://blogs.sas.com/content/iml/2016/03/14/monte-carlo-estimates-of-pi.html) Covers the specific estimate of π we are using.

[Hash Functions for GPU Rendering.](https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/) Discusses the PCG family of hash functions that we use inside the compute shader as a PRNG to generate 10,000 random values of `x`.

[https://github.com/samdauwe/webgpu-native-examples/blob/master/src/examples/prng.c](https://github.com/samdauwe/webgpu-native-examples/blob/master/src/examples/prng.c) Example PCG hash function code in WebGPU shader language (WGSL).

[Get started with GPU Compute on the web.](https://web.dev/gpu-compute/) Tutorial on WebGPU from which the basic structure of the program is derived.
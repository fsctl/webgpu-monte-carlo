# Rust Implementation

This is the CPU-only Rust implementation of the same algorithm as the [WebGPU](../webgpu/) code uses.  To build and run it:

```
make
```

The `main` function is analogous to the `index.js` file in the [WebGPU implementation](../webgpu/).  It uses the native Rust random number generator to generate a large number of random seeds, then calls `shader_main` on each seed.  These calls do not run in paralell like they would on a GPU.

`shader_main` is analogous to the main function in the WebGPU compute shader.  It iterates a pcg hash 10,000 times to compute π/4, then multiplies by 4 to get an estimate of π and returns that value.  The `main` function averages all of these π estimates and prints out the mean estimate.
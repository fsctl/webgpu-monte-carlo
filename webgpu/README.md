# WebGPU Monte Carlo Simulation

This directory contains an `index.html` and `index.js` pair that use WebGPU to compute the value of π using a Monte Carlo approach.  The theory of operation is described in the [main README.md file](../README.md).

## How to Run

1. Regular browsers don't support WebGPU at the time of this writing, so first install [Chrome Canary](https://www.google.com/chrome/canary/).

2.  Enable "Unsafe WebGPU" in **[chrome://flags/#enable-unsafe-webgpu](chrome://flags/#enable-unsafe-webgpu)**.

3.  Verify WebGPU is working using an examples site like [this one](https://austin-eng.com/webgpu-samples/samples/animometer).

4.  Clone this repo and browse to `index.html` on your local drive.

You should get something close to pi ~= 3.14159 in sub-second execution time.  Check the Javascript Console for errors (`View -> Developer -> Javascript Console`).

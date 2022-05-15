(async () => {
    const start_time = Date.now();

    if (!navigator.gpu) {
      console.log("WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag.");
      return;
    }
  
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log("Failed to get GPU adapter.");
      return;
    }
    const device = await adapter.requestDevice();
  
    //
    // Random seeds input array
    //
    const NUM_SEEDS = 1000000;
    const MAX_I32 = 2147483647;
    var randSeeds = new Int32Array(NUM_SEEDS);
    for (var i = 0; i < NUM_SEEDS; i += 1) {
      randSeeds[i] = Math.abs(Math.floor(Math.random() * MAX_I32));
      //console.log(`  ${randSeeds[i]}`);
    }
    console.log("random seeds = ");
    console.log(randSeeds);

    //
    // Create and populate a GPU buffer that stores a Int32 size of the other buffers
    //
    var bufSizes = new Int32Array(1);
    bufSizes[0] = NUM_SEEDS;
    const gpuBuffersBufSizes = device.createBuffer({
      mappedAtCreation: true,
      size: bufSizes.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferBufSizes = gpuBuffersBufSizes.getMappedRange();
    new Int32Array(arrayBufferBufSizes).set(bufSizes);
    gpuBuffersBufSizes.unmap();
    
    //
    // Create and populate the GPU buffer with random seeds
    //
    const gpuBufferRandSeeds = device.createBuffer({
      mappedAtCreation: true,
      size: randSeeds.byteLength,
      usage: GPUBufferUsage.STORAGE
    });
    const arrayBufferRandSeeds = gpuBufferRandSeeds.getMappedRange();
    new Int32Array(arrayBufferRandSeeds).set(randSeeds);
    gpuBufferRandSeeds.unmap();
  
    //
    // Create GPU output buffer for each iteration's pi estimate
    //
    const gpuBufferPiEstimatesSize = Float32Array.BYTES_PER_ELEMENT * NUM_SEEDS;
    const gpuBufferPiEstimates = device.createBuffer({
      size: gpuBufferPiEstimatesSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
  
    //
    // Compute shader code
    //
    const computeShaderModule = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> bufSizes : array<u32>;
        @group(0) @binding(1) var<storage, read> randSeeds : array<i32>;
        @group(0) @binding(2) var<storage, write> piEstimates : array<f32>;

        fn pcg_rng(input: u32) -> u32 {
          var state = input * 747796405u + 2891336453u;
          let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
          return (word >> 22u) ^ word;
        }

        fn calc(randval01 : f32) -> f32 {
          return sqrt(1.0-(randval01*randval01));
        }

        @stage(compute)
        @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
          // Skip if this execution would be past array bounds
          if (global_id.x >= u32(bufSizes[0])) {
            return;
          }
  
          // index for both input and output arrays
          let idx = u32(global_id.x);
  
          // input:  rng seed
          let random_seed : i32 = i32(randSeeds[idx]);

          // estimate pi over 10k iterations
          var rand : u32 = u32(random_seed);
          var pi_ests_sum : f32 = f32(0);
          var pi_ests_count : f32 = f32(0);
          for (var i = 0; i < 10000; i++) {
            rand = pcg_rng(rand);
            let randval01 = f32(rand) * (1.0 / 4294967295.0);
            pi_ests_sum += calc(randval01);
            pi_ests_count += f32(1);
          }
          var pi_est : f32 = 4.0 * (pi_ests_sum / pi_ests_count);
  
          // Store return value in appropriate array el
          piEstimates[idx] = f32(pi_est);
        }
      `
    });

    //
    // Bind group layout
    //
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage"
          }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "read-only-storage"
          }
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "storage"
          }
        }
      ]
    });
  
    //
    // Bind group
    //
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: gpuBuffersBufSizes
          }
        },
        {
          binding: 1,
          resource: {
            buffer: gpuBufferRandSeeds
          }
        },
        {
          binding: 2,
          resource: {
            buffer: gpuBufferPiEstimates
          }
        },
      ]
    });

    //
    // Pipeline setup
    //
    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: computeShaderModule,
        entryPoint: "main"
      }
    });
    
    //
    // Commands submission
    //
    const commandEncoder = device.createCommandEncoder();
  
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(NUM_SEEDS / 256 + 1);
    passEncoder.end();
  
    //
    // Get a GPU buffer for reading in an unmapped state
    //
    const gpuBufferPiEstimatesCopy = device.createBuffer({
      size: gpuBufferPiEstimatesSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    //
    // Encode commands for copying buffer to buffer
    //
    commandEncoder.copyBufferToBuffer(
      gpuBufferPiEstimates,       // source buffer
      0,                          // source offset
      gpuBufferPiEstimatesCopy,   // destination buffer
      0,                          // destination offset
      gpuBufferPiEstimatesSize    // size
    );
  
    //
    // Submit GPU commands
    //
    const gpuCommands = commandEncoder.finish();
    device.queue.submit([gpuCommands]);
  
    //
    // Map copy dst buffer and read it into JS array
    //
    await gpuBufferPiEstimatesCopy.mapAsync(GPUMapMode.READ);
    const piEstimatesMapped = gpuBufferPiEstimatesCopy.getMappedRange();
    const piEstimates = new Float32Array(piEstimatesMapped);
    console.log(piEstimates);
  
    //
    // Update html page w/ avg of JS array values
    //
    var avgPi = piEstimates.reduce((accum,val)=>accum+val, 0) / piEstimates.length;
    const elapsed_time = Date.now() - start_time;
    outputDisplay.textContent = `pi ~= ${(avgPi).toPrecision(6)} in ${elapsed_time} ms`;
  })();  
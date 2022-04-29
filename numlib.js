import glslangModule from "./glslang/dist/web-devel/glslang.onefile.js";


function conv_matrix(mat,array_type) {
  array_type = array_type || Float32Array 
  if (mat[0] === undefined) {
    mat = [mat]; // turn vector to 1 x n matrix
  }
  const arr = new array_type(2 + mat.length*mat[0].length);
  arr[0] = mat.length;
  arr[1] = mat[0].length;
  for (let i = 0, offset = 2; i < mat.length; ++i, offset += (mat[i]||[]).length) {
    arr.set(mat[i],offset);
  }
  return arr;
}


export const matrix_multiplication = async function(first,second) {

console.log("START");

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();


console.time("unpacking");
    // First Matrix
    const firstMatrix = conv_matrix(first);
    // console.log("First: ",firstMatrix);
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    // Second Matrix
    const secondMatrix = conv_matrix(second);
    // console.log("Second: ",secondMatrix);
    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
      size: secondMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });
console.timeEnd("unpacking");

console.time("CORE GPU");

console.log("Unpacked matrices. Start copy to GPU");
console.time("copy-to-gpu");
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();
console.log("Finished copy to GPU");
console.timeEnd("copy-to-gpu");

    // Result Matrix
    const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
    const resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayout = device.createBindGroupLayout({
      bindings: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          type: "readonly-storage-buffer"
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          type: "readonly-storage-buffer"
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          type: "storage-buffer"
        }
      ]
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      bindings: [
        {
          binding: 0,
          resource: {
            buffer: gpuBufferFirstMatrix
          }
        },
        {
          binding: 1,
          resource: {
            buffer: gpuBufferSecondMatrix
          }
        },
        {
          binding: 2,
          resource: {
            buffer: resultMatrixBuffer
          }
        }
      ]
    });

    const computeShaderCode = `#version 450

      layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
          vec2 size;
          float numbers[];
      } firstMatrix;

      layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
          vec2 size;
          float numbers[];
      } secondMatrix;

      layout(std430, set = 0, binding = 2) buffer ResultMatrix {
          vec2 size;
          float numbers[];
      } resultMatrix;

      void main() {
        resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

        ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
        float result = 0.0;
        for (int i = 0; i < firstMatrix.size.y; i++) {
          int a = i + resultCell.x * int(firstMatrix.size.y);
          int b = resultCell.y + i * int(secondMatrix.size.y);
          result += firstMatrix.numbers[a] * secondMatrix.numbers[b];
        }

        int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
        resultMatrix.numbers[index] = result;
      }
    `;

console.timeLog("CORE GPU");
console.log("Waiting for glslangModule");

    const glslang = await glslangModule();

console.log("Got glslangModule. Starting compileGLSL");
console.timeLog("CORE GPU");
    const code = glslang.compileGLSL(computeShaderCode, "compute");
console.log("Finished compileGLSL");
console.timeLog("CORE GPU");

console.time("MINIMAL GPU CYCLE");
    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      computeStage: {
        module: device.createShaderModule({
          code: code
        }),
        entryPoint: "main"
      }
    });

console.timeLog("CORE GPU");

    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatch(firstMatrix[0] /* x */, secondMatrix[1] /* y */);
    passEncoder.endPass();

    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

console.timeLog("CORE GPU");

    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      resultMatrixBufferSize /* size */
    );

console.timeLog("CORE GPU");

console.log("Ready to submit commands");
    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
console.time("GPU Compute");
    device.defaultQueue.submit([gpuCommands]);
console.timeEnd("GPU Compute");

console.log("Commands executed. Reading result.");
console.timeEnd("CORE GPU");


console.time("Reading result");
    // Read and return buffer.
    const arrayBuffer = await gpuReadBuffer.mapReadAsync();
console.timeEnd("Reading result");
console.timeEnd("MINIMAL GPU CYCLE");

    return new Float32Array(arrayBuffer);
}


// Updating data to an existing buffer (like WebGL's bufferSubData)
// See https://github.com/beaufortfrancois/gpuweb/blob/master/design/BufferOperations.md
function bufferSubData(device, destBuffer, destOffset, srcArrayBuffer) {
    const byteCount = srcArrayBuffer.byteLength;
    const [srcBuffer, arrayBuffer] = device.createBufferMapped({
        size: byteCount,
        usage: GPUBufferUsage.TRANSFER_SRC
    });
    new Uint8Array(arrayBuffer).set(new Uint8Array(srcArrayBuffer)); // memcpy
    srcBuffer.unmap();

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, destBuffer, destOffset, byteCount);
    const commandBuffer = encoder.finish();
    const queue = device.getQueue();
    queue.submit([commandBuffer]);

    srcBuffer.destroy();
}



export const calc_net_and_act_iter = async function(first,second,weight_indices,iter) {

    iter = iter || 1;

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const computeShaderCode = `#version 450

      // activations from the innodes
      layout(std430, set = 0, binding = 0) readonly buffer InActivations {
          vec2 size;
          float numbers[];
      } inacts; /* Assumed to be a matrix with a single row */

      // weights from the innodes: each row has the innode weights to one outnode
      layout(std430, set = 0, binding = 1) readonly buffer Weights {
          vec2 size;
          float numbers[];
      } weights;

      // resulting activations
      layout(std430, set = 0, binding = 3) buffer OutActivations {
          vec2 size;
          float numbers[];
      } outacts;

      // weight indices
      layout(std430, set = 0, binding = 4) readonly buffer WeightIndices {
          vec2 size;
          uint numbers[];
      } indices;


      void main() {
 
        outacts.size = vec2(inacts.size.x, weights.size.y);

        ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
        float result = 0.0;
        for (int i = 0; i < inacts.size.y; i++) {
          int a = i + resultCell.x * int(inacts.size.y);
          int b = resultCell.y + i * int(weights.size.y);
          result += inacts.numbers[a] * weights.numbers[b] * indices.numbers[b];
        }

        int index = resultCell.y + resultCell.x * int(weights.size.y);
        outacts.numbers[index] = result;
      }
    `;

    const glslang = await glslangModule();
    const code = glslang.compileGLSL(computeShaderCode, "compute");

    let firstMatrix = conv_matrix(first);
    const secondMatrix = conv_matrix(second);
    let indices = conv_matrix(weight_indices,Uint32Array);

    // First Matrix to buffer
    let [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();

    // Second Matrix to buffer
    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
      size: secondMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();

    // Indices matrix to buffer
    const [gpuBufferIndices,arrayBufferIndices] = device.createBufferMapped({
      size: indices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    new Uint32Array(arrayBufferIndices).set(indices);
    gpuBufferIndices.unmap();


    async function cycle(gpuBufferFirstMatrix,gpuBufferSecondMatrix,gpuBufferIndices,cycles) {

      cycles = cycles || 1;

      // Result Matrix
      const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
      const resultMatrixBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      const bindGroupLayout = device.createBindGroupLayout({
        bindings: [ {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            type: "readonly-storage-buffer"
          }, {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            type: "readonly-storage-buffer"
          }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            type: "storage-buffer"
          }, {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            type: "readonly-storage-buffer"
          }
        ]
      });

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        bindings: [ {
            binding: 0,
            resource: {
              buffer: gpuBufferFirstMatrix
            }
          }, {
            binding: 1,
            resource: {
              buffer: gpuBufferSecondMatrix
            }
          }, {
            binding: 3,
            resource: {
              buffer: resultMatrixBuffer
            }
          }, {
            binding: 4,
            resource: {
              buffer: gpuBufferIndices
            }
          }
        ]
      });

      const computePipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout]
        }),
        computeStage: {
          module: device.createShaderModule({
            code: code
          }),
          entryPoint: "main"
        }
      });

      const commandEncoder = device.createCommandEncoder();

      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatch(firstMatrix[0] /* x */, secondMatrix[1] /* y */);
      passEncoder.endPass();

      // Get a GPU buffer for reading in an unmapped state.
      const gpuReadBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      // Encode commands for copying buffer to buffer.
      commandEncoder.copyBufferToBuffer(
        resultMatrixBuffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        resultMatrixBufferSize /* size */
      );

      // Submit GPU commands.
      const gpuCommands = commandEncoder.finish();
      device.defaultQueue.submit([gpuCommands]);      

      if (--cycles) {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(resultMatrixBuffer,0,gpuBufferFirstMatrix,0,firstMatrix.byteLength);
        const commandBuffer = encoder.finish();
        const queue = device.defaultQueue;
        queue.submit([commandBuffer]);
        return cycle(gpuBufferFirstMatrix,gpuBufferSecondMatrix,gpuBufferIndices,cycles);
      } else {
        return gpuReadBuffer;
      }
    }

    let result = await cycle(gpuBufferFirstMatrix,gpuBufferSecondMatrix,gpuBufferIndices,iter);
    let arrayBuffer = await result.mapReadAsync();
    return new Float32Array(arrayBuffer);
}

// This version reads out and can potentially report the activations ('first') at
// each cycle. This process is expensive however. Even so, we can get 2 to 3 times
// speedup on a laptop with only 24 ALUs at size 1000 with 1000 cycles.
export const calc_net_and_act_iter_report_all = async function(first,second,iter) {

    iter = iter || 1;

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const computeShaderCode = `#version 450

      layout(std430, set = 0, binding = 0) readonly buffer GirstMatrix {
          vec2 size;
          float numbers[];
      } girstMatrix;

      layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
          vec2 size;
          float numbers[];
      } secondMatrix;

      layout(std430, set = 0, binding = 2) buffer ResultMatrix {
          vec2 size;
          float numbers[];
      } resultMatrix;

      void main() {
        resultMatrix.size = vec2(girstMatrix.size.x, secondMatrix.size.y);

        ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
        float result = 0.0;
        for (int i = 0; i < girstMatrix.size.y; i++) {
          int a = i + resultCell.x * int(girstMatrix.size.y);
          int b = resultCell.y + i * int(secondMatrix.size.y);
          result += girstMatrix.numbers[a] * secondMatrix.numbers[b];
        }

        int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
        resultMatrix.numbers[index] = result;
      }
    `;

    const glslang = await glslangModule();
    const code = glslang.compileGLSL(computeShaderCode, "compute");

    let firstMatrix = conv_matrix(first);
    const secondMatrix = conv_matrix(second);

    function cycle(firstMatrix,secondMatrix) {

      // First Matrix
      // console.log("First: ",firstMatrix);
      const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
        size: firstMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
      gpuBufferFirstMatrix.unmap();


      // Second Matrix
      // console.log("Second: ",secondMatrix);
      const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
        size: secondMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
      gpuBufferSecondMatrix.unmap();


      // Result Matrix
      const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
      const resultMatrixBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });

      const bindGroupLayout = device.createBindGroupLayout({
        bindings: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            type: "readonly-storage-buffer"
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            type: "readonly-storage-buffer"
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            type: "storage-buffer"
          }
        ]
      });

      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        bindings: [
          {
            binding: 0,
            resource: {
              buffer: gpuBufferFirstMatrix
            }
          },
          {
            binding: 1,
            resource: {
              buffer: gpuBufferSecondMatrix
            }
          },
          {
            binding: 2,
            resource: {
              buffer: resultMatrixBuffer
            }
          }
        ]
      });

    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      computeStage: {
        module: device.createShaderModule({
          code: code
        }),
        entryPoint: "main"
      }
    });

      const commandEncoder = device.createCommandEncoder();

      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatch(firstMatrix[0] /* x */, secondMatrix[1] /* y */);
      passEncoder.endPass();

      // Get a GPU buffer for reading in an unmapped state.
      const gpuReadBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      // Encode commands for copying buffer to buffer.
      commandEncoder.copyBufferToBuffer(
        resultMatrixBuffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        resultMatrixBufferSize /* size */
      );

      // Submit GPU commands.
      const gpuCommands = commandEncoder.finish();
      device.defaultQueue.submit([gpuCommands]);      


      // Copy result to firstMatrix
      // TODO: Move to function
      // const encoder = device.createCommandEncoder();
      // encoder.copyBufferToBuffer(gpuReadBuffer,0,gpuBufferFirstMatrix,0,resultMatrixBufferSize);
      // const commandBuffer = encoder.finish();
      // const queue = device.defaultQueue;
      // queue.submit([commandBuffer]);


      return gpuReadBuffer;
    }

    for (let k = 0; k < iter; ++k) {
      let result = cycle(firstMatrix,secondMatrix);;
      // Read and return buffer.
      let arrayBuffer = await result.mapReadAsync();
      firstMatrix = new Float32Array(arrayBuffer);
    }

    return firstMatrix;
}




// This is the original function, for reference
export const matrix_multiplication_from_tutorial = async function() {

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    console.log("Device: ",device);


    // First Matrix
    const firstMatrix = new Float32Array([
      2 /* rows */, 4 /* columns */,
      1, 2, 3, 4,
      5, 6, 7, 8
    ]);
    const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
      size: firstMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
    gpuBufferFirstMatrix.unmap();


    // Second Matrix
    const secondMatrix = new Float32Array([
      4 /* rows */, 2 /* columns */,
      1, 2,
      3, 4,
      5, 6,
      7, 8
    ]);
    const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
      size: secondMatrix.byteLength,
      usage: GPUBufferUsage.STORAGE,
    });
    new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
    gpuBufferSecondMatrix.unmap();


    // Result Matrix
    const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
    const resultMatrixBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayout = device.createBindGroupLayout({
      bindings: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          type: "readonly-storage-buffer"
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          type: "readonly-storage-buffer"
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          type: "storage-buffer"
        }
      ]
    });

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      bindings: [
        {
          binding: 0,
          resource: {
            buffer: gpuBufferFirstMatrix
          }
        },
        {
          binding: 1,
          resource: {
            buffer: gpuBufferSecondMatrix
          }
        },
        {
          binding: 2,
          resource: {
            buffer: resultMatrixBuffer
          }
        }
      ]
    });

    const computeShaderCode = `#version 450

      layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
          vec2 size;
          float numbers[];
      } firstMatrix;

      layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
          vec2 size;
          float numbers[];
      } secondMatrix;

      layout(std430, set = 0, binding = 2) buffer ResultMatrix {
          vec2 size;
          float numbers[];
      } resultMatrix;

      void main() {
        resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

        ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
        float result = 0.0;
        for (int i = 0; i < firstMatrix.size.y; i++) {
          int a = i + resultCell.x * int(firstMatrix.size.y);
          int b = resultCell.y + i * int(secondMatrix.size.y);
          result += firstMatrix.numbers[a] * secondMatrix.numbers[b];
        }

        int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
        resultMatrix.numbers[index] = result;
      }
    `;

    const glslang = await glslangModule();

    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      computeStage: {
        module: device.createShaderModule({
          code: glslang.compileGLSL(computeShaderCode, "compute")
        }),
        entryPoint: "main"
      }
    });

    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatch(firstMatrix[0] /* x */, secondMatrix[1] /* y */);
    passEncoder.endPass();

    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
      size: resultMatrixBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode commands for copying buffer to buffer.
    commandEncoder.copyBufferToBuffer(
      resultMatrixBuffer /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      resultMatrixBufferSize /* size */
    );

    // Submit GPU commands.
    const gpuCommands = commandEncoder.finish();
    device.defaultQueue.submit([gpuCommands]);

    // Read buffer.
    const arrayBuffer = await gpuReadBuffer.mapReadAsync();
    console.log(new Float32Array(arrayBuffer));

}

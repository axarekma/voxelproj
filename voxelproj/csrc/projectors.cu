#include <torch/types.h>

// move to header if needed?
#include <cuda.h>
#include <cuda_runtime.h>

#include "include/sp_forward_z0.cuh"
#include "include/sp_backward_z0.cuh"
#include "include/sp_forward_z2.cuh"
#include "include/sp_backward_z2.cuh"

// #include "include/sp_backward_z0_opt.cuh"

bool createTextureObject3D(cudaArray *array, cudaTextureObject_t &texObj)
{
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.addressMode[2] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaError_t err = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    return err == cudaSuccess;
}

void forward_z0(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8})
{
    TORCH_CHECK(in.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(in.is_contiguous(), "Tensor must be contiguous");
    // Ensure tensor is 3D
    TORCH_CHECK(in.dim() == 3, "Tensor must be 3D");

    // // Print dimensions of input and output tensors
    // printf("Input tensor dimensions: [%lld, %lld, %lld]\n",
    //        in.size(0), in.size(1), in.size(2));
    // printf("Output tensor dimensions: [%lld, %lld, %lld]\n",
    //        out.size(0), out.size(1), out.size(2));
    // printf("Angles tensor dimensions: [%lld]\n", angles.size(0));

    int width = in.size(2);
    int height = in.size(1);
    int depth = in.size(0);

    // Create the channel descriptor (float32)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // Create a CUDA array for the 3D texture (x, y, z) dimensions
    cudaArray_t cudaArray;
    cudaError_t err = cudaMalloc3DArray(&cudaArray, &channelDesc, make_cudaExtent(width, height, depth));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA 3D array");
    }

    cudaTextureObject_t tex_volume;
    if (!createTextureObject3D(cudaArray, tex_volume))
    {
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to create texture object");
    }

    // Set up the copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.dstArray = cudaArray;
    copyParams.srcPtr = make_cudaPitchedPtr(
        in.data_ptr(),         // Pointer to tensor data
        width * sizeof(float), // Pitch (bytes per row)
        width,                 // xsize
        height                 // ysize
    );

    // Copy data into the 3D CUDA array
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess)
    {
        cudaDestroyTextureObject(tex_volume);
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to copy data to CUDA 3D array");
    }

    // dim3 blockDim(8, 8, 8);
    dim3 blockDim(block_size[0], block_size[1], block_size[2]);
    dim3 gridDim(num_blocks(out.size(2), blockDim.x),
                 num_blocks(out.size(1), blockDim.y),
                 num_blocks(out.size(0), blockDim.z));

    sp_forward_z0<<<gridDim, blockDim>>>(tex_volume, out.data_ptr<float>(), angles.data_ptr<float>(),
                                         in.size(2), in.size(1), in.size(0),
                                         out.size(2), out.size(1), out.size(0));

    cudaDestroyTextureObject(tex_volume);
    cudaFreeArray(cudaArray);
}

void backward_z0(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8})
{
    TORCH_CHECK(in.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(in.is_contiguous(), "Tensor must be contiguous");
    // Ensure tensor is 3D
    TORCH_CHECK(in.dim() == 3, "Tensor must be 3D");

    int width = in.size(2);
    int height = in.size(1);
    int depth = in.size(0);

    // Create the channel descriptor (float32)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // Create a CUDA array for the 3D texture (x, y, z) dimensions
    cudaArray_t cudaArray;
    cudaError_t err = cudaMalloc3DArray(&cudaArray, &channelDesc, make_cudaExtent(width, height, depth));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA 3D array");
    }

    cudaTextureObject_t tex_proj;
    if (!createTextureObject3D(cudaArray, tex_proj))
    {
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to create texture object");
    }

    // Set up the copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.dstArray = cudaArray;
    copyParams.srcPtr = make_cudaPitchedPtr(
        in.data_ptr(),         // Pointer to tensor data
        width * sizeof(float), // Pitch (bytes per row)
        width,                 // xsize
        height                 // ysize
    );

    // Copy data into the 3D CUDA array
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess)
    {
        cudaDestroyTextureObject(tex_proj);
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to copy data to CUDA 3D array");
    }

    // dim3 blockDim(8, 8, 8);
    dim3 blockDim(block_size[0], block_size[1], block_size[2]);
    dim3 gridDim(num_blocks(out.size(2), blockDim.x),
                 num_blocks(out.size(1), blockDim.y),
                 num_blocks(out.size(0), blockDim.z));

    sp_backward_z0<<<gridDim, blockDim>>>(tex_proj, out.data_ptr<float>(), angles.data_ptr<float>(),
                                          in.size(2), in.size(1), in.size(0),
                                          out.size(2), out.size(1), out.size(0));

    cudaDestroyTextureObject(tex_proj);
    cudaFreeArray(cudaArray);
}

void forward_z2(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8})
{
    TORCH_CHECK(in.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(in.is_contiguous(), "Tensor must be contiguous");
    // Ensure tensor is 3D
    TORCH_CHECK(in.dim() == 3, "Tensor must be 3D");

    int width = in.size(2);
    int height = in.size(1);
    int depth = in.size(0);

    // Create the channel descriptor (float32)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // Create a CUDA array for the 3D texture (x, y, z) dimensions
    cudaArray_t cudaArray;
    cudaError_t err = cudaMalloc3DArray(&cudaArray, &channelDesc, make_cudaExtent(width, height, depth));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA 3D array");
    }

    cudaTextureObject_t tex_volume;
    if (!createTextureObject3D(cudaArray, tex_volume))
    {
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to create texture object");
    }

    // Set up the copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.dstArray = cudaArray;
    copyParams.srcPtr = make_cudaPitchedPtr(
        in.data_ptr(),         // Pointer to tensor data
        width * sizeof(float), // Pitch (bytes per row)
        width,                 // xsize
        height                 // ysize
    );

    // Copy data into the 3D CUDA array
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess)
    {
        cudaDestroyTextureObject(tex_volume);
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to copy data to CUDA 3D array");
    }

    // dim3 blockDim(8, 8, 8);
    dim3 blockDim(block_size[0], block_size[1], block_size[2]);
    dim3 gridDim(num_blocks(out.size(2), blockDim.x),
                 num_blocks(out.size(1), blockDim.y),
                 num_blocks(out.size(0), blockDim.z));

    sp_forward_z2<<<gridDim, blockDim>>>(tex_volume, out.data_ptr<float>(), angles.data_ptr<float>(),
                                         in.size(2), in.size(1), in.size(0),
                                         out.size(2), out.size(1), out.size(0));

    cudaDestroyTextureObject(tex_volume);
    cudaFreeArray(cudaArray);
}

void backward_z2(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8})
{
    TORCH_CHECK(in.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(in.scalar_type() == torch::kFloat, "Only float32 tensors supported");
    TORCH_CHECK(in.is_contiguous(), "Tensor must be contiguous");
    // Ensure tensor is 3D
    TORCH_CHECK(in.dim() == 3, "Tensor must be 3D");

    int width = in.size(2);
    int height = in.size(1);
    int depth = in.size(0);

    // Create the channel descriptor (float32)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    // Create a CUDA array for the 3D texture (x, y, z) dimensions
    cudaArray_t cudaArray;
    cudaError_t err = cudaMalloc3DArray(&cudaArray, &channelDesc, make_cudaExtent(width, height, depth));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA 3D array");
    }

    cudaTextureObject_t tex_proj;
    if (!createTextureObject3D(cudaArray, tex_proj))
    {
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to create texture object");
    }

    // Set up the copy parameters
    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent = make_cudaExtent(width, height, depth);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    copyParams.dstArray = cudaArray;
    copyParams.srcPtr = make_cudaPitchedPtr(
        in.data_ptr(),         // Pointer to tensor data
        width * sizeof(float), // Pitch (bytes per row)
        width,                 // xsize
        height                 // ysize
    );

    // Copy data into the 3D CUDA array
    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess)
    {
        cudaDestroyTextureObject(tex_proj);
        cudaFreeArray(cudaArray);
        throw std::runtime_error("Failed to copy data to CUDA 3D array");
    }

    // dim3 blockDim(8, 8, 8);
    dim3 blockDim(block_size[0], block_size[1], block_size[2]);
    dim3 gridDim(num_blocks(out.size(2), blockDim.x),
                 num_blocks(out.size(1), blockDim.y),
                 num_blocks(out.size(0), blockDim.z));

    sp_backward_z2<<<gridDim, blockDim>>>(tex_proj, out.data_ptr<float>(), angles.data_ptr<float>(),
                                          in.size(2), in.size(1), in.size(0),
                                          out.size(2), out.size(1), out.size(0));

    cudaDestroyTextureObject(tex_proj);
    cudaFreeArray(cudaArray);
}

// void backward_z0_opt(torch::Tensor in, torch::Tensor out, torch::Tensor angles, std::vector<int> block_size = {8, 8, 8})
// {
//     TORCH_CHECK(in.is_cuda(), "Tensor must be on CUDA");
//     TORCH_CHECK(in.scalar_type() == torch::kFloat, "Only float32 tensors supported");
//     TORCH_CHECK(in.is_contiguous(), "Tensor must be contiguous");
//     // Ensure tensor is 3D
//     TORCH_CHECK(in.dim() == 3, "Tensor must be 3D");

//     int width = in.size(2);
//     int height = in.size(1);
//     int depth = in.size(0);

//     // Create the channel descriptor (float32)
//     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

//     // Create a CUDA array for the 3D texture (x, y, z) dimensions
//     cudaArray_t cudaArray;
//     cudaError_t err = cudaMalloc3DArray(&cudaArray, &channelDesc, make_cudaExtent(width, height, depth));
//     if (err != cudaSuccess)
//     {
//         throw std::runtime_error("Failed to allocate CUDA 3D array");
//     }

//     cudaTextureObject_t tex_proj;
//     if (!createTextureObject3D(cudaArray, tex_proj))
//     {
//         cudaFreeArray(cudaArray);
//         throw std::runtime_error("Failed to create texture object");
//     }

//     // Set up the copy parameters
//     cudaMemcpy3DParms copyParams = {0};
//     copyParams.extent = make_cudaExtent(width, height, depth);
//     copyParams.kind = cudaMemcpyDeviceToDevice;
//     copyParams.dstArray = cudaArray;
//     copyParams.srcPtr = make_cudaPitchedPtr(
//         in.data_ptr(),         // Pointer to tensor data
//         width * sizeof(float), // Pitch (bytes per row)
//         width,                 // xsize
//         height                 // ysize
//     );

//     // Copy data into the 3D CUDA array
//     err = cudaMemcpy3D(&copyParams);
//     if (err != cudaSuccess)
//     {
//         cudaDestroyTextureObject(tex_proj);
//         cudaFreeArray(cudaArray);
//         throw std::runtime_error("Failed to copy data to CUDA 3D array");
//     }

//     // dim3 blockDim(8, 8, 8);
//     dim3 blockDim(block_size[0], block_size[1], block_size[2]);
//     dim3 gridDim(num_blocks(out.size(2), blockDim.x),
//                  num_blocks(out.size(1), blockDim.y),
//                  num_blocks(out.size(0), blockDim.z));

//     sp_backward_z0_opt<<<gridDim, blockDim>>>(tex_proj, out.data_ptr<float>(), angles.data_ptr<float>(),
//                                               in.size(2), in.size(1), in.size(0),
//                                               out.size(2), out.size(1), out.size(0));

//     cudaDestroyTextureObject(tex_proj);
//     cudaFreeArray(cudaArray);
// }

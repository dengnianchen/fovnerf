#include <device_launch_parameters.h>
#include <glm/glm.hpp>

#define IDX2 glm::uvec2 { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y }
#define IDX3 glm::uvec3 { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z }

__device__ __forceinline__ unsigned int flattenIdx(glm::uvec3 idx3)
{
    return idx3.x + idx3.y * blockDim.x * gridDim.x + idx3.z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}

__device__ __forceinline__ unsigned int flattenIdx()
{
    return flattenIdx(IDX3);
}
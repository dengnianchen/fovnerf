#include "Sampler.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "../utils/cuda.h"

__device__ glm::vec3 _raySphereIntersect(glm::vec3 p, glm::vec3 v, float r, float &o_depth)
{
    float pp = glm::dot(p, p);
    float vv = glm::dot(v, v);
    float pv = glm::dot(p, v);
    o_depth = (sqrtf(pv * pv - vv * (pp - r * r)) - pv) / vv;
    return p + o_depth * v;
}

__device__ float _getAngle(float x, float y)
{
    return -atan(x / y) - (y < 0) * (float)M_PI + 0.5f * (float)M_PI;
}

/**
 * Dispatch with block_size=(n_samples, *), grid_size=(1, nRays/*)
 * Index with (sample_idx, ray_idx)
 */
__global__ void cu_sampleOnRays(float *o_coords, float *o_depths, glm::vec3 *rays, uint nRays,
                                glm::vec3 origin, Range range, bool outputRadius)
{
    glm::uvec3 idx3 = IDX3;
    uint idx = flattenIdx(idx3);
    uint sampleIdx = idx3.x;
    uint rayIdx = idx3.y;
    if (rayIdx >= nRays)
        return;
    float r_reciprocal = range.get(sampleIdx);
    glm::vec3 p = _raySphereIntersect(origin, rays[rayIdx], 1.0f / r_reciprocal, o_depths[idx]);
    glm::vec3 sp(r_reciprocal, _getAngle(p.z, p.x), asin(p.y * r_reciprocal));
    if (outputRadius)
        ((glm::vec3 *)o_coords)[idx] = sp;
    else
        ((glm::vec2 *)o_coords)[idx] = {sp.y, sp.z};
}

/**
 * Dispatch with block_size=(n_samples, *), grid_size=(1, nRays/*)
 * Index with (sample_idx, ray_idx)
 */
__global__ void cu_calcDists(float *o_dists, glm::vec3 *rays, float* depths, uint nRays)
{
    glm::uvec3 idx3 = IDX3;
    uint sampleIdx = idx3.x;
    uint rayIdx = idx3.y;
    uint nSamples = blockDim.x;
    if (rayIdx >= nRays)
        return;
    uint idx = rayIdx * nSamples + sampleIdx;
    if (sampleIdx == nSamples - 1)
        o_dists[idx] = 1.e10f;
    else
        o_dists[idx] = depths[idx + 1] - depths[idx];
    o_dists[idx] *= glm::length(rays[rayIdx]);
}

void Sampler::sampleOnRays(sptr<CudaArray<float>> o_coords, sptr<CudaArray<float>> o_depths,
                           sptr<CudaArray<float>> o_dists, sptr<CudaArray<glm::vec3>> rays,
                           glm::vec3 rayCenter)
{
    dim3 blkSize(_dispRange.steps(), 1024 / _dispRange.steps());
    dim3 grdSize(1, (uint)ceil(rays->n() / (float)blkSize.y));
    CU_INVOKE(cu_sampleOnRays)(*o_coords, *o_depths, *rays, rays->n(), rayCenter, _dispRange,
                               _outputRadius);
    CU_INVOKE(cu_calcDists)(*o_dists, *rays, *o_depths, rays->n());
    CHECK_EX(cudaGetLastError());
}
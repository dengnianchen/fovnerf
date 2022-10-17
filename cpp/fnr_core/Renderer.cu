#include "Renderer.h"
#include "../utils/cuda.h"

/// Dispatch (n_rays, -)
__global__ void cu_render(glm::vec4 *o_colors, float *dists, glm::vec4 *rgbd, uint nSamples,
                          uint nRays, glm::vec4 bgColor)
{
    glm::uvec3 idx3 = IDX3;
    uint rayIdx = idx3.x;
    if (rayIdx >= nRays)
        return;
    glm::vec4 outColor = bgColor;
    for (int si = nSamples - 1; si >= 0; --si)
    {
        int i = rayIdx * nSamples + si;
        glm::vec4 c = rgbd[i];
        c.a = 1.f - exp(-max(c.a, 0.f) * dists[i]);
        outColor = outColor * (1 - c.a) + c * c.a;
    }
    outColor.a = 1.0f;
    o_colors[rayIdx] = outColor;
}

Renderer::Renderer() {}

void Renderer::render(sptr<CudaArray<glm::vec4>> o_colors,
                      sptr<CudaArray<float>> dists,
                      sptr<CudaArray<glm::vec4>> rgbd)
{
    dim3 blkSize(1024);
    dim3 grdSize(ceilDiv(o_colors->n(), blkSize.x));
    CU_INVOKE(cu_render)(*o_colors, *dists, *rgbd, rgbd->n() / o_colors->n(), o_colors->n(), {});
    CHECK_EX(cudaGetLastError());
}
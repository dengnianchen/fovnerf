#pragma once
#include "../utils/common.h"

class Sampler {
public:
    Sampler(glm::vec2 depthRange, unsigned int samples, bool outputRadius)
        : _dispRange(1.0f / depthRange, samples), _outputRadius(outputRadius) {}

    void sampleOnRays(sptr<CudaArray<float>> o_coords, sptr<CudaArray<float>> o_depths,
                      sptr<CudaArray<float>> o_dists, sptr<CudaArray<glm::vec3>> rays,
                      glm::vec3 rayCenter);

private:
    Range _dispRange;
    bool _outputRadius;
};
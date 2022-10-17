#pragma once
#include "../utils/common.h"
#include "Sampler.h"
#include "Encoder.h"
#include "Renderer.h"
#include "FovNeRFCore.h"

class FovNeRF {
public:
    FovNeRF(sptr<FovNeRFCore> net, uint nRays, uint nSamplesPerRay,
            glm::vec2 depthRange, uint encodeDim, uint coordChns);

    void run(sptr<CudaArray<glm::vec4>> o_colors, sptr<CudaArray<glm::vec3>> rays, glm::vec3 origin,
             bool showPerf = false);

    uint nRays() const { return _nRays; }

private:
    uint _nRays;
    uint _nSamplesPerRay;
    uint _coordChns;
    sptr<FovNeRFCore> _net;
    sptr<Sampler> _sampler;
    sptr<Encoder> _encoder;
    sptr<Renderer> _renderer;
    sptr<CudaArray<float>> _coords;
    sptr<CudaArray<float>> _depths;
    sptr<CudaArray<float>> _dists;
    sptr<CudaArray<float>> _encoded;
    sptr<CudaArray<glm::vec4>> _rgbd;

};
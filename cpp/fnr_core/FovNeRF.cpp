#include "FovNeRF.h"

FovNeRF::FovNeRF(sptr<FovNeRFCore> net, uint nRays, uint nSamplesPerRay, glm::vec2 depthRange,
                 uint encodeDim, uint coordChns)
    : _nRays(nRays),
      _nSamplesPerRay(nSamplesPerRay),
      _coordChns(coordChns),
      _net(net),
      _sampler(new Sampler(depthRange, nSamplesPerRay, coordChns == 3)),
      _encoder(new Encoder(encodeDim, coordChns, false)),
      _renderer(new Renderer()) {
    auto nSamples = _nRays * _nSamplesPerRay;
    _coords = sptr<CudaArray<float>>(new CudaArray<float>(nSamples * coordChns));
    _depths = sptr<CudaArray<float>>(new CudaArray<float>(nSamples));
    _dists = sptr<CudaArray<float>>(new CudaArray<float>(nSamples));
    _encoded = sptr<CudaArray<float>>(new CudaArray<float>(nSamples * _encoder->outDim()));
    _rgbd = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(nSamples));
    _net->bindResources(_encoded.get(), _rgbd.get());
}

void FovNeRF::run(sptr<CudaArray<glm::vec4>> o_colors, sptr<CudaArray<glm::vec3>> rays,
                  glm::vec3 origin, bool showPerf) {
    rays = sptr<CudaArray<glm::vec3>>(rays->subArray(0, _nRays));
    o_colors = sptr<CudaArray<glm::vec4>>(o_colors->subArray(0, _nRays));
    CudaEvent eStart, eSampled, eEncoded, eInferred, eRendered;
    cudaEventRecord(eStart);

    _sampler->sampleOnRays(_coords, _depths, _dists, rays, origin);
    cudaEventRecord(eSampled);

    _encoder->encode(_encoded, _coords);
    cudaEventRecord(eEncoded);

    _net->infer();
    cudaEventRecord(eInferred);

    _renderer->render(o_colors, _dists, _rgbd);
    cudaEventRecord(eRendered);

    if (showPerf) {
        CHECK_EX(cudaDeviceSynchronize());

        float timeTotal, timeSample, timeEncode, timeInfer, timeRender;
        cudaEventElapsedTime(&timeTotal, eStart, eRendered);
        cudaEventElapsedTime(&timeSample, eStart, eSampled);
        cudaEventElapsedTime(&timeEncode, eSampled, eEncoded);
        cudaEventElapsedTime(&timeInfer, eEncoded, eInferred);
        cudaEventElapsedTime(&timeRender, eInferred, eRendered);

        std::ostringstream sout;
        sout << "FovNeRF infer: " << timeTotal << "ms (Sample: " << timeSample
             << "ms, Encode: " << timeEncode << "ms, Infer: " << timeInfer
             << "ms, Render: " << timeRender << "ms)";
        Logger::instance.info(sout.str().c_str());
    }
}
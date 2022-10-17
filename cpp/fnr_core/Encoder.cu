#include "Encoder.h"
#include "../utils/cuda.h"

/// idx3.z = 0: x, y, z, sin(x), sin(y), sin(z), cos(x), cos(y), cos(z)
/// idx3.z = 1: sin(2x), sin(2y), sin(2z), cos(2x), cos(2y), cos(2z)
/// ...
/// idx3.z = n_freq-1: sin(2^(n_freq-1)x), sin(2^(n_freq-1)y), sin(2^(n_freq-1)z),
///                    cos(2^(n_freq-1)x), cos(2^(n_freq-1)y), cos(2^(n_freq-1)z)
/// Dispatch (n, in_chns, n_freqs)
__global__ void cu_encode0(float *output, float *input, uint n, uint nFreqs) {
    glm::uvec3 idx3 = IDX3;
    if (idx3.x >= n)
        return;
    uint inChns = blockDim.y;
    uint outChns = inChns * (nFreqs * 2 + 1);
    uint i = idx3.x, chn = idx3.y;
    output[i * outChns + chn] = input[i * inChns + chn];
}

__global__ void cu_encode(float *output, float *input, float *freqs, uint n, bool catInput) {
    glm::uvec3 idx3 = IDX3;
    if (idx3.x >= n)
        return;
    uint offset = (uint)catInput;
    uint inChns = blockDim.y, nFreqs = blockDim.z;
    uint i = idx3.x, chn = idx3.y, freq = idx3.z;
    uint elem = i * inChns + chn;
    uint outChns = inChns * (nFreqs * 2 + offset);
    uint base = i * outChns + chn;
    if (freq == 0 && catInput)
        output[base] = input[elem];
    float x = freqs[freq] * input[elem];
    float s, c;
    __sincosf(x, &s, &c);
    output[base + inChns * (freq * 2 + offset)] = s;
    output[base + inChns * (freq * 2 + offset + 1)] = c;
}

__global__ void cu_encode2(glm::vec2 *output, glm::vec2 *input, float *freqs, uint n) {
    glm::uvec3 idx3 = IDX3;
    if (idx3.x >= n)
        return;
    uint nFreqs = blockDim.y;
    uint i = idx3.x, freq = idx3.y;
    uint outChns = nFreqs * 2 + 1;
    uint base = i * outChns;
    if (freq == 0)
        output[base] = input[i];
    glm::vec2 x = freqs[freq] * input[i];
    glm::vec2 s, c;
    __sincosf(x.x, &s.x, &c.x);
    __sincosf(x.y, &s.y, &c.y);
    output[base + (freq * 2 + 1)] = s;
    output[base + (freq * 2 + 2)] = c;
}

/**
 * @brief
 *
 * @param output encoded data, n x out_chns
 * @param input coord data, n x in_chns
 */
void Encoder::encode(sptr<CudaArray<float>> output, sptr<CudaArray<float>> input) {
    std::ostringstream sout;
    sout << "Encoder => input size: (" << input->n() / _chns << ", " << _chns << "), output size: ("
         << output->n() / outDim() << ", " << outDim() << ")";
    //Logger::instance.info(sout.str());
    uint n = input->n() / _chns;
    dim3 blkSize(1024 / _chns / _multires, _chns, _multires);
    dim3 grdSize(ceilDiv(n, blkSize.x), 1, 1);
    CU_INVOKE(cu_encode)(*output, *input, *_freqs, n, _catInput);
    // blkSize = dim3(1024 / _chns, _chns);
    // grdSize = dim3(ceilDiv(n, blkSize.x), 1, 1);
    // CU_INVOKE(cu_encode0)(*output, *input, n, _multires);
    CHECK_EX(cudaGetLastError());
}

void Encoder::_genFreqArray() {
    float *arr = new float[_multires];
    arr[0] = 1.0f;
    for (auto i = 1; i < _multires; ++i)
        arr[i] = arr[i - 1] * 2.0f;
    _freqs = sptr<CudaArray<float>>(new CudaArray<float>(_multires));
    cudaMemcpy(_freqs->getBuffer(), arr, _multires * sizeof(float), cudaMemcpyHostToDevice);
    delete[] arr;
}

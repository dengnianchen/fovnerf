#pragma once
#include "../utils/common.h"

class Encoder {
public:
    Encoder(unsigned int multires, unsigned int chns, bool catInput)
        : _multires(multires), _chns(chns), _catInput(catInput) {
        _genFreqArray();
    }

    unsigned int outDim() const { return _chns * ((int)_catInput + _multires * 2); }
    void encode(sptr<CudaArray<float>> output, sptr<CudaArray<float>> input);

private:
    unsigned int _multires;
    unsigned int _chns;
    bool _catInput;
    sptr<CudaArray<float>> _freqs;

    void _genFreqArray();
}; 
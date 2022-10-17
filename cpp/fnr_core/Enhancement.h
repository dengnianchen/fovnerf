#pragma once
#include "../utils/common.h"

class Enhancement
{
public:
    Enhancement(glm::uvec2 res, glm::vec2 params);

    void run(sptr<CudaArray<glm::vec4>> imageData);

private:
    glm::uvec2 _res;
    glm::vec2 _params;
    sptr<CudaArray<glm::vec4>> _boxFiltered;

};
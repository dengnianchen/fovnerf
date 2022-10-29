#pragma once
#include "../utils/common.h"
#include "FovNeRF.h"
#include "View.h"
#include "Enhancement.h"

class ImageGen {
public:
    ImageGen(glm::uvec2 res);

    void run(sptr<CudaArray<glm::vec4>> colors);

    GLuint getGlResultTexture() const { return _glResultTexture; }

protected:
    glm::uvec2 _res;
    GLuint _glResultTexture;
    GLuint _glResultBuffer;

    GLuint _createGlResultTexture(glm::uvec2 res);
    GLuint _createGlResultBuffer(unsigned int elements);
};
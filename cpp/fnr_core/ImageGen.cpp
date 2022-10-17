#include "ImageGen.h"

ImageGen::ImageGen(glm::uvec2 res) : _res(res) {
    _glResultBuffer = _createGlResultBuffer(_res.x * _res.y);
    _glResultTexture = _createGlResultTexture(_res);
}

void ImageGen::run(sptr<CudaArray<glm::vec4>> colors) {
    // Copy result from Cuda array to OpenGL buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _glResultBuffer);
    void *bufferData = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    cudaMemcpy(bufferData, colors->getBuffer(), colors->size(), cudaMemcpyDeviceToHost);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Upload data from OpenGL buffer to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _glResultBuffer);
    glBindTexture(GL_TEXTURE_2D, _glResultTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _res.x, _res.y, GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

GLuint ImageGen::_createGlResultTexture(glm::uvec2 res) {
    GLuint textureID;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.x, res.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    return textureID;
}

GLuint ImageGen::_createGlResultBuffer(unsigned int elements) {
    GLuint glBuffer;
    glGenBuffers(1, &glBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, elements * sizeof(glm::vec4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return glBuffer;
}
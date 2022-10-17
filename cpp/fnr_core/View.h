#pragma once
#include "../utils/common.h"


class View {
public:
    View(glm::vec3 t, glm::mat3 r) : _t(t), _r(r) {}

    glm::vec3 t() const { return _t; }
    glm::mat3 r() const { return _r; }

    void transPoints(sptr<CudaArray<glm::vec3>> results, sptr<CudaArray<glm::vec3>> points,
                     sptr<CudaArray<int>> indices = nullptr, bool inverse = false);

    void transVectors(sptr<CudaArray<glm::vec3>> results, sptr<CudaArray<glm::vec3>> vectors,
                      sptr<CudaArray<int>> indices = nullptr, bool inverse = false);

    View getStereoEye(float ipd, Eye eye);

private:
    glm::vec3 _t;
    glm::mat3 _r;
};

class Camera {
public:
    Camera(glm::uvec2 res, float fov);
    Camera(glm::uvec2 res, float fov, glm::vec2 c);

    bool loadMaskData(std::string filepath);
    sptr<CudaArray<glm::vec3>> localRays();
    void getRays(sptr<CudaArray<glm::vec3>> o_rays, View &view, glm::vec3 offset);
    void restoreImage(sptr<CudaArray<glm::vec4>> o_imgData, sptr<CudaArray<glm::vec4>> colors);
    glm::vec2 f() const { return _f; }
    glm::vec2 c() const { return _c; }
    glm::uvec2 res() const { return _res; }
    uint nPixels() const { return _res.x * _res.y; }
    uint nRays() const { return _subsetIndices == nullptr ? nPixels() : (uint)_subsetIndices->n(); }
    sptr<CudaArray<int>> subsetIndices() const { return _subsetIndices; };
    sptr<CudaArray<int>> subsetInverseIndices() const { return _subsetInverseIndices; }

private:
    glm::vec2 _f;
    glm::vec2 _c;
    glm::uvec2 _res;
    sptr<CudaArray<glm::vec3>> _localRays;
    sptr<CudaArray<int>> _subsetIndices;
    sptr<CudaArray<int>> _subsetInverseIndices;

    void _genLocalRays(bool norm);
};

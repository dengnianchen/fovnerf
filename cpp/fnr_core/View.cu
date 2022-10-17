#include "View.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <fstream>
#include "../utils/cuda.h"

__global__ void cu_transPoints(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::vec3 t, glm::mat3 rot_t,
                               uint n) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[idx] * rot_t + t;
}

__global__ void cu_transPoints(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::vec3 t, glm::mat3 rot_t,
                               uint n, int *indices) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[indices[idx]] * rot_t + t;
}

__global__ void cu_transPointsInverse(glm::vec3 *o_pts, glm::vec3 *pts, glm::vec3 t,
                                      glm::mat3 inv_rot_t, uint n) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_pts[idx] = (pts[idx] - t) * inv_rot_t;
}

__global__ void cu_transPointsInverse(glm::vec3 *o_pts, glm::vec3 *pts, glm::vec3 t,
                                      glm::mat3 inv_rot_t, uint n, int *indices) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_pts[idx] = (pts[indices[idx]] - t) * inv_rot_t;
}

__global__ void cu_transVectors(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::mat3 rot_t, uint n) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[idx] * rot_t;
}

__global__ void cu_transVectors(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::mat3 rot_t, uint n,
                                int *indices) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[indices[idx]] * rot_t;
}

__global__ void cu_genLocalRays(glm::vec3 *o_rays, glm::vec2 f, glm::vec2 c, glm::uvec2 res) {
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    o_rays[idx] = glm::vec3((glm::vec2(idx2) - c) / f, -1.0f);
}

__global__ void cu_genLocalRaysNormed(glm::vec3 *o_rays, glm::vec2 f, glm::vec2 c, glm::uvec2 res) {
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    o_rays[idx] = glm::normalize(glm::vec3((glm::vec2(idx2) - c) / f, -1.0f));
}

__global__ void cu_camGetRays(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::vec3 offset, glm::mat3 rot_t, uint n) {
	uint idx = flattenIdx();
	if (idx >= n)
		return;
	o_vecs[idx] = glm::normalize(vecs[idx] + offset) * rot_t;
}

__global__ void cu_camGetRays(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::vec3 offset, glm::mat3 rot_t, uint n,
	int *indices) {
	uint idx = flattenIdx();
	if (idx >= n)
		return;
	o_vecs[idx] = (vecs[indices[idx]] + offset) * rot_t;
}

__global__ void cu_indexedCopy(glm::vec4 *o_colors, glm::vec4 *colors, int *indices, uint n) {
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    int srcIdx = indices[idx];
    o_colors[idx] = srcIdx >= 0 ? colors[srcIdx] : glm::vec4();
}

void View::transPoints(sptr<CudaArray<glm::vec3>> results, sptr<CudaArray<glm::vec3>> points,
                       sptr<CudaArray<int>> indices, bool inverse) {
    glm::mat3 r_t = inverse ? _r : glm::transpose(_r);
    dim3 blkSize(1024);
    dim3 grdSize(ceilDiv(results->n(), blkSize.x));
    if (inverse) {
        if (indices == nullptr)
            CU_INVOKE(cu_transPointsInverse)(*results, *points, _t, r_t, points->n());
        else
            CU_INVOKE(cu_transPointsInverse)(*results, *points, _t, r_t, points->n(), *indices);
    } else {
        if (indices == nullptr)
            CU_INVOKE(cu_transPoints)(*results, *points, _t, r_t, results->n());
        else
            CU_INVOKE(cu_transPoints)(*results, *points, _t, r_t, results->n(), *indices);
    }
}

void View::transVectors(sptr<CudaArray<glm::vec3>> results, sptr<CudaArray<glm::vec3>> vectors,
                        sptr<CudaArray<int>> indices, bool inverse) {
    glm::mat3 r_t = inverse ? _r : glm::transpose(_r);
    dim3 blkSize(1024);
    dim3 grdSize(ceilDiv(results->n(), blkSize.x));
    if (indices == nullptr)
        CU_INVOKE(cu_transVectors)(*results, *vectors, r_t, results->n());
    else
        CU_INVOKE(cu_transVectors)(*results, *vectors, r_t, results->n(), *indices);
}

View View::getStereoEye(float ipd, Eye eye) {
    glm::vec3 eyeOffset((eye == Eye_Left ? -ipd : ipd) / 2.0f, 0.0f, 0.0f);
    return View(_r * eyeOffset + _t, _r);
}

Camera::Camera(glm::uvec2 res, float fov) {
    _f.x = _f.y = 0.5f * res.y / tan(fov * (float)M_PI / 360.0f);
    _f.y *= -1.0f;
    _c = res / 2u;
    _res = res;
}

Camera::Camera(glm::uvec2 res, float fov, glm::vec2 c) {
    _f.x = _f.y = 0.5f * res.y / tan(fov * (float)M_PI / 360.0f);
    _f.y *= -1.0f;
    _c = c;
    _res = res;
}

sptr<CudaArray<glm::vec3>> Camera::localRays() {
    if (_localRays == nullptr)
        _genLocalRays(false);
    return _localRays;
}

bool Camera::loadMaskData(std::string filepath) {
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin)
        return false;
    int n;

    fin.read((char *)&n, sizeof(n));
    std::vector<int> subsetIndicesBuffer(n);
    fin.read((char *)subsetIndicesBuffer.data(), sizeof(int) * n);
    _subsetIndices.reset(new CudaArray<int>(subsetIndicesBuffer));

    fin.read((char *)&n, sizeof(n));
    std::vector<int> subsetInverseIndicesBuffer(n);
    fin.read((char *)subsetInverseIndicesBuffer.data(), sizeof(int) * n);
    _subsetInverseIndices.reset(new CudaArray<int>(subsetInverseIndicesBuffer));

    if (!fin) {
        _subsetIndices = nullptr;
        _subsetInverseIndices = nullptr;
        return false;
    }
    Logger::instance.info("Mask data loaded. Subset indices: %d, subset inverse indices: %d",
        _subsetIndices->n(), _subsetInverseIndices->n());
    return true;
}

void Camera::getRays(sptr<CudaArray<glm::vec3>> o_rays, View &view, glm::vec3 offset) {
	glm::mat3 r_t = glm::transpose(view.r());
	dim3 blkSize(1024);
	dim3 grdSize(ceilDiv(nRays(), blkSize.x));
	if (_subsetIndices == nullptr)
		CU_INVOKE(cu_camGetRays)(*o_rays, *localRays(), offset, r_t, nRays());
	else
		CU_INVOKE(cu_camGetRays)(*o_rays, *localRays(), offset, r_t, nRays(), *_subsetIndices);
}

void Camera::restoreImage(sptr<CudaArray<glm::vec4>> o_imgData, sptr<CudaArray<glm::vec4>> colors) {
    if (_subsetInverseIndices == nullptr) {
        cudaMemcpy(o_imgData->getBuffer(), colors->getBuffer(), nPixels() * sizeof(glm::vec4),
                   cudaMemcpyDeviceToDevice);
    } else {
        dim3 blkSize(1024);
        dim3 grdSize(ceilDiv(nPixels(), blkSize.x));
        CU_INVOKE(cu_indexedCopy)(*o_imgData, *colors, *_subsetInverseIndices, nPixels());
    }
}

void Camera::_genLocalRays(bool norm) {
    _localRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(nPixels()));
    dim3 blkSize(32, 32);
    dim3 grdSize(ceilDiv(_res.x, blkSize.x), ceilDiv(_res.y, blkSize.y));
    if (norm)
        CU_INVOKE(cu_genLocalRaysNormed)(*_localRays, _f, _c, _res);
    else
        CU_INVOKE(cu_genLocalRays)(*_localRays, _f, _c, _res);
}

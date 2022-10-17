#include "Enhancement.h"
#include "../utils/cuda.h"

#define max(__a__, __b__) (__a__ > __b__ ? __a__ : __b__)
#define min(__a__, __b__) (__a__ < __b__ ? __a__ : __b__)

__global__ void cu_boxFilter(glm::vec4 *o_filtered, glm::vec4 *imageData, glm::uvec2 res) {
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    glm::vec4 c (0.0f, 0.0f, 0.0f, 0.0f);
    float n = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            glm::ivec2 idx2_ = (glm::ivec2)idx2 + glm::ivec2(dx, dy);
			if (idx2_.x < 0 || idx2_.x >= res.x || idx2_.y < 0 || idx2_.y >= res.y)
				continue;
            int idx_ = idx2_.x + idx2_.y * res.x;
            c += imageData[idx_];
            n += 1.0f;
        }
    }
	if (n < 0.5f)
		o_filtered[idx] = glm::vec4();
	else
		o_filtered[idx] = c / n;
}

__global__ void cu_constrastEnhance(glm::vec4 *io_imageData, glm::vec4 *filtered, float cScale,
                                    glm::uvec2 res) {
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
	glm::vec4 c = filtered[idx] + (io_imageData[idx] - filtered[idx]) * cScale;
    io_imageData[idx].r = min(max(c.r, 0.0f), 1.0f);
    io_imageData[idx].g = min(max(c.g, 0.0f), 1.0f);
    io_imageData[idx].b = min(max(c.b, 0.0f), 1.0f);
}

Enhancement::Enhancement(glm::uvec2 res, glm::vec2 params)
    : _res(res), _boxFiltered(new CudaArray<glm::vec4>(res.x * res.y)), _params(params) {}

void Enhancement::run(sptr<CudaArray<glm::vec4>> imageData) {
    dim3 blkSize(32, 32);
    dim3 grdSize(ceilDiv(_res.x, blkSize.x), ceilDiv(_res.y, blkSize.y));
    CU_INVOKE(cu_boxFilter)(*_boxFiltered, *imageData, _res);
    CU_INVOKE(cu_constrastEnhance)(*imageData, *_boxFiltered, 1.0f + _params[0] * _params[1], _res);
}
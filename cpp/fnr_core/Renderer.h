#pragma once
#include "../utils/common.h"

class Renderer
{
public:
    Renderer();

    /**
     * @brief
     *
     * @param o_colors
     * @param dists
     * @param layeredRGBDs
     */
    void render(sptr<CudaArray<glm::vec4>> o_colors, sptr<CudaArray<float>> dists,
                sptr<CudaArray<glm::vec4>> rgbd);
};
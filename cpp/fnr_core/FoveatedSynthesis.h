#pragma once
#include "../utils/common.h"
#include "View.h"

class FoveatedSynthesis_Impl;

class FoveatedSynthesis {
public:
	FoveatedSynthesis(const std::string& modelDir, sptr<Camera> cam,
		              const std::vector<sptr<Camera>>& layerCams, bool stereo = false);

	void run(View& view, glm::vec2 foveaPos, bool showPerf = false, glm::vec2 foveaPosR = {});

	GLuint getGlResultTexture(uint index) const;
	glm::vec2 getDepthRange() const;

private:
	sptr<FoveatedSynthesis_Impl> _impl;

};
#pragma once
#include "../utils/common.h"
#include "../utils/Shader.h"
#include "View.h"


class FoveatedBlend {
public:
	FoveatedBlend(sptr<Camera> cam, const std::vector<sptr<Camera>>& layerCams, bool forBaked = false);

	void run(GLuint glTexs[], glm::vec2 foveaPos, float shift, float globalShift, bool showPerf = false);

private:
	bool _forBaked;
	sptr<Camera> _cam;
	std::vector<sptr<Camera>> _layerCams;
	sptr<Shader> _blendShader;
	GLuint _vertBuf;
	GLuint _shaderProp_tex;
	GLuint _shaderProp_innerR;
	GLuint _shaderProp_outerR;
	GLuint _shaderProp_shift;
	GLuint _shaderProp_globalShift;
	GLuint _shaderProp_frameRes;
	GLuint _shaderProp_foveaCenter;
	GLuint _glQuery;
	GLuint _loc_MVP;

};
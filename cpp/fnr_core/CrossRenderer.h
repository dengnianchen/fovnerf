#pragma once
#include "../utils/common.h"
#include "../utils/Shader.h"


class CrossRenderer {
public:
	CrossRenderer(glm::vec2 frameRes, float crossSize, glm::vec4 crossColor);

	void render(glm::vec2 p, float globalShift);

private:
	glm::vec2 _frameRes;
	float _crossSize;
	glm::vec4 _crossColor;
	sptr<Shader> _shader;
	GLuint _vertBuf;
	GLuint _shaderProp_crossSize;
	GLuint _shaderProp_crossColor;
	GLuint _shaderProp_crossPos;
	GLuint _shaderProp_frameRes;
	GLuint _shaderProp_globalShift;
	GLuint _loc_MVP;

};
#include "CrossRenderer.h"
#include <glm/gtx/transform.hpp>


static const struct {
	float x, y;
	float u, v;
} vertices[4] = {
	{-1.0f, -1.0f, 0.f, 1.f},
	{1.0f, -1.0f, 1.f, 1.f},
	{1.0f, 1.0f, 1.f, 0.f},
	{-1.0f, 1.0f, 0.f, 0.f}
};

static const char* vertex_shader_text =
"#version 300 es\n"
"uniform mat4 MVP;\n"
"in vec2 vUV;\n"
"in vec2 vPos;\n"
"out vec2 uv;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
"    uv = vUV;\n"
"}\n";

static const char* fragment_shader_text =
"#version 300 es\n"
"precision mediump float;\n"
"out vec4 FragColor;\n"
"in vec2 uv;\n"
"uniform sampler2D tex;\n"
"uniform float crossSize;\n"
"uniform vec4 crossColor;\n"
"uniform vec2 crossPos;\n"
"uniform vec2 frameRes;\n"
"uniform float globalShift;\n"
"void main()\n"
"{\n"
"    vec2 p = uv * frameRes;\n"
"    p.x += globalShift;\n"
"    if (p.x >= crossPos.x - crossSize && p.x <= crossPos.x + crossSize && p.y >= crossPos.y - 1.0 && p.y <= crossPos.y + 1.0)\n"
"        FragColor = crossColor;\n"
"    else if (p.x >= crossPos.x - 1.0 && p.x <= crossPos.x + 1.0 && p.y >= crossPos.y - crossSize && p.y <= crossPos.y + crossSize)\n"
"        FragColor = crossColor;\n"
"    else\n"
"        FragColor = vec4(0, 0, 0, 0);\n"
"}\n";


CrossRenderer::CrossRenderer(glm::vec2 frameRes, float crossSize, glm::vec4 crossColor) :
	_frameRes(frameRes),
	_crossSize(crossSize),
	_crossColor(crossColor)
{
	_shader.reset(new Shader("Cross", vertex_shader_text, fragment_shader_text));
	_shader->compile();
	glGenBuffers(1, &_vertBuf);
	glBindBuffer(GL_ARRAY_BUFFER, _vertBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	auto program = _shader->getId();
	_shaderProp_crossSize = glGetUniformLocation(program, "crossSize");
	_shaderProp_crossColor = glGetUniformLocation(program, "crossColor");
	_shaderProp_crossPos = glGetUniformLocation(program, "crossPos");
	_shaderProp_frameRes = glGetUniformLocation(program, "frameRes");
	_shaderProp_globalShift = glGetUniformLocation(program, "globalShift");

	_loc_MVP = glGetUniformLocation(program, "MVP");
	auto loc_vPos = glGetAttribLocation(program, "vPos");
	auto loc_vUV = glGetAttribLocation(program, "vUV");

	glEnableVertexAttribArray(loc_vPos);
	glVertexAttribPointer(loc_vPos, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*)0);
	glEnableVertexAttribArray(loc_vUV);
	glVertexAttribPointer(loc_vUV, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]),
		(void*)(sizeof(float) * 2));
}


void CrossRenderer::render(glm::vec2 p, float globalShift)
{
	glm::mat4 mvp = glm::ortho(-1.f, 1.f, -1.f, 1.f, 1.f, -1.f);

	glUseProgram(_shader->getId());
	glUniformMatrix4fv(_loc_MVP, 1, GL_FALSE, (float*)&mvp[0][0]);

	glUniform1f(_shaderProp_crossSize, _crossSize);
	glUniform4fv(_shaderProp_crossColor, 1, &_crossColor[0]);
	glUniform2fv(_shaderProp_crossPos, 1, &p[0]);
	glUniform2fv(_shaderProp_frameRes, 1, &_frameRes[0]);
	glUniform1f(_shaderProp_globalShift, globalShift);
	glDrawArrays(GL_QUADS, 0, 4);
}

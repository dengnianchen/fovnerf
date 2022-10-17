#include "FoveatedBlend.h"
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
"precision highp float;\n"
"out vec4 FragColor;\n"
"in vec2 uv;\n"
"uniform sampler2D tex;\n"
"uniform float innerR;\n"
"uniform float outerR;\n"
"uniform float shift;\n"
"uniform float globalShift;\n"
"uniform vec2 foveaCenter;\n"
"uniform vec2 frameRes;\n"
"void main()\n"
"{\n"
"    vec2 u = uv;"
"    FragColor = texture(tex, u);"
"    u.x += (shift + globalShift) / frameRes.x;\n"
"    if(outerR < 1e-2) {\n"
"        FragColor = texture(tex, u);\n"
"        return;\n"
"    }\n"
"    vec2 p = u * frameRes;\n"
"    float r = distance(p, foveaCenter);\n"
"    vec2 coord = (p - foveaCenter) / outerR / 2.0 + 0.5;\n"
"    FragColor = vec4(coord, 0, 1);\n"
"    if(coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) {\n"
"        FragColor = vec4(0, 0, 0, 0);\n"
"        return;\n"
"    }\n"
"    vec4 c = texture(tex, coord);\n"
"    float alpha = 1.0 - smoothstep((outerR - innerR) * 0.6 + innerR, outerR, r);\n"
"    c.a = c.a * alpha;\n"
"    FragColor = c;\n"
"}\n";

static const char* fragment_shader_text_baked =
"#version 300 es\n"
"precision highp float;\n"
"out vec4 FragColor;\n"
"in vec2 uv;\n"
"uniform sampler2D tex;\n"
"uniform float innerR;\n"
"uniform float outerR;\n"
"uniform float shift;\n"
"uniform float globalShift;\n"
"uniform vec2 foveaCenter;\n"
"uniform vec2 frameRes;\n"
"void main()\n"
"{\n"
"    vec2 u = uv;"
"    u.x += (shift + globalShift) / frameRes.x;\n"
"    if(outerR < 1e-2) {\n"
"        FragColor = texture(tex, u);\n"
"        return;\n"
"    }\n"
"    vec2 p = u * frameRes;\n"
"    float r = distance(p, foveaCenter);\n"
"    vec2 coord = (p - foveaCenter) / outerR / 2.0 + 0.5;\n"
"    if(coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) {\n"
"        FragColor = vec4(0, 0, 0, 0);\n"
"        return;\n"
"    }\n"
"    vec4 c = texture(tex, u);\n"
"    float alpha = 1.0 - smoothstep((outerR - innerR) * 0.6 + innerR, outerR, r);\n"
"    c.a = c.a * alpha;\n"
"    FragColor = c;\n"
"}\n";

FoveatedBlend::FoveatedBlend(sptr<Camera> cam, const std::vector<sptr<Camera>>& layerCams, bool forBaked) :
	_cam(cam), _layerCams(layerCams), _forBaked(forBaked) {
	_blendShader.reset(new Shader("FoveaBlend", vertex_shader_text,
		forBaked ? fragment_shader_text_baked : fragment_shader_text));
	_blendShader->compile();
	glGenBuffers(1, &_vertBuf);
	glBindBuffer(GL_ARRAY_BUFFER, _vertBuf);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	auto program = _blendShader->getId();
	_shaderProp_tex = glGetUniformLocation(program, "tex");
	_shaderProp_innerR = glGetUniformLocation(program, "innerR");
	_shaderProp_outerR = glGetUniformLocation(program, "outerR");
	_shaderProp_shift = glGetUniformLocation(program, "shift");
	_shaderProp_globalShift = glGetUniformLocation(program, "globalShift");
	_shaderProp_frameRes = glGetUniformLocation(program, "frameRes");
	_shaderProp_foveaCenter = glGetUniformLocation(program, "foveaCenter");

	_loc_MVP = glGetUniformLocation(program, "MVP");
	auto loc_vPos = glGetAttribLocation(program, "vPos");
	auto loc_vUV = glGetAttribLocation(program, "vUV");

	glEnableVertexAttribArray(loc_vPos);
	glVertexAttribPointer(loc_vPos, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]), (void*)0);
	glEnableVertexAttribArray(loc_vUV);
	glVertexAttribPointer(loc_vUV, 2, GL_FLOAT, GL_FALSE, sizeof(vertices[0]),
		(void*)(sizeof(float) * 2));

	glGenQueries(1, &_glQuery);
}


void FoveatedBlend::run(GLuint glTexs[], glm::vec2 foveaPos, float shift, float globalShift, bool showPerf) {
	glm::mat4 mvp = glm::ortho(-1.f, 1.f, -1.f, 1.f, 1.f, -1.f);

	glBeginQuery(GL_TIME_ELAPSED, _glQuery);

	glUseProgram(_blendShader->getId());
	glUniformMatrix4fv(_loc_MVP, 1, GL_FALSE, (float*)&mvp[0][0]);
	glUniform1i(_shaderProp_tex, 0);
	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);

	int i = _layerCams.size() - 1;
	
	glUniform1f(_shaderProp_outerR, 0.0f);
	glUniform1f(_shaderProp_shift, shift);
	glUniform1f(_shaderProp_globalShift, globalShift);
	glUniform2f(_shaderProp_frameRes, _cam->res().x, _cam->res().y);
	glUniform2f(_shaderProp_foveaCenter, _cam->res().x / 2.0f, _cam->res().y / 2.0f);
	glBindTexture(GL_TEXTURE_2D, glTexs[i]);
	glDrawArrays(GL_QUADS, 0, 4);
	
	for (i -= 1; i >= 0; --i) {
		auto outerR = _layerCams[i]->res().y / _layerCams[i]->f().y * _cam->f().y * 0.5f;
		auto innerR = i == 0 ?
			0.0f :
			_layerCams[i - 1]->res().y / _layerCams[i - 1]->f().y * _cam->f().y * 0.5f;
		glUniform1f(_shaderProp_outerR, outerR);
		glUniform1f(_shaderProp_shift, _forBaked ? shift : 0.0f);
		glUniform1f(_shaderProp_globalShift, globalShift);
		glUniform2f(_shaderProp_frameRes, _cam->res().x, _cam->res().y);
		glUniform2f(_shaderProp_foveaCenter, foveaPos.x, foveaPos.y);
		glBindTexture(GL_TEXTURE_2D, glTexs[i]);
		glDrawArrays(GL_QUADS, 0, 4);
	}

	glDisable(GL_TEXTURE_2D);

	glEndQuery(GL_TIME_ELAPSED);

	if (showPerf) {
		GLint available = 0;
		while (!available)
			glGetQueryObjectiv(_glQuery, GL_QUERY_RESULT_AVAILABLE, &available);
	
		// timer queries can contain more than 32 bits of data, so always
		// query them using the 64 bit types to avoid overflow
		GLuint64 timeElapsed = 0;
		glGetQueryObjectui64v(_glQuery, GL_QUERY_RESULT, &timeElapsed);

		Logger::instance.info("Blend: %fms", timeElapsed / 1000000.0f);
	}
}
#include "Shader.h"
#include "Logger.h"


Shader::Shader(const std::string& name, const std::string& vertProg, const std::string& fragProg) :
	_name(name),
	_vertProg(vertProg),
	_fragProg(fragProg),
	_id(0) {
}


bool Shader::compile() {
	GLuint vertex_shader, fragment_shader;
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	auto vertProgCstr = _vertProg.c_str();
	glShaderSource(vertex_shader, 1, &vertProgCstr, NULL);
	glCompileShader(vertex_shader);
	bool success = _checkCompileErrors(vertex_shader, "VERTEX");

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	auto fragProgCstr = _fragProg.c_str();
	glShaderSource(fragment_shader, 1, &fragProgCstr, NULL);
	glCompileShader(fragment_shader);
	success = success && _checkCompileErrors(fragment_shader, "FRAGMENT");

	if (!success) {
		_id = 0;
		return false;
	}

	_id = glCreateProgram();
	glAttachShader(_id, vertex_shader);
	glAttachShader(_id, fragment_shader);
	glLinkProgram(_id);
	if (!_checkCompileErrors(_id, "PROGRAM")) {
		_id = 0;
		return false;
	}

	Logger::instance.info("Shader program is loaded");
	return true;
}


bool Shader::_checkCompileErrors(GLuint id, const std::string& type) {
	int success;
	char infoLog[1024];
	if (type != "PROGRAM") {
		glGetShaderiv(id, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(id, 1024, NULL, infoLog);
			Logger::instance.error("Shader compilation error of %s:\n%s\n", type.c_str(), infoLog);
		}
	} else {
		glGetProgramiv(id, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(id, 1024, NULL, infoLog);
			Logger::instance.error("Shader program linking error:\n%s\n", infoLog);
		}
	}
	return (bool)success;
}
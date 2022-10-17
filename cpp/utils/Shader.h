#pragma once
#include<GL/glew.h>
#include <string>


class Shader {
public:
	Shader(const std::string& name, const std::string& vertProg, const std::string& fragProg);

	const std::string& getName() const { return _name; }

	GLuint getId() const { return _id; }

	bool compile();

private:
	std::string _name;
	GLuint _id;
	std::string _vertProg;
	std::string _fragProg;

	bool _checkCompileErrors(GLuint id, const std::string& type);

};
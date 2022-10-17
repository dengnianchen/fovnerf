#pragma once
#include "common.h"


class Texture2D {
public:
	Texture2D(const char* path);
	~Texture2D();

	GLuint getId() const { return _id; }

private:
	GLuint _id;
};


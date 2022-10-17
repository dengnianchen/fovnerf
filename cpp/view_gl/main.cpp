#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>
#include "../utils/Shader.h"
#include "../fnr_core/FoveatedSynthesis.h"
#include "../fnr_core/FoveatedBlend.h"
#include "../fnr_core/CrossRenderer.h"

#define STEREO_DISABLED 0
#define STEREO_OPTI 1
#define STEREO_NORMAL 2

// Command arguments
bool showPerf = false;
int stereo = STEREO_DISABLED;
std::string modelDir;

glm::vec2 frameRes(1440.0f, 1600.0f);
float windowScale = 0.5f;
glm::vec3 trs;
glm::vec3 rot_angle;
glm::mat3 rot;
glm::vec2 fovea_pos = frameRes / 2.0f;
glm::vec2 fovea_pos_r = frameRes / 2.0f;
float disparity = 0.0f;
bool mouse_button_down[] = {false, false, false};
glm::vec2 mouse_pos;
bool enable_shift = true;

static void error_callback(int error, const char *description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	if (action == GLFW_PRESS || action == GLFW_REPEAT)
	{
		float speed = 0.005f;
		glm::vec3 dt;
		switch (key)
		{
		case GLFW_KEY_RIGHT:
			dt.x += speed;
			break;
		case GLFW_KEY_LEFT:
			dt.x -= speed;
			break;
		case GLFW_KEY_UP:
			dt.y += speed;
			break;
		case GLFW_KEY_DOWN:
			dt.y -= speed;
			break;
		case GLFW_KEY_PAGE_UP:
			dt.z += speed;
			break;
		case GLFW_KEY_PAGE_DOWN:
			dt.z -= speed;
			break;
		case GLFW_KEY_S:
			enable_shift = !enable_shift;
			break;
		case GLFW_KEY_SPACE:
			trs = {};
			rot_angle = {};
			rot = {};
			fovea_pos = fovea_pos_r = frameRes / 2.0f;
			disparity = 0.0f;
		}
		trs += rot * dt;
	}
}

void setFoveaPos(glm::vec2 mousePos)
{
	mousePos /= windowScale;
	if (stereo)
	{
		if (mousePos.x >= frameRes.x)
		{
			fovea_pos_r = mousePos - glm::vec2(frameRes.x, 0.0f);
			fovea_pos = glm::vec2(fovea_pos_r.x + disparity, fovea_pos_r.y);
		}
		else
		{
			fovea_pos = mousePos;
			fovea_pos_r = glm::vec2(fovea_pos.x - disparity, fovea_pos.y);
		}
	}
	else
		fovea_pos = mousePos;
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	if (button <= 2)
		mouse_button_down[button] = (action != GLFW_RELEASE);
	if (mouse_button_down[0])
		setFoveaPos(mouse_pos);
}

void mouse_scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
	if (!stereo)
		return;
	disparity += (float)yoffset * 0.5f;
	auto avg_pos = (fovea_pos + fovea_pos_r) / 2.0f;
	fovea_pos = avg_pos + glm::vec2(disparity / 2.0f, 0.0f);
	fovea_pos_r = avg_pos - glm::vec2(disparity / 2.0f, 0.0f);
}

void cursor_position_callback(GLFWwindow *window, double x, double y)
{
	glm::vec2 cur_mouse_pos((float)x, (float)y);
	if (mouse_button_down[1])
	{
		float speed = 0.1f;
		auto dpos = cur_mouse_pos - mouse_pos;
		rot_angle += glm::vec3(dpos.y * speed, dpos.x * speed, 0.0f);
		rot_angle.x = std::max(std::min(rot_angle.x, 80.0f), -80.0f);
		rot = glm::mat3_cast(glm::quat(glm::radians(rot_angle)));
	}
	if (mouse_button_down[0])
	{
		setFoveaPos(cur_mouse_pos);
	}
	mouse_pos = cur_mouse_pos;
}

GLFWwindow *initGl(glm::uvec2 windowRes)
{
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		return nullptr;
	GLFWwindow *window = glfwCreateWindow(windowRes.x, windowRes.y,
										  "Foveated Neueral Rendering", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return nullptr;
	}
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetScrollCallback(window, mouse_scroll_callback);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glewInit();
	glViewport(0, 0, windowRes.x, windowRes.y);
	glClearColor(0.0f, 0.0f, 0.3f, 1.0f);

	Logger::instance.info("OpenGL is initialized");

	return window;
}

void printHelp()
{
	std::cout << "Usage: view_gl [-s|-fs] [-p] [-h] model_dir" << std::endl;
}

void parseCommandArguments(int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
	{
		std::string arg(argv[i]);
		if (arg == "--help" || arg == "-h")
		{
			printHelp();
			exit(0);
		}
		else if (arg == "--stereo" || arg == "-s")
			stereo = STEREO_OPTI;
		else if (arg == "--fullstereo" || arg == "-fs")
			stereo = STEREO_NORMAL;
		else if (arg == "--perf" || arg == "-p")
			showPerf = true;
		else if (arg[0] != '-')
			modelDir = arg;
		else
		{
			printHelp();
			std::cerr << "Unknown option: " + arg << std::endl;
			exit(-1);
		}
	}
	if (modelDir == "")
	{
		printHelp();
		std::cerr << "Require positional argument: data_dir" << std::endl;
		exit(-1);
	}
}

int main(int argc, char **argv)
{
	Logger::instance.logLevel = 3;
	parseCommandArguments(argc, argv);

	glm::uvec2 windowRes = frameRes * windowScale;
	if (stereo)
		windowRes.x *= 2;

	GLFWwindow *window = initGl(windowRes);

	sptr<Camera> _cam;
	std::vector<sptr<Camera>> _layerCams;
	_layerCams.resize(3);
	_layerCams[0].reset(new Camera(glm::uvec2(256, 256), 20)); // 406 -> 406
	_layerCams[0]->loadMaskData("../nets/fovea.mask");
	_layerCams[1].reset(new Camera(glm::uvec2(256, 256), 45)); // 784 -> 392
	_layerCams[1]->loadMaskData("../nets/mid.mask");
	_layerCams[2].reset(new Camera(glm::uvec2(230, 256), 110)); // 1600 -> 400
	_cam.reset(new Camera(frameRes, 110));

	sptr<CrossRenderer> crossRenderer(new CrossRenderer(frameRes, 10, {0, 1, 0, 1}));
	sptr<FoveatedBlend> foveatedBlend(new FoveatedBlend(_cam, _layerCams));

	Logger::instance.info("Load model from %s", modelDir.c_str());
	sptr<FoveatedSynthesis> syn(new FoveatedSynthesis(modelDir, _cam, _layerCams, stereo == STEREO_OPTI));
	Logger::instance.info("Start main loop");

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);
		auto view = View(trs, rot);

		if (stereo == STEREO_NORMAL)
		{
			{
				auto view_l = view.getStereoEye(0.06, Eye_Left);
				syn->run(view, fovea_pos, showPerf);
				GLuint glTexs[] = {
					syn->getGlResultTexture(0),
					syn->getGlResultTexture(1),
					syn->getGlResultTexture(2)};
				glViewport(0, 0, windowRes.x / 2, windowRes.y);
				foveatedBlend->run(glTexs, fovea_pos, 0.0f, showPerf);
				crossRenderer->render(fovea_pos, 0.0f);
			}
			{
				auto view_r = view.getStereoEye(0.06, Eye_Right);
				syn->run(view, fovea_pos_r, showPerf);
				GLuint glTexs[] = {
					syn->getGlResultTexture(0),
					syn->getGlResultTexture(1),
					syn->getGlResultTexture(2)};
				glViewport(windowRes.x / 2, 0, windowRes.x / 2, windowRes.y);
				foveatedBlend->run(glTexs, fovea_pos_r, 0.0f, 0.0f, showPerf);
				crossRenderer->render(fovea_pos_r, 0.0f);
			}
		}
		else
		{
			syn->run(view, fovea_pos, showPerf, fovea_pos_r);

			if (stereo == STEREO_DISABLED)
			{
				GLuint glTexs[] = {
					syn->getGlResultTexture(0),
					syn->getGlResultTexture(1),
					syn->getGlResultTexture(2)};
				foveatedBlend->run(glTexs, fovea_pos, 0.0f, 0.0f, showPerf);
				crossRenderer->render(fovea_pos, 0.0f);
			}
			else
			{
				GLuint glTexsL[] = {
					syn->getGlResultTexture(0),
					syn->getGlResultTexture(1),
					syn->getGlResultTexture(2)};
				GLuint glTexsR[] = {
					syn->getGlResultTexture(3),
					syn->getGlResultTexture(1),
					syn->getGlResultTexture(2)};
				float shift = (fovea_pos.x - fovea_pos_r.x) / 2.0f;
				glViewport(0, 0, windowRes.x / 2, windowRes.y);
				foveatedBlend->run(glTexsL, fovea_pos, -shift * enable_shift, 0.0f, showPerf);
				crossRenderer->render(fovea_pos, 0.0f);
				glViewport(windowRes.x / 2, 0, windowRes.x / 2, windowRes.y);
				foveatedBlend->run(glTexsR, fovea_pos_r, shift * enable_shift, 0.0f, showPerf);
				crossRenderer->render(fovea_pos_r, 0.0f);
			}
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();
	exit(EXIT_SUCCESS);
}

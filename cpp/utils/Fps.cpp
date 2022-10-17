#include "Fps.h"
#include <time.h>
#include "Logger.h"


Fps::Fps() {
	_t = clock() / (float)CLOCKS_PER_SEC;
	_n_frames = 0;
}


void Fps::update() {
	auto t = clock() / (float)CLOCKS_PER_SEC;
	_n_frames++;
	if (t - _t >= 0.5f) {
		Logger::instance.info("Frames/Sec: %.1f", _n_frames / (t - _t));
		_t = t;
		_n_frames = 0;
	}
}
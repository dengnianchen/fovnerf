#pragma once
#include <stdarg.h>
#include <iostream>
#include <string>
#include <sstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace nv = nvinfer1;


typedef void(*ExternalLogFuncPtr)(int severity, const char*);


class Logger : public nv::ILogger {
public:
	ExternalLogFuncPtr externalLogFunc = nullptr;
	int logLevel = 1;
	static Logger instance;

	void verbose(const char* fmt, ...) {
		va_list args;
		va_start(args, fmt);
		logf(nv::ILogger::Severity::kVERBOSE, fmt, args);
		va_end(args);
	}

	void info(const char* fmt, ...) {
		va_list args;
		va_start(args, fmt);
		logf(nv::ILogger::Severity::kINFO, fmt, args);
		va_end(args);
	}

	void warning(const char* fmt, ...) {
		va_list args;
		va_start(args, fmt);
		logf(nv::ILogger::Severity::kWARNING, fmt, args);
		va_end(args);
	}

	void error(const char* fmt, ...) {
		va_list args;
		va_start(args, fmt);
		logf(nv::ILogger::Severity::kERROR, fmt, args);
		va_end(args);
	}

	bool checkErr(cudaError_t err, const char* file, int line) {
		if (err == cudaSuccess)
			return true;
		error("Cuda error %s at %s (Line %d): %s", cudaGetErrorName(err), file, line,
			cudaGetErrorString(err));
		return false;
	}

	virtual void log(nv::ILogger::Severity severity, const char* msg) noexcept {
		if ((int)severity > logLevel)
			return;
		if (externalLogFunc == nullptr) {
			switch (severity) {
			case nv::ILogger::Severity::kVERBOSE:
				std::cout << "[VERBOSE] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kINFO:
				std::cout << "[INFO] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kWARNING:
				std::cerr << "[WARNING] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kERROR:
				std::cerr << "[ERROR] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "[ERROR] " << msg << std::endl;
				break;
			}
			return;
		}
		externalLogFunc((int)severity, msg);
	}

	void logf(nv::ILogger::Severity severity, const char* fmt, va_list args) {
		char buffer[4096];
		vsprintf(buffer, fmt, args);
		log(severity, buffer);
	}

};


#define CHECK(__ERR_CODE__) do { if (!Logger::instance.checkErr((__ERR_CODE__), __FILE__, __LINE__)) return false; } while (0)
#define CHECK_EX(__ERR_CODE__) do { if (!Logger::instance.checkErr((__ERR_CODE__), __FILE__, __LINE__)) throw std::exception(); } while (0)

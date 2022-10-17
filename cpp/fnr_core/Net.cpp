#include "../utils/half.h"
#include "Net.h"
#include <fstream>
#include <numeric>
#include <assert.h>
#include <time.h>
#include <NvOnnxParser.h>

bool Net::load(const std::string &path)
{
	_deserialize(path);
	if (!mEngine)
	{
		Logger::instance.error("Failed to build net: failed to load engine.");
		return false;
	}

	mContext = std::shared_ptr<nv::IExecutionContext>(
		mEngine->createExecutionContext(), Destroy<nv::IExecutionContext>());
	if (!mContext)
		return false;

	std::ostringstream sout;
	sout << "NbBindings: " << mEngine->getNbBindings() << std::endl;
	for (auto i = 0; i < mEngine->getNbBindings(); ++i)
	{
		auto name = mEngine->getBindingName(i);
		auto dims = mEngine->getBindingDimensions(i);
		//nv::DataType type = mEngine->getBindingDataType(i);
		auto isInput = mEngine->bindingIsInput(i);
		sout << "Binding " << i << ": " << name << "("
			 << (isInput ? "Input " : "Output ")
			 << Formatter::toString(dims) << ")\n";
	}
	Logger::instance.info(sout.str().c_str());

	return true;
}

void Net::bindResource(const std::string &name, Resource *res)
{
	mResources.addResource(name, res);
}

bool Net::dispose()
{
	mResources.clear();
	mEngine = nullptr;
	return true;
}

bool Net::infer(cudaStream_t stream, bool dumpInputOutput)
{
	CudaMapScope mapScope(mResources.graphicsResources);
	CHECK(mapScope.map());

	auto bindings = _getBindings();

	if (!mContext->enqueueV2(bindings.data(), stream, nullptr))
	{
		Logger::instance.error("Failed to enqueue inference");
		return false;
	}
	/*
	if (stream == nullptr) {
		auto inferStart = clock();
		if (!mContext->executeV2(bindings.data())) {
			Logger::instance.error("Failed to execute inference");
			return false;
		}
		auto inferEnd = clock();
		std::ostringstream sout;
		sout << "Infer takes " << INTERVAL(inferStart, inferEnd) << "ms" << std::endl;
		Logger::instance.info(sout.str());
	} else {
		if (!mContext->enqueueV2(bindings.data(), stream, nullptr)) {
			Logger::instance.error("Failed to enqueue inference");
			return false;
		}
	}
	*/

	if (dumpInputOutput)
	{
		if (stream != nullptr)
			CHECK(cudaStreamSynchronize(stream));
		_dumpInputOutput();
	}

	return true;
}

void Net::_deserialize(const std::string &path)
{
	std::ifstream fin(path, std::ios::in | std::ios::binary);
	if (!fin.is_open())
		return;

	std::streampos begin, end;
	begin = fin.tellg();
	fin.seekg(0, std::ios::end);
	end = fin.tellg();
	std::size_t size = end - begin;
	fin.seekg(0, std::ios::beg);
	char *engine_data = new char[size];
	fin.read(engine_data, size);
	fin.close();

	uptr<nv::IRuntime> runtime(nv::createInferRuntime(Logger::instance));
	mEngine = std::shared_ptr<nv::ICudaEngine>(
		runtime->deserializeCudaEngine(engine_data, size, nullptr),
		Destroy<nv::ICudaEngine>());
	delete[] engine_data;
	Logger::instance.info("Engine is deserialized");
}

std::vector<void *> Net::_getBindings()
{
	std::vector<void *> bindings(mEngine->getNbBindings());
	for (auto it = mResources.resources.begin();
		 it != mResources.resources.end(); ++it)
	{
		auto idx = mEngine->getBindingIndex(it->first.c_str());
		if (idx < 0)
			continue;
		bindings[idx] = it->second->getBuffer();
	}
	return bindings;
}

void Net::_dumpInputOutput()
{
	auto bindings = _getBindings();
	for (auto it = mResources.resources.begin();
		 it != mResources.resources.end(); ++it)
	{
		auto idx = mEngine->getBindingIndex(it->first.c_str());
		if (idx < 0)
			continue;
		if (mEngine->bindingIsInput(idx))
		{
			std::ostringstream sout;
			sout << "Input Buffer " << it->first << ": ";
			_dumpBuffer(sout, bindings[idx], idx);
			Logger::instance.info(sout.str().c_str());
		}
		else
		{
			std::ostringstream sout;
			sout << "Output Buffer " << it->first << ": ";
			_dumpBuffer(sout, bindings[idx], idx);
			Logger::instance.info(sout.str().c_str());
		}
	}
}

bool Net::_dumpBuffer(std::ostream &os, void *deviceBuf, int index)
{
	return _dumpBuffer(os, deviceBuf, mEngine->getBindingDimensions(index),
					   mEngine->getBindingDataType(index));
}

bool Net::_dumpBuffer(std::ostream &os, void *deviceBuf, nv::Dims bufDims, nv::DataType dataType)
{
	auto size = std::accumulate(bufDims.d, bufDims.d + bufDims.nbDims, 1,
								std::multiplies<int64_t>()) *
				getElementSize(dataType);
	char *hostBuf = new char[size];
	CHECK(cudaMemcpyAsync(hostBuf, deviceBuf, size, cudaMemcpyDeviceToHost));
	int mBatchSize = 0;
	size_t rowCount = static_cast<size_t>(bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);
	int leadDim = mBatchSize;
	int *trailDims = bufDims.d;
	int nbDims = bufDims.nbDims;

	// Fix explicit Dimension networks
	if (!leadDim && nbDims > 0)
	{
		leadDim = bufDims.d[0];
		++trailDims;
		--nbDims;
	}

	os << "[" << leadDim;
	for (int i = 0; i < nbDims; i++)
		os << ", " << trailDims[i];
	os << "]" << std::endl;
	switch (dataType)
	{
	case nv::DataType::kINT32:
		dumpHostBuffer<int32_t>(os, (int32_t*)hostBuf, size, rowCount);
		break;
	case nv::DataType::kFLOAT:
		dumpHostBuffer<float>(os, (float*)hostBuf, size, rowCount);
		break;
	case nv::DataType::kHALF:
		dumpHostBuffer<half_float::half>(os, (half_float::half*)hostBuf, size, rowCount);
		break;
	case nv::DataType::kINT8:
		assert(0 && "Int8 network-level input and output is not supported");
		break;
	case nv::DataType::kBOOL:
		assert(0 && "Bool network-level input and output are not supported");
		break;
	}
	return true;
}
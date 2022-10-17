#pragma once
#include "../utils/common.h"


class Net {
public:
    Net() : mEngine(nullptr) { }

    bool load(const std::string& path);

    void bindResource(const std::string& name, Resource* res);

    bool dispose();

    bool infer(cudaStream_t stream = nullptr, bool dumpInputOutput = false);

private:
    std::shared_ptr<nv::ICudaEngine> mEngine;
	std::shared_ptr<nv::IExecutionContext> mContext;
    Resources mResources;

    void _deserialize(const std::string& path);

	std::vector<void*> _getBindings();

	void _dumpInputOutput();

protected:
    bool _dumpBuffer(std::ostream& os, void* deviceBuf, int index);

    bool _dumpBuffer(std::ostream& os, void* deviceBuf, nv::Dims bufDims, nv::DataType dataType);

};

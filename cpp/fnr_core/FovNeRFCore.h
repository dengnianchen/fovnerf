#pragma once
#include "../utils/common.h"
#include "Net.h"

class FovNeRFCore {
public:
    Net *net;

    FovNeRFCore();

    virtual bool load(const std::string &netDir);
    virtual void bindResources(Resource *resEncoded, Resource *resRgbd);
    virtual bool infer();
    virtual void dispose();
};

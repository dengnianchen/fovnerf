#include "FovNeRFCore.h"
#include <time.h>

FovNeRFCore::FovNeRFCore() : net(nullptr) {}

bool FovNeRFCore::load(const std::string &netPath) {
    net = new Net();
    if (net->load(netPath))
        return true;
    dispose();
    return false;
}

void FovNeRFCore::bindResources(Resource *resEncoded, Resource *resRgbd) {
    net->bindResource("Encoded", resEncoded);
    net->bindResource("RGBD", resRgbd);
}

bool FovNeRFCore::infer() { return net->infer(); }

void FovNeRFCore::dispose() {
    if (net != nullptr) {
        net->dispose();
        delete net;
        net = nullptr;
    }
}

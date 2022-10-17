#include "Config.h"
#include <fstream>
#include <sstream>

Config::Config(std::string path)
{
    std::ifstream fin(path);
    if (!fin)
        throw std::runtime_error("Cannot read config file " + path);
    std::string line_content;
    while (std::getline(fin, line_content))
    {
        auto sep_index = line_content.find_first_of('=');
        _data.insert(std::make_pair(line_content.substr(0, sep_index), line_content.substr(sep_index + 1)));
    }
}

std::string Config::getString(std::string key)
{
    return _data[key];
}

int Config::getInt(std::string key)
{
    std::istringstream sin(_data[key]);
    int value;
    sin >> value;
    return value;
}

float Config::getFloat(std::string key)
{
    std::istringstream sin(_data[key]);
    float value;
    sin >> value;
    return value;
}

bool Config::getBool(std::string key)
{
    if (_data[key] == "True" || _data[key] == "true" || _data[key] == "1")
        return true;
    return false;
}
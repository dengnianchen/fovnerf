#pragma once
#include <string>
#include <map>

class Config
{
public:
    Config(std::string path);

    std::string getString(std::string key);
    int getInt(std::string key);
    float getFloat(std::string key);
    bool getBool(std::string key);

private:
    std::map<std::string, std::string> _data;
};
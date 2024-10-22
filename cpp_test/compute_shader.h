//
// Created by yuan on 10/20/24.
//

#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class compute_shader {
public:
    // the program ID
    unsigned int ID;

    // constructor reads and builds the shader
    compute_shader(const char *ComputePath);
    // use/activate the shader
    void use();
    // utility uniform functions
    void setBool(const std::string &name, bool value) const;
    void setInt(const std::string &name, int value) const;
    void setFloat(const std::string &name, float value) const;
};



#endif //COMPUTE_SHADER_H

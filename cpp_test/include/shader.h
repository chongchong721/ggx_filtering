//
// Created by yuan on 10/20/24.
//

//Shader class copied from LearnOpenGL.com

#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <GL/glew.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glm/glm.hpp>

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



class Shader
{
public:
    unsigned int ID;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char* vertexPath, const char* fragmentPath);

    // activate the shader
    // ------------------------------------------------------------------------
    void use();
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const;
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const;
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const;
    // ------------------------------------------------------------------------
    void setVec2(const std::string &name, const glm::vec2 &value) const;

    void setVec2(const std::string &name, float x, float y) const;

    // ------------------------------------------------------------------------
    void setVec3(const std::string &name, const glm::vec3 &value) const;

    void setVec3(const std::string &name, float x, float y, float z) const;

    // ------------------------------------------------------------------------
    void setVec4(const std::string &name, const glm::vec4 &value) const;

    void setVec4(const std::string &name, float x, float y, float z, float w) const;

    // ------------------------------------------------------------------------
    void setMat2(const std::string &name, const glm::mat2 &mat) const;

    // ------------------------------------------------------------------------
    void setMat3(const std::string &name, const glm::mat3 &mat) const;

    // ------------------------------------------------------------------------
    void setMat4(const std::string &name, const glm::mat4 &mat) const;


private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type);
};


#endif //COMPUTE_SHADER_H

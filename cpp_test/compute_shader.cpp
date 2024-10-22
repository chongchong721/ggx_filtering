//
// Created by yuan on 10/20/24.
//

#include "compute_shader.h"

compute_shader::compute_shader(const char *ComputePath) {
    // 1. retrieve the vertex/fragment source code from filePath
    std::string compute_code;
    std::ifstream comput_shader_file;
    // ensure ifstream objects can throw exceptions:
    comput_shader_file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        comput_shader_file.open(ComputePath);

        std::stringstream ShaderStream;
        // read file's buffer contents into streams
        ShaderStream << comput_shader_file.rdbuf();

        // close file handlers
        comput_shader_file.close();

        // convert stream into string
        compute_code  = ShaderStream.str();

    }
    catch(std::ifstream::failure e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }
    const char* compute_shader_code = compute_code.c_str();

    // 2. compile shaders
    unsigned int compute;
    int success;
    char infoLog[512];

    // vertex Shader
    compute = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(compute, 1, &compute_shader_code, NULL);
    glCompileShader(compute);
    // print compile errors if any
    glGetShaderiv(compute, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(compute, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n" << infoLog << std::endl;
    };


    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, compute);
    glLinkProgram(ID);
    // print linking errors if any
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if(!success)
    {
        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(compute);

}

void compute_shader::use() {
    glUseProgram(ID);
}


void compute_shader::setBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
}
void compute_shader::setInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}
void compute_shader::setFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}
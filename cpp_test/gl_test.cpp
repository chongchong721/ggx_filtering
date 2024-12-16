//
// Created by yuan on 10/9/24.
//
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

#include <random>
#include <map>

#include "shader.h"

struct SampleInfo {
    glm::vec3 direction;
    glm::float32 weight;
    glm::float32 level;
    glm::uint sample_idx;
};



int get_inbound_array_idx(glm::ivec2 uv_arr_idx, int face, int level) {
    /**
     * the original array, has order like
     * face 0->5
     * res 128->2
     */

    //compute the array elements count before this level
    int res;
    int offset = 0;
    for(int tmp = 0; tmp < level; tmp++) {
        res = 2 << (6 - tmp);
        offset += res * res * 6;
    }
    res = 2 << (6 - level);
    int i = uv_arr_idx.y;
    int j = uv_arr_idx.x;
    int face_n = res * res;
    int face_start = face * face_n;
    int face_offset = i * res + j;
    int final_idx = offset + face_start + face_offset;
    return final_idx;


}




auto make_map() {
    std::map<std::pair<int,int>,std::pair<int,int>> face_edge_to_face_edge;
    std::map<std::pair<int,int>,bool> face_edge_to_reverse;
    face_edge_to_face_edge[std::make_pair(0,0)] = std::make_pair(4,1);
    face_edge_to_face_edge[std::make_pair(0,1)] = std::make_pair(5,0);
    face_edge_to_face_edge[std::make_pair(0,2)] = std::make_pair(2,1);
    face_edge_to_face_edge[std::make_pair(0,3)] = std::make_pair(3,1);
    face_edge_to_face_edge[std::make_pair(1,0)] = std::make_pair(5,1);
    face_edge_to_face_edge[std::make_pair(1,1)] = std::make_pair(4,0);
    face_edge_to_face_edge[std::make_pair(1,2)] = std::make_pair(2,0);
    face_edge_to_face_edge[std::make_pair(1,3)] = std::make_pair(3,0);
    face_edge_to_face_edge[std::make_pair(2,0)] = std::make_pair(1,2);
    face_edge_to_face_edge[std::make_pair(2,1)] = std::make_pair(0,2);
    face_edge_to_face_edge[std::make_pair(2,2)] = std::make_pair(5,2);
    face_edge_to_face_edge[std::make_pair(2,3)] = std::make_pair(4,2);
    face_edge_to_face_edge[std::make_pair(3,0)] = std::make_pair(1,3);
    face_edge_to_face_edge[std::make_pair(3,1)] = std::make_pair(0,3);
    face_edge_to_face_edge[std::make_pair(3,2)] = std::make_pair(4,3);
    face_edge_to_face_edge[std::make_pair(3,3)] = std::make_pair(5,3);
    face_edge_to_face_edge[std::make_pair(4,0)] = std::make_pair(1,1);
    face_edge_to_face_edge[std::make_pair(4,1)] = std::make_pair(0,0);
    face_edge_to_face_edge[std::make_pair(4,2)] = std::make_pair(2,3);
    face_edge_to_face_edge[std::make_pair(4,3)] = std::make_pair(3,2);
    face_edge_to_face_edge[std::make_pair(5,0)] = std::make_pair(0,1);
    face_edge_to_face_edge[std::make_pair(5,1)] = std::make_pair(1,0);
    face_edge_to_face_edge[std::make_pair(5,2)] = std::make_pair(2,2);
    face_edge_to_face_edge[std::make_pair(5,3)] = std::make_pair(3,3);

    face_edge_to_reverse[std::make_pair(0,0)] = false;
    face_edge_to_reverse[std::make_pair(0,1)] = true;
    face_edge_to_reverse[std::make_pair(0,2)] = false;
    face_edge_to_reverse[std::make_pair(0,3)] = false;
    face_edge_to_reverse[std::make_pair(1,0)] = false;
    face_edge_to_reverse[std::make_pair(1,1)] = false;
    face_edge_to_reverse[std::make_pair(1,2)] = false;
    face_edge_to_reverse[std::make_pair(1,3)] = true;
    face_edge_to_reverse[std::make_pair(2,0)] = false;
    face_edge_to_reverse[std::make_pair(2,1)] = true;
    face_edge_to_reverse[std::make_pair(2,2)] = true;
    face_edge_to_reverse[std::make_pair(2,3)] = false;
    face_edge_to_reverse[std::make_pair(3,0)] = true;
    face_edge_to_reverse[std::make_pair(3,1)] = false;
    face_edge_to_reverse[std::make_pair(3,2)] = false;
    face_edge_to_reverse[std::make_pair(3,3)] = true;
    face_edge_to_reverse[std::make_pair(4,0)] = false;
    face_edge_to_reverse[std::make_pair(4,1)] = false;
    face_edge_to_reverse[std::make_pair(4,2)] = false;
    face_edge_to_reverse[std::make_pair(4,3)] = false;
    face_edge_to_reverse[std::make_pair(5,0)] = false;
    face_edge_to_reverse[std::make_pair(5,1)] = false;
    face_edge_to_reverse[std::make_pair(5,2)] = true;
    face_edge_to_reverse[std::make_pair(5,3)] = true;

    return std::make_pair(face_edge_to_face_edge,face_edge_to_reverse);
}





/**
 * There is a deterministic map on
 * how to add out of boundary weight to another face
 * we should precompute this mapping
 */
auto create_edge_texture_map() {
    //For left,right edge. Order is tom->bot(ij order)
    //edge in order of left-right-top-bot
    //then face 0->5
    //then res 128 -> 2

    // 128 * 4 * 6 + 64 * 4 * 6 + 32 * 4 * 6......

    auto map_pair = make_map();
    auto face_edge_to_face_edge = map_pair.first;
    auto face_edge_to_reverse = map_pair.second;


    int edge_texture_map_idx = 0;

    std::vector<int> edge_texture_map(24 * (128 + 64 + 32 + 16 + 8 + 4 + 2));


    for(int level = 0; level < 7; level++) {
        int res = 2 << (6 - level);
        for(int face_idx = 0; face_idx < 6; face_idx++) {
            for(int edge_idx = 0; edge_idx < 4; edge_idx++) {
                bool reverse = face_edge_to_reverse[std::make_pair(face_idx,edge_idx)];
                auto face_edge = face_edge_to_face_edge[std::make_pair(face_idx,edge_idx)];

                int bound_face_idx = face_edge.first;
                int bound_edge_idx = face_edge.second;

                bool u_fixed;
                int fixed_idx;
                if(bound_edge_idx == 0 || bound_edge_idx == 1) {
                    u_fixed = true;
                    fixed_idx = bound_edge_idx == 0 ? 0 : res - 1;
                }else {
                    u_fixed = false;
                    fixed_idx = bound_edge_idx == 2 ? 0 : res - 1;
                }

                for(int idx = 0; idx < res; idx++) {
                    glm::ivec2 uv_arr_idx;
                    if (u_fixed) {
                        uv_arr_idx.x = fixed_idx;
                        if(reverse) {
                            uv_arr_idx.y = (res-1) - idx;
                        }else {
                            uv_arr_idx.y = idx;
                        }
                    }else {
                        uv_arr_idx.y = fixed_idx;
                        if(reverse) {
                            uv_arr_idx.x = (res-1) - idx;
                        }else {
                            uv_arr_idx.x = idx;
                        }
                    }
                    edge_texture_map[edge_texture_map_idx] = get_inbound_array_idx(uv_arr_idx, bound_face_idx, level);
                    edge_texture_map_idx ++;

                }

            }
        }
    }

    return edge_texture_map;
}






void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void check_shader_success(GLuint shader) {
    int  success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if(!success)
    {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
}

void check_program_success(GLuint program) {
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
}


int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(800,600,"LearnOpenGL",NULL,NULL);
    if(window == NULL) {
        std::cout << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW." << std::endl;
        return -1;
    }

    //Write some compute shader
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> dist(0.0f, 6.0f);

        uint n_dir = 1;
        uint n_sample_per_dir = 1;
        GLuint ssbo_sample_info;
        glGenBuffers(1,&ssbo_sample_info);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_sample_info);
        std::vector<SampleInfo> samples(n_dir * n_sample_per_dir);
        for(uint i = 0; i < n_dir * n_sample_per_dir; i++) {
            samples[i].direction = glm::vec3(1.0,0.0,1.0);
            samples[i].weight = 1.0;
            samples[i].level = 1.0;
            samples[i].sample_idx = (uint)(i / n_sample_per_dir);
        }
        glBufferData(GL_SHADER_STORAGE_BUFFER,samples.size() * sizeof(SampleInfo),samples.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,ssbo_sample_info);


        int n_texture = n_dir * 6 * (128*128+64*64+32*32+16*16+8*8+4*4+2*2);
        std::vector<float> all_texture(n_texture,0.0);
        GLuint ssbo_texture;
        glGenBuffers(1,&ssbo_texture);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_texture);
        glBufferData(GL_SHADER_STORAGE_BUFFER,all_texture.size() * sizeof(float),all_texture.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,ssbo_texture);


        int n_edges = n_dir * 6 * 4 * (130+68+34+18+10+6+4);
        std::vector<float> all_edges(n_edges,0.0);
        GLuint ssbo_edges;
        glGenBuffers(1,&ssbo_edges);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_edges);
        glBufferData(GL_SHADER_STORAGE_BUFFER,all_edges.size() * sizeof(float),all_edges.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,ssbo_edges);


        auto edge_texture_map = create_edge_texture_map();
        GLuint ssbo_edge_tex_map;
        glGenBuffers(1,&ssbo_edge_tex_map);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_edge_tex_map);
        glBufferData(GL_SHADER_STORAGE_BUFFER,edge_texture_map.size() * sizeof(int),edge_texture_map.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,3,ssbo_edge_tex_map);


        std::vector<float> output_dbg(32,0.0);
        GLuint ssbo_dbg;
        glGenBuffers(1,&ssbo_dbg);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_dbg);
        glBufferData(GL_SHADER_STORAGE_BUFFER,output_dbg.size() * sizeof(float),output_dbg.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,ssbo_dbg);


        shader my_shader("/home/yuan/school/ggx_filtering/cpp_test/bilinear_sample_process.glsl");
        my_shader.use();
        glDispatchCompute(1,1,1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        std::cout << "Compute shader program complete." << std::endl;

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_dbg);
        float * ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,output_dbg.size() * sizeof(float),GL_MAP_READ_BIT);



        if(ptr) {
            std::vector<float> tmp(ptr,ptr+output_dbg.size());
            std::cout << tmp[0] << std::endl;
            std::cout << tmp[1] << std::endl;
            std::cout << tmp[2] << std::endl;
            std::cout << tmp[3] << std::endl;
            std::cout << tmp[4] << std::endl;
            std::cout << tmp[5] << std::endl;
            std::cout << tmp[6] << std::endl;
            std::cout << tmp[7] << std::endl;
            std::cout << tmp[8] << std::endl;
            std::cout << tmp[9] << std::endl;
            std::cout << tmp[10] << std::endl;
            std::cout << tmp[11] << std::endl;
            std::cout << tmp[12] << std::endl;
            std::cout << tmp[13] << std::endl;
            std::cout << tmp[14] << std::endl;
            std::cout << tmp[15] << std::endl;
            std::cout << tmp[16] << std::endl;
            std::cout << tmp[17] << std::endl;
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_texture);
        ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,all_texture.size() * sizeof(float),GL_MAP_READ_BIT);

        int tmp_offset = 128 * 128 * 6;

        if(ptr) {
            std::vector<float> tmp(ptr,ptr+all_texture.size());

            float sum = 0;
            for(int i = 0; i < 6 * 128 * 128; i++) {
                if(std::isnan(tmp[i])) {
                    std::cout << "NaN" << std::endl;
                }
                sum += tmp[i];
            }


            std::cout << tmp[tmp_offset+ 64*64*4 + 64*33-1] << std::endl;
            std::cout << tmp[tmp_offset+ 64*64*4 + 64*32-1] << std::endl;
            std::cout << tmp[tmp_offset+ 64*64*4 + 64*31-1] << std::endl;
            std::cout << tmp[tmp_offset+ 64*64*4 + 64*30-1] << std::endl;

            std::cout << tmp[tmp_offset+ 64*31] << std::endl;
            std::cout << tmp[tmp_offset+ 64*32] << std::endl;
            std::cout << tmp[tmp_offset+ 64*33] << std::endl;

            std::cout << tmp[63*128+63] << std::endl;
            std::cout << tmp[63*128+64] << std::endl;
            std::cout << tmp[64*128+62] << std::endl;
            std::cout << tmp[64*128+63] << std::endl;
            std::cout << tmp[64*128+64] << std::endl;
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_dbg);
        glBufferData(GL_SHADER_STORAGE_BUFFER,output_dbg.size() * sizeof(float),output_dbg.data(),GL_DYNAMIC_COPY);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,4,ssbo_dbg);


        shader push_back_shader("/home/yuan/school/ggx_filtering/cpp_test/push_back.glsl");
        push_back_shader.use();

        int level = 1;
        int res = 2 << ( 6 - level );

        push_back_shader.setInt("current_level",level);

        int x_size = res * res / 64;
        x_size = x_size == 0 ? 1 : x_size;

        glDispatchCompute(x_size,6,n_dir);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);


        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_dbg);
        ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,output_dbg.size() * sizeof(float),GL_MAP_READ_BIT);

        if(ptr) {
            std::vector<float> tmp(ptr,ptr+output_dbg.size());


            std::cout << tmp[0] << std::endl;
            std::cout << tmp[1] << std::endl;
            std::cout << tmp[2] << std::endl;
            std::cout << tmp[3] << std::endl;
            std::cout << tmp[4] << std::endl;
            std::cout << tmp[5] << std::endl;
            std::cout << tmp[6] << std::endl;
            std::cout << tmp[7] << std::endl;
            std::cout << tmp[8] << std::endl;
            std::cout << tmp[9] << std::endl;
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo_texture);
        ptr = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,0,all_texture.size() * sizeof(float),GL_MAP_READ_BIT);
        auto error_code = glGetError();
        if(error_code != GL_NO_ERROR) {
            std::cout << error_code << std::endl;
        }

        if(ptr) {
            //compute sum?


            std::vector<float> tmp(ptr,ptr+all_texture.size());

            float sum = 0;
            for(int i = 0; i < 6 * 128 * 128; i++) {
                if(std::isnan(tmp[i])) {
                    std::cout << "NaN" << std::endl;
                }
                sum += tmp[i];
            }
            std::cout << sum << std::endl;

            std::cout << tmp[128*128*4 + 128*65-1] << std::endl;
            std::cout << tmp[128*128*4 + 128*64-1] << std::endl;
            std::cout << tmp[128*128*4 + 128*63-1] << std::endl;
            std::cout << tmp[128*128*4 + 128*62-1] << std::endl;

            std::cout << tmp[128*62] << std::endl;
            std::cout << tmp[128*63] << std::endl;
            std::cout << tmp[128*64] << std::endl;

            std::cout << tmp[63*128+63] << std::endl;
            std::cout << tmp[63*128+64] << std::endl;
            std::cout << tmp[64*128+62] << std::endl;
            std::cout << tmp[64*128+63] << std::endl;
            std::cout << tmp[64*128+64] << std::endl;
        }

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }






    glViewport(0, 0, 800, 600);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);


    check_shader_success(vertexShader);


    const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\0";
    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    check_shader_success(fragmentShader);



    GLuint shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    check_program_success(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);





    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // unbind?
    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);


    while(!glfwWindowShouldClose(window)) {
        processInput(window);


        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glfwPollEvents();
        glfwSwapBuffers(window);

    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
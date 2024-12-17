//
// Created by yuan on 12/16/24.
// A lot of code copied from LearnOpenGL
//
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

#include <random>
#include <map>

#include "shader.h"
#include "camera.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Cubesphere.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

std::string program_directory = "/home/yuan/school/ggx_filtering/";

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;

// Function to load cubemap textures
unsigned int loadHDRCubemap(std::vector<std::string> faces)
{
    std::string path_prefix = "/home/yuan/school/ggx_filtering/cubemap_face/";

    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrComponents;
    for(unsigned int i = 0; i < faces.size(); i++)
    {
        float *data = stbi_loadf((path_prefix + faces[i]).c_str(), &width, &height, &nrComponents, 0);
        if (data)
        {
            GLenum format = (nrComponents == 3) ? GL_RGB : GL_RGBA;
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                0,
                GL_RGB16F, // Internal format for HDR
                width,
                height,
                0,
                format,
                GL_FLOAT, // Data type
                data
            );
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Failed to load HDR cubemap at path: " << faces[i] << std::endl;
            stbi_image_free(data);
        }
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    return textureID;
}

// Also in HDR format
unsigned int load2DHDRTexture(std::string filename) {
    std::string path_prefix = "/home/yuan/school/ggx_filtering/second_sum/";
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    int width, height, nrComponents;
    float *data = stbi_loadf((path_prefix + filename).c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format = (nrComponents == 3) ? GL_RGB : GL_RGBA;
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB16F, // Internal format for HDR
            width,
            height,
            0,
            format,
            GL_FLOAT, // Data type
            data
        );
        stbi_image_free(data);
    }
    else
    {
        std::cout << "Failed to load HDR cubemap at path: " << filename << std::endl;
        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
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




    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    //glViewport(0, 0, 800, 600);

    glEnable(GL_DEPTH_TEST);


    std::vector<std::string> faces
    {
        "pos_x.hdr",   // GL_TEXTURE_CUBE_MAP_POSITIVE_X
        "neg_x.hdr",    // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
        "pos_y.hdr",     // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
        "neg_y.hdr",  // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
        "pos_z.hdr",   // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        "neg_z.hdr"     // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
    };


    std::vector<std::string> filtered_faces{
        "filtered_pos_x.hdr",   // GL_TEXTURE_CUBE_MAP_POSITIVE_X
        "filtered_neg_x.hdr",    // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
        "filtered_pos_y.hdr",     // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
        "filtered_neg_y.hdr",  // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
        "filtered_pos_z.hdr",   // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
        "filtered_neg_z.hdr"     // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
    };


    float skyboxVertices[] = {
        // positions
        -1.0f, -1.0f,  1.0f, //face 4(pos z)
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,


        -1.0f,  1.0f, -1.0f, //face 5(neg z)
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f, //face 1(neg x)
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f, //face 0(pos x)
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f,  1.0f, -1.0f, //face 2(pos y)
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f, //face 3(neg y)
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };


    // skybox VAO
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // Unbind the VAO to prevent accidental modification
    glBindVertexArray(0);

    Shader skybox_shader( (program_directory + "cpp_test/shader_program/env_map_vertex.glsl").c_str(),
        (program_directory + "cpp_test/shader_program/env_map_fragment.glsl").c_str());
    Shader merl_sphere_shader((program_directory + "cpp_test/shader_program/sphere_vertex.glsl").c_str(),
        (program_directory + "cpp_test/shader_program/sphere_fragment.glsl").c_str());

    //Load skybox(environment map
    unsigned int cubemap_texture = loadHDRCubemap(faces);

    //Load second sum texture and second sum texture
    unsigned int filtered_cubemap_texture = loadHDRCubemap(filtered_faces);
    unsigned int second_sum_texture = load2DHDRTexture("second_sum.hdr");


    //Bind skybox to Texture unit 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture);

    //bind filtered cubemap(first sum) to unit 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, filtered_cubemap_texture);

    //bind second sum to unit2
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, second_sum_texture);



    skybox_shader.use();
    skybox_shader.setInt("skybox", 0);

    merl_sphere_shader.use();
    merl_sphere_shader.setInt("skybox",0);
    merl_sphere_shader.setInt("filteredMap",1);
    merl_sphere_shader.setInt("secondTermTexture",2);



    //setup a sphere
    //Code from https://www.songho.ca/opengl/gl_sphere.html#example_icosphere

    Cubesphere sphere = Cubesphere(1.0,4);
    unsigned int sphereVAO,sphereVerticesBO,sphereIndicesBO;


    // VAO
    glGenVertexArrays(1, &sphereVAO);
    glBindVertexArray(sphereVAO);


    // Vertices buffer
    glGenBuffers(1, &sphereVerticesBO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVerticesBO);           // for vertex data
    glBufferData(GL_ARRAY_BUFFER,                   // target
                 sphere.getInterleavedVertexSize(), // data size, # of bytes
                 sphere.getInterleavedVertices(),   // ptr to vertex data
                 GL_STATIC_DRAW);


    // Indices buffer
    glGenBuffers(1, &sphereIndicesBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereIndicesBO);   // for index data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,           // target
                 sphere.getIndexSize(),             // data size, # of bytes
                 sphere.getIndices(),               // ptr to index data
                 GL_STATIC_DRAW);                   // usage

    //set attributes 0 for vertices 1 for normals for now

    glEnableVertexAttribArray(0); //vertices
    glEnableVertexAttribArray(1); //normals
    glEnableVertexAttribArray(2); //uv coordinates(maybe not used)

    int stride = sphere.getInterleavedStride();     // should be 32 bytes
    glVertexAttribPointer(0, 3, GL_FLOAT, false, stride, (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, stride, (void*)(sizeof(float)*3));
    glVertexAttribPointer(2,  2, GL_FLOAT, false, stride, (void*)(sizeof(float)*6));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    while(!glfwWindowShouldClose(window)) {

        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);


        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDepthFunc(GL_LEQUAL);// Change depth function so skybox is rendered behind other objects
        skybox_shader.use();

        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        skybox_shader.setMat4("projection", projection);

        glm::mat4 view = glm::mat4(glm::mat3(camera.GetViewMatrix())); //remove 4th column(translation) skybox should not be translated
        //glm::mat4 view = camera.GetViewMatrix(); //remove 4th column(translation) skybox should not be translated
        skybox_shader.setMat4("view", view);

        glBindVertexArray(skyboxVAO);
        // glActiveTexture(GL_TEXTURE0);
        // glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glBindVertexArray(0);
        glDepthFunc(GL_LESS);

        glfwSwapBuffers(window);
        glfwPollEvents();


    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);

    glfwTerminate();
    return 0;
}





// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
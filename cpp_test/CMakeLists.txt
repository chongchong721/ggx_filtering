cmake_minimum_required(VERSION 3.10)
project(TEST VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)

#opencv setting
set(CMAKE_PREFIX_PATH "/home/yuan/software/opencv_source/build" ${CMAKE_PREFIX_PATH})
find_package( OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(test test.cpp
        GGX.cpp
        include/GGX.h
        include/util.h)
target_include_directories(test PUBLIC include/)
target_link_libraries(test ${OpenCV_LIBS})
add_executable(opencv_test opencv_test.cpp)
target_link_libraries( opencv_test ${OpenCV_LIBS} )


#OpenGL cmake setting
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif ()


set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# For include directory, can use target_include_directories alternatively
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
if (GLFW_FOUND)
    message(STATUS "GLFW found: ${GLFW_INCLUDE_DIRS}")
    #include_directories(${GLFW_INCLUDE_DIRS})
    #link_directories(${GLFW_LIBRARY_DIRS})
    add_definitions(${GLFW_CFLAGS_OTHER})
else()
    message(FATAL_ERROR "GLFW not found")
endif()

pkg_search_module(XI REQUIRED xi)
pkg_search_module(XRANDR REQUIRED xrandr)

if(Xi_FOUND)
    message(STATUS "XI found: ${XI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "XI not found")
endif ()

if(Xrandr_FOUND)
    message(STATUS "Xrandr found: ${XRANDR_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Xrandr not found")
endif ()


find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)



find_package(X11 REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_IUCLUDE_DIRS} ${GLFW_INCLUDE_DIRS} ${X11_INCLUDE_DIRS} ${XRANDR_INCLUDE_DIRS} ${XI_INCLUDE_DIRS})


add_executable(gl_test gl_test.cpp
        shader.cpp
        include/shader.h)
target_link_libraries(gl_test OpenGL::GL)
target_link_libraries(gl_test ${GLFW_LIBRARIES})
target_link_libraries(gl_test ${GLEW_LIBRARIES})
target_link_libraries(gl_test ${X11_LIBRARIES})
target_link_libraries(gl_test ${XRANDR_LIBRARIES})
target_link_libraries(gl_test ${XI_LIBRARIES})
target_link_libraries(gl_test Threads::Threads)
target_link_libraries(gl_test ${CMAKE_DL_LIBS})



add_executable(render_test
        render_test.cpp shader.cpp Cubesphere.cpp include/shader.h
        include/camera.h include/stb_image.h include/Cubesphere.h)

target_include_directories(render_test PUBLIC include/)

target_link_libraries(render_test OpenGL::GL)
target_link_libraries(render_test ${GLFW_LIBRARIES})
target_link_libraries(render_test ${GLEW_LIBRARIES})
target_link_libraries(render_test ${X11_LIBRARIES})
target_link_libraries(render_test ${XRANDR_LIBRARIES})
target_link_libraries(render_test ${XI_LIBRARIES})
target_link_libraries(render_test Threads::Threads)
target_link_libraries(render_test ${CMAKE_DL_LIBS})








add_compile_definitions(DJ_BRDF_IMPLEMENTATION)
add_executable(dj_brdf_test powit_test/dj_brdf.h
        powit_test/powit.cpp
        powit_test/MerlBRDF.h)
target_include_directories(dj_brdf_test PUBLIC ../extern/pybind11/include)
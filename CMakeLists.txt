# Set minimum required CMake version.
cmake_minimum_required(VERSION 3.7)
project(machinelearning)

# Set package name and version.
set(CPACK_PACKAGE_NAME "machinelearning")
set(CPACK_PACKAGE_VERSION "1.0.0")

# Use GCC to compile for Linux.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_COMPILER gcc)
set(CMAKE_FIND_ROOT_PATH /usr/bin/gcc)

# Register all source files.
file(GLOB_RECURSE sources src/*.c src/*.h)

# Add executable from all source files.
add_executable(machinelearning ${sources})

# Provide all warning output on compile.
target_compile_options(machinelearning PUBLIC -Wall -Wno-unknown-pragmas)

target_link_libraries(machinelearning m)

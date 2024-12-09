cmake_minimum_required(VERSION 3.29)

project(uint-div LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Od")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    add_compile_options(/W4)
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    add_compile_options(-Wall -Wextra -MMD)
endif()

add_subdirectory(cpp)

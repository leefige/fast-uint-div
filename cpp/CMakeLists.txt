file(GLOB_RECURSE CPP_SRC *.cpp)
add_executable(uint-div-cpp ${CPP_SRC})
target_include_directories(uint-div-cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

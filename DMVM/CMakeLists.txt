cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

project(DMVM LANGUAGES C)

option(ENABLE_CHECK "Enable Check CODE" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/executable)

find_package(MPI REQUIRED)

add_subdirectory(lib)
add_subdirectory(impl)



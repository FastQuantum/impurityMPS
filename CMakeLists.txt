cmake_minimum_required(VERSION 3.9)

project(fbr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external/external.cmake)

add_library(${PROJECT_NAME} INTERFACE)

file(GLOB_RECURSE _fbr_headers CONFIGURE_DEPENDS include/fbr/*.h)
target_sources(${PROJECT_NAME} INTERFACE
    "$<BUILD_INTERFACE:${_fbr_headers}>")
add_custom_target(${PROJECT_NAME}_lib SOURCES ${_fbr_headers})

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${MKL_INCLUDE_DIRS}
)

if(MKL_FOUND)
    target_link_libraries(${PROJECT_NAME} INTERFACE "-Wl,--start-group" $ENV{MKL_SEQUENTIAL_LAYER_LIBRARY} "-Wl,--end-group -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)
else()
    target_link_libraries(${PROJECT_NAME} INTERFACE lapacke tdvp itensor OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} INTERFACE armadillo)

##------------ examples ----------

add_subdirectory(example)

#----------- tests -----------------

# enable_testing() must run in the top-level list file so CMake writes the root
# CTestTestfile.cmake that recurses into test/; without it `ctest` in the build
# dir reports "No tests were found" even though test/ registers Catch2 tests.
enable_testing()

# Reference dynamics tests compare against snapshots up to t=20, which is slow.
# By default they stop at t=5; configure with -DFBR_ENABLE_LONG_TEST=ON to run
# the full t=20 sweep.
option(FBR_ENABLE_LONG_TEST "Run reference dynamics tests up to t=20 (default: t=5)" OFF)

add_subdirectory(test)

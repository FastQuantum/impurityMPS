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

add_subdirectory(test)

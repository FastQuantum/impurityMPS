cmake_minimum_required(VERSION 3.9)

project(impurityMPS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external/external.cmake)

add_library(${PROJECT_NAME} STATIC
    include/impurityMPS/fermionic.h
    include/impurityMPS/irlm.h
    include/impurityMPS/it_dmrg.h
    include/impurityMPS/givens_rotation.h
    include/impurityMPS/irlm_gs.h
    empty.cpp
    README.md
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${MKL_INCLUDE_DIRS}
)


target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,--start-group" $ENV{MKL_SEQUENTIAL_LAYER_LIBRARY} "-Wl,--end-group -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)
#target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,--start-group" $ENV{MKLROOT}/lib/intel64/libmkl_rt.so "-Wl,--end-group -Wl,--no-as-needed -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)

target_link_libraries(${PROJECT_NAME} PUBLIC armadillo)

##------------ examples ----------

add_subdirectory(example)

#----------- tests -----------------

add_subdirectory(test)

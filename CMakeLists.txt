cmake_minimum_required(VERSION 3.9)

project(fbr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external/external.cmake)

add_library(${PROJECT_NAME} STATIC
    include/fbr/fermionic.h
    include/fbr/givens_rotation.h
    include/fbr/fb_mps.h
    include/fbr/fbr_dyn.h
    empty.cpp
    README.md
    include/fbr/fbr_param.h
    include/fbr/fb_mps_spin.h
    include/fbr/fbr_gs_spin.h
    include/fbr/fbr_param_spin.h
    include/fbr/fbr_dyn_spin.h
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${MKL_INCLUDE_DIRS}
)

if(MKL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,--start-group" $ENV{MKL_SEQUENTIAL_LAYER_LIBRARY} "-Wl,--end-group -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)
    #target_link_libraries(${PROJECT_NAME} PUBLIC "-Wl,--start-group" $ENV{MKLROOT}/lib/intel64/libmkl_rt.so "-Wl,--end-group -Wl,--no-as-needed -lpthread -lm -ldl" tdvp itensor OpenMP::OpenMP_CXX)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC lapacke tdvp itensor OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC armadillo)

##------------ examples ----------

add_subdirectory(example)

#----------- tests -----------------

add_subdirectory(test)

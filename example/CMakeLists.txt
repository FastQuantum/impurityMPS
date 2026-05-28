# file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")

list(APPEND test_SRC
    fbr_gs_irlm.cpp
    fbr_dyn_irlm.cpp
    fbr_gs_siam.cpp
    fbr_dyn_siam.cpp
    fbr_dyn_ns_siam.cpp
    fbr_dyn_siam_center.cpp
    fbr_dyn_ns__man_siam.cpp
    fbr_dyn_siam_block.cpp
)

option(FBR_EXAMPLE_REF "Build reference star/chain examples" OFF)
if(FBR_EXAMPLE_REF)
    add_subdirectory(ref)
endif()

foreach (test ${test_SRC})
    get_filename_component (TName ${test} NAME_WE)

    add_executable(${TName} ${TName}.cpp)
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()


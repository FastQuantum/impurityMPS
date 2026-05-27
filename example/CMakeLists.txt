# file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")

list(APPEND test_SRC
    fbr_gs_siam.cpp
    fbr_dyn_siam.cpp
    fbr_dyn_ns_siam.cpp
    fbr_dyn_siam_center.cpp
    fbr_dyn_ns__man_siam.cpp
    star_gs_siam.cpp
    star_dyn_siam.cpp
    star_dyn_siam_center.cpp
    star_dyn_siam_spin.cpp
    chain_dyn_siam_center.cpp
    star_dyn_siam_center_ip.cpp
    fbr_dyn_siam_block.cpp
)

foreach (test ${test_SRC})
    get_filename_component (TName ${test} NAME_WE)

    add_executable(${TName} ${TName}.cpp)
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()


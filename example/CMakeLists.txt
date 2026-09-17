# file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")

list(APPEND test_SRC
    fbr_gs_irlm.cpp
    fbr_dyn_irlm.cpp
    fbr_green_irlm.cpp
    fbr_green_irlm_separate.cpp
    fbr_gs_siam_spin_sym.cpp
    fbr_dyn_siam_spin_sym.cpp
    fbr_dyn_siam.cpp
    fbr_dyn_siam_spin_sym_manual.cpp
    fbr_dyn_siam_manual.cpp
    fbr_dyn_siam_spin_block.cpp
)

foreach (test ${test_SRC})
    get_filename_component (TName ${test} NAME_WE)

    add_executable(${TName} ${TName}.cpp)
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()


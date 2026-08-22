list(APPEND ref_SRC
    chain_dyn_siam_center.cpp
    star_gs_siam.cpp
    star_dyn_siam.cpp
    star_dyn_siam_center.cpp
    star_dyn_siam_spin.cpp
    star_dyn_siam_center_ip.cpp
    star_dyn_siam_center_ipc.cpp
    star_dyn_tune.cpp
    fbr_dyn_tune.cpp
)

foreach (src ${ref_SRC})
    get_filename_component(TName ${src} NAME_WE)

    add_executable(${TName} ${src})
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()

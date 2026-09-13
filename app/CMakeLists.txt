# Numerical experiments. Each .cpp is its own executable and, like the examples,
# a single TU (the TDVP headers define addBasis non-inline).
file(GLOB app_SRC CONFIGURE_DEPENDS "*.cpp")

foreach (src ${app_SRC})
    get_filename_component(TName ${src} NAME_WE)
    add_executable(${TName} ${src})
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()

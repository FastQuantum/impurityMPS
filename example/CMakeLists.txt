file(GLOB test_SRC CONFIGURE_DEPENDS "*.cpp")

foreach (test ${test_SRC})
    get_filename_component (TName ${test} NAME_WE)

    add_executable(${TName} ${TName}.cpp)
    target_compile_features(${TName} PUBLIC cxx_std_17)
    target_link_libraries(${TName} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${TName} PRIVATE stdc++fs)
    target_link_libraries(${TName} PRIVATE nlohmann_json::nlohmann_json)
endforeach()

file(COPY ${CMAKE_SOURCE_DIR}/example/param.json
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

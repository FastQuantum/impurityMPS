add_executable(${PROJECT_NAME}_test
    test_main.cpp
    test_givens.cpp
    # test_itensor.cpp
    test_graph.cpp
    test_fb_mps.cpp
    test_impurity_param.cpp
)

target_include_directories(${PROJECT_NAME}_test INTERFACE Catch2::Catch2)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_test PRIVATE Catch2::Catch2)

if(FBR_EXAMPLE_REF)
    add_subdirectory(ref)
endif()

# Each FBR dynamics variant needs its own executable: the TDVP headers define
# non-inline functions, so two TUs including a dyn header in one link target
# would produce multiple-definition errors.
foreach(variant fbr block ns green)
    add_executable(${PROJECT_NAME}_test_${variant}
        test_main.cpp
        test_ref_${variant}.cpp
    )
    target_include_directories(${PROJECT_NAME}_test_${variant} INTERFACE Catch2::Catch2)
    target_link_libraries(${PROJECT_NAME}_test_${variant} PRIVATE ${PROJECT_NAME})
    target_link_libraries(${PROJECT_NAME}_test_${variant} PRIVATE Catch2::Catch2)
    if(FBR_ENABLE_LONG_TEST)
        target_compile_definitions(${PROJECT_NAME}_test_${variant} PRIVATE FBR_ENABLE_LONG_TEST)
    endif()
endforeach()

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib/)
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME}_test)
catch_discover_tests(${PROJECT_NAME}_test_fbr)
catch_discover_tests(${PROJECT_NAME}_test_block)
catch_discover_tests(${PROJECT_NAME}_test_ns)
catch_discover_tests(${PROJECT_NAME}_test_green)

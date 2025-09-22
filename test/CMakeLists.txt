add_executable(${PROJECT_NAME}_test 
    test_main.cpp
    test_givens.cpp
    test_itensor.cpp
)

target_include_directories(${PROJECT_NAME}_test INTERFACE Catch2::Catch2)
target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}_test PRIVATE Catch2::Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib/)
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME}_test)


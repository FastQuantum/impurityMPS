include(FetchContent)

# Keep downloaded dependency sources in a shared cache outside the build tree so
# they are not re-downloaded every time a build directory is deleted/recreated.
# This must be a cache variable: FetchContent otherwise installs its per-build
# default (<build>/_deps) before it sees this normal variable.
set(_fbr_fetchcontent_dir "$ENV{HOME}/.cache/${PROJECT_NAME}/fetchcontent")
if(NOT DEFINED FETCHCONTENT_BASE_DIR
   OR FETCHCONTENT_BASE_DIR STREQUAL "${CMAKE_BINARY_DIR}/_deps")
  set(FETCHCONTENT_BASE_DIR "${_fbr_fetchcontent_dir}" CACHE PATH
      "Shared FetchContent cache" FORCE)
else()
  set(FETCHCONTENT_BASE_DIR "${FETCHCONTENT_BASE_DIR}" CACHE PATH
      "Shared FetchContent cache")
endif()
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Do not update FetchContent dependencies automatically")

FetchContent_Declare(
  armadillo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG        11.4.x
)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.x
)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)

FetchContent_MakeAvailable(armadillo Catch2 json)


add_library(itensor STATIC IMPORTED) # or STATIC instead of SHARED
set_target_properties(itensor PROPERTIES
  IMPORTED_LOCATION "$ENV{HOME}/opt/ITensor/lib/libitensor.a"
  IMPORTED_LOCATION_RELEASE "$ENV{HOME}/opt/ITensor/lib/libitensor.a"
  IMPORTED_LOCATION_DEBUG "$ENV{HOME}/opt/ITensor/lib/libitensor-g.a"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/opt/ITensor"
)

add_library(tdvp INTERFACE IMPORTED) # or STATIC instead of SHARED
set_target_properties(tdvp PROPERTIES
 INTERFACE_INCLUDE_DIRECTORIES "$ENV{HOME}/opt/TDVP"
)


find_package(OpenMP REQUIRED)

include(external/FindMKL.cmake)

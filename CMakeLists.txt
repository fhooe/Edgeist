cmake_minimum_required(VERSION 3.21)
include(cmake/functions.cmake)

get_version_info(VERSION_STRING YEAR TODAY GIT_REVISION)
set(CMAKE_CXX_FLAGS_RELEASE "-O0")
set(PROJECT_NAME "Edgeist")
project(${PROJECT_NAME} VERSION ${VERSION_STRING} DESCRIPTION "The Edgeist Project.")

add_subdirectory(desktop_application)

# Author: Daniel Giritzer, Msc and Tobias Egger, Msc
set(FUNCTIONS_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Default build type: RelWithDebInfo" FORCE)
endif()

# make sure the output dir is the same, no matter which generator is used
if(NOT CMAKE_GENERATOR MATCHES "^Visual*.")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>)
endif()

# find all git submodules (to exclude them from doxygen)
set(DOXYGEN_EXCLUDE_DIR "")
execute_process(COMMAND bash ${FUNCTIONS_CMAKE_DIR}/../scripts/list_submodules.sh
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE GIT_SUBMODULES OUTPUT_STRIP_TRAILING_WHITESPACE)
if(DEFINED GIT_SUBMODULES)
    if(NOT "${GIT_SUBMODULES} " STREQUAL " ")
        string(REPLACE "\n" ";" GIT_SUBMODULES_LIST ${GIT_SUBMODULES})
        foreach(SUBMODULE_PATH ${GIT_SUBMODULES_LIST})
            set(DOXYGEN_EXCLUDE_DIR "${DOXYGEN_EXCLUDE_DIR}EXCLUDE+=${SUBMODULE_PATH}\n")
        endforeach()
    endif()
endif()

# ! enable_clang_tidy : Enables clang tidy static code analysis for the given target
#
# \arg:TARGET_NAME Name of the target
function(enable_clang_tidy TARGET_NAME)
    # clang-tidy static code analysis
    set(ENABLE_CLANG_TIDY FALSE CACHE BOOL "Enable static code analysis using clang tidy (default off).")
    if(ENABLE_CLANG_TIDY)
        find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
        set(CLANG_TIDY_CONFIG_FILE ${CMAKE_SOURCE_DIR}/.clang-tidy)
        set(CLANG_TIDY_COMMAND "${CLANG_TIDY_EXE}" "--config-file=${CLANG_TIDY_CONFIG_FILE}")
        set_target_properties(${TARGET_NAME} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        message(STATUS "Enable static code analysis for target '${TARGET_NAME}' using clang-tidy")
    endif()
endfunction()

# ! add_test_target : Helper function, adds unit tests to given targets. Is indirectly called by add_target.
#
# \arg:TEST_NAME Name of the unittest executable to be created.
#
# \arg:TARGET_TYPE Type of target (DYNAMIC_LIBRARY, STATIC_LIBRARY, INTERFACE).
#
# \arg:TARGET_NAME Name of the target to add unittests for.
#
# \arg:TEST_SRC Source files for this unit test (make sure to pass variables containing lists under quotes).
function(add_test_target TEST_NAME TARGET_TYPE TARGET_NAME TESTS_SRC)
    if(ENABLE_TESTS)

        if(${TARGET_TYPE} STREQUAL "STATIC_LIBRARY" OR ${TARGET_TYPE} STREQUAL "DYNAMIC_LIBRARY"
           OR ${TARGET_TYPE} STREQUAL "INTERFACE")
            find_package(GTest)
            if(NOT TARGET GTest::gtest_main)
                include(FetchContent)
                set(GTEST_LIBRARY_VERSION "1.13.0"
                    CACHE STRING "Version of the GTest library to use, if none is found on the system.")
                FetchContent_Declare(
                    googletest URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_LIBRARY_VERSION}.zip)
                FetchContent_GetProperties(googletest)
                if(NOT googletest_POPULATED)
                    message(
                        "GTest library was not found! Trying to download GTest library ${GTEST_LIBRARY_VERSION} instead!"
                    )
                    FetchContent_Populate(googletest)
                    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
                endif()
            endif()

            add_executable("${TEST_NAME}" ${TESTS_SRC})

            include(CheckIncludeFile)
            check_include_file("fff.h" HAVE_FFF_H)
            if(NOT HAVE_FFF_H AND NOT fff_POPULATED)
                include(FetchContent)
                set(FFF_LIBRARY_VERSION "1.1"
                    CACHE STRING "Version of the FFF library to use, if none is found on the system.")
                FetchContent_Declare(
                    fff URL https://github.com/meekrosoft/fff/releases/download/v${FFF_LIBRARY_VERSION}/fff.h
                    DOWNLOAD_NO_EXTRACT TRUE)
                FetchContent_GetProperties(fff)
                if(NOT ${fff_POPULATED})
                    message("FFF library was not found! Trying to download FFF library ${FFF_LIBRARY_VERSION} instead!")
                    FetchContent_Populate(fff)
                endif()
            endif()

            target_link_libraries("${TEST_NAME}" PRIVATE ${TARGET_NAME} GTest::gtest_main GTest::gtest GTest::gmock
                                                         GTest::gmock_main)
            target_link_libraries("${TEST_NAME}" PRIVATE "-lgcov")
            target_include_directories("${TEST_NAME}" PRIVATE "tests" "${fff_SOURCE_DIR}")
            target_compile_options("${TEST_NAME}" PRIVATE "--coverage")
            target_link_options("${TEST_NAME}" PRIVATE "--coverage")
            set_property(TARGET "${TEST_NAME}" PROPERTY CXX_STANDARD 17)
            if(NOT ${TARGET_TYPE} STREQUAL "INTERFACE")
                target_link_libraries("${TARGET_NAME}" PRIVATE "-lgcov")
                target_compile_options("${TARGET_NAME}" PRIVATE "--coverage")
                target_link_options("${TARGET_NAME}" PRIVATE "--coverage")
            endif()

            include(GoogleTest)
            gtest_discover_tests("${TEST_NAME}" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

            add_dependencies(ALL_TESTS ${TARGET_NAME})
        endif()
    endif()
endfunction()

# ! add_target : Add new build target with default options.
#
# Source files from "TARGET_DIR/src/**" are automatically added. Include dir "TARGET_DIR/include" is automatically added
# as PUBLIC include dir. Include dir "TARGET_DIR/private_include" is automatically added as PRIVATE include dir.
#
# \arg:TARGET_NAME Name of the target.
#
# \arg:TARGET_TYPE Type of target (DYNAMIC_LIBRARY, STATIC_LIBRARY, EXECUTABLE, INTERFACE).
function(add_target TARGET_NAME TARGET_TYPE)
    set(ENABLE_TESTS FALSE CACHE BOOL "Build unit tests (default off).")
    if(ENABLE_TESTS)
        set(ENABLE_MEMCHECK FALSE
            CACHE BOOL "Perform a memory check on the unit tests (may detect memory leaks, default off).")
        if(ENABLE_MEMCHECK)
            find_program(MEMORYCHECK_COMMAND valgrind)
            set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --error-exitcode=1")
        endif()
        include(CTest)
    endif()

    file(
        GLOB
        TARGET_SRC
        CONFIGURE_DEPENDS
        "src/*.cpp"
        "src/**/*.cpp"
        "src/**/**/*.cpp"
        "src/*.c"
        "src/**/*.c"
        "src/**/**/*.c")

    if(${TARGET_TYPE} STREQUAL "STATIC_LIBRARY")
        add_library(${TARGET_NAME} STATIC ${TARGET_SRC})
        target_include_directories(${TARGET_NAME} PUBLIC include)
        target_include_directories(${TARGET_NAME} PUBLIC generated)
        target_include_directories(${TARGET_NAME} PRIVATE private_include)
    elseif(${TARGET_TYPE} STREQUAL "EXECUTABLE")
        add_executable(${TARGET_NAME} ${TARGET_SRC})
        target_include_directories(${TARGET_NAME} PUBLIC include)
        target_include_directories(${TARGET_NAME} PUBLIC generated)
        target_include_directories(${TARGET_NAME} PRIVATE private_include)
    elseif(${TARGET_TYPE} STREQUAL "DYNAMIC_LIBRARY")
        add_library(${TARGET_NAME} SHARED ${TARGET_SRC})
        target_include_directories(${TARGET_NAME} PUBLIC include)
        target_include_directories(${TARGET_NAME} PUBLIC generated)
        target_include_directories(${TARGET_NAME} PRIVATE private_include)
    elseif(${TARGET_TYPE} STREQUAL "INTERFACE")
        add_library(${TARGET_NAME} INTERFACE)
        target_include_directories(${TARGET_NAME} INTERFACE include)
    else()
        message(FATAL_ERROR "Wrong target type passed!")
    endif()

    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

    enable_clang_tidy(${TARGET_NAME})

    if(NOT ${TARGET_TYPE} STREQUAL "INTERFACE")
        # Basic information hiding
        target_compile_options(${TARGET_NAME} PRIVATE -fvisibility=hidden)
        target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>)

        # Enforce strict compiler options (for higher code quality) target_compile_options(${TARGET_NAME} PRIVATE
        # -Weffc++)
        target_compile_options(${TARGET_NAME} PRIVATE -Werror)
        target_compile_options(${TARGET_NAME} PRIVATE -Wall)
        target_compile_options(${TARGET_NAME} PRIVATE -Wextra)

        # Enable some security features address space layout randomization
        target_compile_options(${TARGET_NAME} PRIVATE -fpie)
        target_link_options(${TARGET_NAME} PRIVATE -pie)

        target_compile_options(${TARGET_NAME} PRIVATE -fexceptions)

        # fixes lazy binding issues
        if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
            target_link_options(${TARGET_NAME} PRIVATE -Wl,-z,relro,-z,now)
        endif()
    endif()

    # Store debug symbols in separate file on Linux
    if(${TARGET_TYPE} STREQUAL "EXECUTABLE" OR ${TARGET_TYPE} STREQUAL "DYNAMIC_LIBRARY")
        if(${CMAKE_SYSTEM_NAME} STREQUAL Linux)
            add_custom_command(
                TARGET ${TARGET_NAME}
                POST_BUILD
                COMMAND ${CMAKE_OBJCOPY} --only-keep-debug $<TARGET_FILE:${TARGET_NAME}>
                        ${CMAKE_BINARY_DIR}/$<CONFIG>/${TARGET_NAME}.debug
                COMMAND ${CMAKE_STRIP} -g $<TARGET_FILE:${TARGET_NAME}>
                COMMAND ${CMAKE_OBJCOPY} --add-gnu-debuglink ${CMAKE_BINARY_DIR}/$<CONFIG>/${TARGET_NAME}.debug
                        $<TARGET_FILE:${TARGET_NAME}>)
        endif()
    endif()

    # add unittests
    if(ENABLE_TESTS)
        file(GLOB TESTS_SRC_LIST CONFIGURE_DEPENDS "tests/*.cpp" "tests/**/*.cpp" "tests/**/**/*.cpp")

        if(NOT TARGET ALL_TESTS)
            add_custom_target(ALL_TESTS)
        endif()

        if(ATOMIC_TESTS)
            foreach(TEST_FILE ${TESTS_SRC_LIST})
                set(TEST_FILE_NAME "none")
                cmake_path(GET TEST_FILE FILENAME TEST_FILE_NAME)
                cmake_path(REMOVE_EXTENSION TEST_FILE_NAME)
                if(NOT TEST_FILE_NAME STREQUAL "none")
                    add_test_target(TEST_${TEST_FILE_NAME} ${TARGET_TYPE} ${TARGET_NAME} ${TEST_FILE})
                endif()
            endforeach()
        else()
            add_test_target(TESTS_${TARGET_NAME} ${TARGET_TYPE} ${TARGET_NAME} "${TESTS_SRC_LIST}")
        endif()
    endif()

    if(NOT ${TARGET_TYPE} STREQUAL "INTERFACE")
        configure_file(${FUNCTIONS_CMAKE_DIR}/config.h.in config-${CMAKE_PROJECT_NAME}-${PROJECT_NAME}.h)
        target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_BINARY_DIR}")
    endif()

    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
endfunction()

# ! get_version_info : This function gets version info variables with the help of git.
#
# For this to work use semantic versioning in git tags with the following syntax: v1.0.0.0 or v1.0.0 (recommended)
#
# \arg:prj_version Variable to set a version string that can be passed on to a project() call.
#
# \arg:prj_year Variable to set the current year.
#
# \arg:prj_timestamp Variable to set the current timestamp.
#
# \arg:prj_revision Variable to set the git revision (including a dirty marking if there are uncommitted changes).
# cmake-lint: disable=R0912,R0915
function(get_version_info prj_version prj_year prj_timestamp prj_revision)

    # Internally used variables: YEAR - current year TIMESTAMP - current timestamp git_revision - if a git repo the
    # revision hash version_string - if a tag exists in format v0.0.0.0 a version string
    string(TIMESTAMP YEAR "%Y")
    string(TIMESTAMP TODAY)
    find_package(Git)
    if(GIT_FOUND)
        # check for git repo
        execute_process(COMMAND ${GIT_EXECUTABLE} status WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" RESULT_VARIABLE RC
                        OUTPUT_VARIABLE git_revision ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(RC MATCHES 0)
            # set git revision variable
            execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                            OUTPUT_VARIABLE git_revision ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

            execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                OUTPUT_VARIABLE LAST_TAG OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE RC)
            if(RC MATCHES 0)
                message(STATUS "Last GIT tag: ${LAST_TAG}")

                execute_process(
                    COMMAND ${GIT_EXECUTABLE} rev-list ${LAST_TAG}.. --count --ancestry-path
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE nr_commits
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                message(STATUS "Commits since last GIT tag: ${nr_commits}")

                string(FIND ${LAST_TAG} "v" pos)
                if(pos MATCHES 0)
                    string(REGEX REPLACE "^v(.*)" "\\1" version_string ${LAST_TAG})
                    if("${version_string}" MATCHES "^([0-9]+\.[0-9]+\.[0-9])$")
                        # tag has only 3 digits
                        set(stripped "${version_string}.")
                        set(tweak "0")
                    else()
                        string(REGEX REPLACE "^([0-9]+\\.[0-9]+\\.[0-9]+\\.)([0-9]+)" "\\1" stripped ${version_string})
                        string(REGEX REPLACE "^([0-9]+\\.[0-9]+\\.[0-9]+\\.)([0-9]+)" "\\2" tweak ${version_string})
                    endif()
                else()
                    message(STATUS "Tag format incorrect: ${LAST_TAG}")
                    set(stripped "0.0.0.")
                    set(tweak "0")
                endif()
            else()
                message(STATUS "No GIT tag found in the history of this branch.")
                set(nr_commits 0)
                set(stripped "0.0.0.")
                set(tweak "0")
            endif()

            # detect local changes
            execute_process(
                COMMAND ${GIT_EXECUTABLE} status --untracked-files=no --porcelain
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE GIT_DIRTY ERROR_QUIET
                                                                                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            string(COMPARE EQUAL "${GIT_DIRTY}" "" clean)
            if(clean AND nr_commits MATCHES 0)
                set(fix_nr ${tweak})
            else()
                math(EXPR fix_nr "10000 + 1000 * ${tweak} + ${nr_commits}")
                if(NOT clean)
                    set(git_revision "${git_revision} (DIRTY)")
                endif()
            endif()

            string(APPEND stripped ${fix_nr})
            message(STATUS "Set Version: ${stripped}")
            set(version_string "${stripped}")
            message(STATUS "GIT revision: ${git_revision}")
        else()
            message(STATUS "Not a GIT repository")
            set(git_revision "NULL")
            set(version_string "0.0.0.0")
        endif()
    else()
        message(STATUS "GIT not found")
        set(git_revision "NULL")
        set(version_string "0.0.0.0")
    endif()

    set(${prj_version} "${version_string}" PARENT_SCOPE)
    set(${prj_year} "${YEAR}" PARENT_SCOPE)
    set(${prj_timestamp} "${TODAY}" PARENT_SCOPE)
    set(${prj_revision} "${git_revision}" PARENT_SCOPE)
endfunction()

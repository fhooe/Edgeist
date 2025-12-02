cmake_minimum_required(VERSION 3.16) # 3.16+ for target_precompile_headers and reasonable generator expression support
project(BackProp LANGUAGES CXX)

# ---- Project-wide defaults ----
# You can change this if you want a different C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer multi-config generators (Visual Studio, Xcode) keep configs; single-config (Ninja) will map to chosen CMAKE_BUILD_TYPE.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type" FORCE)
endif()

# ---- Source list ----
set(BP_HEADERS
    AdaptiveAvgPool1d.h
    AdaptiveAvgPool2d.h
    BatchNorm1d.h
    BatchNorm2d.h
    Conf2d.h
    Dropout.h
    nmcf_ErrorTypes.h
    Flatten.h
    Layer.h
    Linear.h
    LossFunction.h
    MaxPool2d.h
    mnist_loader.h
    nmcm.h
    Modelenums.h
    Modelstructs.h
    Modeltypes.h
    object.h
    OptimizerDataTypes.h
    pch.h
    Relu.h
    Softmax.h
)

set(BP_SOURCES
    BackProp.cpp
    mnist_loader.cpp
    pch.cpp
)

# Optional data / asset files that were in the project
set(BP_ASSETS
    model.hex
    model_trainable.hex
)

# ---- Create target ----
add_executable(${PROJECT_NAME} ${BP_SOURCES} ${BP_HEADERS})

# Put headers in IDE view
target_sources(${PROJECT_NAME} PRIVATE ${BP_HEADERS})

# ---- Include directories ----
# The project used an extra IncludePath in Release|x64. We add a variable that preserves that behavior if present.
# If that path doesn't exist on your machine, remove or override BP_EXTRA_INCLUDE below (or pass -DBP_EXTRA_INCLUDE=...).
if(NOT DEFINED BP_EXTRA_INCLUDE)
  # default to the path that was in the vcxproj (kept as a configurable hint)
  set(BP_EXTRA_INCLUDE "C:/Users/p43110/Documents/work/git/24-01_jrz_ai-on-rld/Retraining_Framework/Code_Desktop_Application/BackProp/mnist_batch_output" CACHE PATH "Extra include path from original VS project")
endif()

# Always include project root so headers can be found.
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<$<BOOL:${BP_EXTRA_INCLUDE}>:${BP_EXTRA_INCLUDE}>
)

# ---- Precompiled header (MSVC and others) ----
# vcxproj had pch.h / pch.cpp; use CMake's precompile header helper if available.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.16")
  # pch.h is included in the project; we mark it for precompile
  target_precompile_headers(${PROJECT_NAME} PRIVATE "pch.h")
endif()

# ---- Compiler / Linker settings tuned to the original project ----
# Apply MSVC-specific flags similar to the vcxproj:
if (MSVC)
  # Conformance mode (equivalent to /permissive-)
  target_compile_options(${PROJECT_NAME} PRIVATE /permissive-)

  # Warning level 3
  target_compile_options(${PROJECT_NAME} PRIVATE /W3)

  # Enable ConformanceMode was true in the vcxproj; already /permissive- does much of that.

  # SDL checks (enable buffer security level) - vcxproj had SDLCheck true.
  target_compile_options(${PROJECT_NAME} PRIVATE /sdl)

  # Map per-configuration compile and link options using generator expressions:
  # Debug configs: disabled optimization, generate debug info, incremental linking
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/Od>                               # Optimization Disabled
    $<$<CONFIG:Release>:/O2>                             # Optimization: MaxSpeed
  )

  # Preprocessor definitions per config + platform specifics
  # Note: original project had WIN32 defined for Win32 configurations only.
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<STREQUAL:$<PLATFORM_ID>,Win32>>:WIN32;_DEBUG;_CONSOLE>
    $<$<AND:$<CONFIG:Release>,$<STREQUAL:$<PLATFORM_ID>,Win32>>:WIN32;NDEBUG;_CONSOLE>
    $<$<AND:$<CONFIG:Debug>,$<NOT:$<STREQUAL:$<PLATFORM_ID>,Win32>>>:_DEBUG;_CONSOLE>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<STREQUAL:$<PLATFORM_ID>,Win32>>>:NDEBUG;_CONSOLE>
  )

  # Linker options: enable debug info in both Debug and Release (vcxproj had GenerateDebugInformation true)
  target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/DEBUG>
    $<$<CONFIG:Release>:/DEBUG>
  )

  # Whole program optimization / Link Time Code Generation for Release (x64 had LTCG)
  target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:/GL>
  )
  target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:/LTCG>
  )

  # Subsystem: Console (default is usually console for add_executable on Windows)
  # (No explicit action required unless you want to force it:)
  # target_link_options(${PROJECT_NAME} PRIVATE /SUBSYSTEM:CONSOLE)

  # Link incremental: Debug true, Release false
  target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:/INCREMENTAL>
    $<$<CONFIG:Release>:/INCREMENTAL:NO>
  )
endif()

# ---- Extra: treat .hex files as runtime assets and copy to output dir on build ----
foreach(_asset IN LISTS BP_ASSETS)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_asset}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/${_asset}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
  endif()
endforeach()

# ---- IDE niceties ----
set_target_properties(${PROJECT_NAME} PROPERTIES
  # Put compiled binary in a "bin/<config>/" layout for single-config generators
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin/Debug"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin/Release"
)

# ---- Helpful messages ----
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Extra include (BP_EXTRA_INCLUDE): ${BP_EXTRA_INCLUDE}")

# End of CMakeLists.txt

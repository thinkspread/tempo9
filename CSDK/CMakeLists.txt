# Copyright (c) 2026 Jiejing Zhang.
#
# libtempo9_llm, built against the engine from a C/C++ SDK download:
#
#   cmake -S CSDK -B build -DCMAKE_PREFIX_PATH=<sdk directory>

cmake_minimum_required(VERSION 3.24)
project(Tempo9LLM LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

option(TEMPO9_BUILD_EXAMPLES "Build the C/C++ examples" ON)

# Shared with the Swift SDK; a release sets it.
set(TEMPO9_SDK_VERSION "1.3.1" CACHE STRING
    "Version shared by the C/C++ and Swift SDKs")
set(TEMPO9_LLM_PACKAGE_VERSION "${TEMPO9_SDK_VERSION}")

find_package(Tempo9EngineABI 1.4 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# These dependencies belong to libtempo9_llm, not to the engine. Populate their
# sources without adding their install rules or public CMake targets.
FetchContent_Declare(
  tempo9_llm_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR tempo9-no-upstream-project)
FetchContent_Declare(
  tempo9_llm_utf8proc
  GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
  GIT_TAG e5e799221b45bbb90f5fdc5c69b6b8dfbf017e78
  GIT_SHALLOW FALSE
  SOURCE_SUBDIR tempo9-no-upstream-project)
FetchContent_MakeAvailable(tempo9_llm_json tempo9_llm_utf8proc)

add_library(tempo9_llm_utf8proc STATIC
            ${tempo9_llm_utf8proc_SOURCE_DIR}/utf8proc.c)
set_target_properties(tempo9_llm_utf8proc PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      C_VISIBILITY_PRESET hidden)
target_compile_definitions(tempo9_llm_utf8proc PUBLIC UTF8PROC_STATIC)
target_include_directories(tempo9_llm_utf8proc
                           PUBLIC ${tempo9_llm_utf8proc_SOURCE_DIR})

# llama.cpp's Jinja engine, vendored unmodified; see its README.
set(tempo9_llm_jinja_dir
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-jinja/common)
add_library(tempo9_llm_jinja STATIC
            ${tempo9_llm_jinja_dir}/json.cpp
            ${tempo9_llm_jinja_dir}/unicode.cpp
            ${tempo9_llm_jinja_dir}/jinja/caps.cpp
            ${tempo9_llm_jinja_dir}/jinja/lexer.cpp
            ${tempo9_llm_jinja_dir}/jinja/parser.cpp
            ${tempo9_llm_jinja_dir}/jinja/runtime.cpp
            ${tempo9_llm_jinja_dir}/jinja/string.cpp
            ${tempo9_llm_jinja_dir}/jinja/value.cpp)
set_target_properties(tempo9_llm_jinja PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)
target_compile_features(tempo9_llm_jinja PUBLIC cxx_std_17)
target_include_directories(tempo9_llm_jinja
                           PUBLIC ${tempo9_llm_jinja_dir}
                           PRIVATE ${tempo9_llm_json_SOURCE_DIR}/include)

add_library(tempo9_llm SHARED
            src/llm.cpp
            src/metadata.cpp
            src/template_renderer.cpp
            src/tokenizer.cpp)
add_library(Tempo9::llm ALIAS tempo9_llm)
set_target_properties(tempo9_llm PROPERTIES
                      EXPORT_NAME llm
                      OUTPUT_NAME tempo9_llm
                      VERSION ${TEMPO9_LLM_PACKAGE_VERSION}
                      SOVERSION 1
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON
                      BUILD_RPATH "@loader_path"
                      INSTALL_RPATH "@loader_path")
target_compile_features(tempo9_llm PRIVATE cxx_std_17)
target_compile_definitions(tempo9_llm PRIVATE TEMPO9_LLM_BUILDING_LIBRARY)
target_include_directories(
  tempo9_llm
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${tempo9_llm_json_SOURCE_DIR}/include)
target_link_libraries(tempo9_llm
                      PUBLIC Tempo9::engine_abi_shared
                      PRIVATE tempo9_llm_jinja tempo9_llm_utf8proc)
target_link_options(
  tempo9_llm PRIVATE
  "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/tempo9_llm.exports")

add_library(tempo9_llm_cpp INTERFACE)
add_library(Tempo9::llm_cpp ALIAS tempo9_llm_cpp)
set_target_properties(tempo9_llm_cpp PROPERTIES EXPORT_NAME llm_cpp)
target_compile_features(tempo9_llm_cpp INTERFACE cxx_std_17)
target_include_directories(
  tempo9_llm_cpp
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(tempo9_llm_cpp INTERFACE tempo9_llm Threads::Threads)

if(TEMPO9_BUILD_EXAMPLES)
  add_executable(tempo9-llm-encode examples/llm_encode.c)
  add_executable(tempo9-llm-chat-c examples/llm_chat.c)
  target_compile_features(tempo9-llm-encode PRIVATE c_std_11)
  target_compile_features(tempo9-llm-chat-c PRIVATE c_std_11)
  target_link_libraries(tempo9-llm-encode PRIVATE Tempo9::llm)
  target_link_libraries(tempo9-llm-chat-c PRIVATE Tempo9::llm)
  foreach(example IN ITEMS generate chat async_load multimodal audio)
    string(REPLACE "_" "-" name "tempo9-llm-${example}")
    add_executable(${name} examples/llm_${example}.cpp)
    target_link_libraries(${name} PRIVATE Tempo9::llm_cpp)
  endforeach()
endif()

set(tempo9_llm_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Tempo9LLM")
set(tempo9_llm_runtime_manifest
    "${CMAKE_CURRENT_BINARY_DIR}/Tempo9LLMRuntimeManifest.json")
configure_file(cmake/Tempo9LLMRuntimeManifest.json.in
               ${tempo9_llm_runtime_manifest} @ONLY)
configure_package_config_file(
  cmake/Tempo9LLMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/Tempo9LLMConfig.cmake
  INSTALL_DESTINATION ${tempo9_llm_cmake_dir})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/Tempo9LLMConfigVersion.cmake
  VERSION ${TEMPO9_LLM_PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion)
install(TARGETS tempo9_llm tempo9_llm_cpp
        EXPORT Tempo9LLMTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/tempo9/llm.h include/tempo9/llm.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tempo9)
install(EXPORT Tempo9LLMTargets
        FILE Tempo9LLMTargets.cmake
        NAMESPACE Tempo9::
        DESTINATION ${tempo9_llm_cmake_dir})
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/Tempo9LLMConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/Tempo9LLMConfigVersion.cmake
        DESTINATION ${tempo9_llm_cmake_dir})
install(FILES ${tempo9_llm_runtime_manifest}
        DESTINATION ${CMAKE_INSTALL_DATADIR}/tempo9)

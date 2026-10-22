# Built against the llama-prism submodule's own build (scripts/build_llama.sh):
#   cmake -S src/eval_service/llama_states -B build/llama_states -DLLAMA_DIR=llama-prism
cmake_minimum_required(VERSION 3.18)
project(llama_states CXX)
set(CMAKE_CXX_STANDARD 17)
set(LLAMA_DIR "" CACHE PATH "the llama-prism checkout, built in its build/")
add_executable(llama-states states.cpp)
target_include_directories(llama-states PRIVATE
    ${LLAMA_DIR}/include ${LLAMA_DIR}/ggml/include ${LLAMA_DIR}/vendor)
target_link_directories(llama-states PRIVATE ${LLAMA_DIR}/build/bin)
target_link_libraries(llama-states PRIVATE llama ggml ggml-base)
set_target_properties(llama-states PROPERTIES BUILD_RPATH ${LLAMA_DIR}/build/bin)

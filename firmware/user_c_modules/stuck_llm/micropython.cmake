# MicroPython user C module: stuck_llm
#
# Wraps llm/runtime/stuck_llm.c for the badge firmware. Add to the board's
# mpconfigboard.cmake (or pass on the make line) as
#
#   list(APPEND USER_C_MODULES <path-to-repo>/firmware/user_c_modules/stuck_llm/micropython.cmake)
#
# The inference core is shared with the host harness in llm/runtime/host, so
# anything tested there is what runs on the chip.

add_library(usermod_stuck_llm INTERFACE)

target_sources(usermod_stuck_llm INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/mod_stuck_llm.c
    ${CMAKE_CURRENT_LIST_DIR}/../../../llm/runtime/stuck_llm.c
)

target_include_directories(usermod_stuck_llm INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../../../llm/runtime
)

# The int8 dot product is the hot loop; let the compiler unroll it.
target_compile_options(usermod_stuck_llm INTERFACE -O3 -funroll-loops)

target_link_libraries(usermod INTERFACE usermod_stuck_llm)

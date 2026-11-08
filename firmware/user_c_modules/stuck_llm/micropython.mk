# Makefile-style user C module (unix / other make-based ports), the twin of
# micropython.cmake. Build the unix port against it with
#   make -C ports/unix USER_C_MODULES=<dir containing only this module>
# to exercise the Python-facing API on a PC.
STUCK_LLM_MOD_DIR := $(USERMOD_DIR)
STUCK_LLM_CORE_DIR := $(USERMOD_DIR)/../../../llm/runtime

SRC_USERMOD_C += $(STUCK_LLM_MOD_DIR)/mod_stuck_llm.c
SRC_USERMOD_C += $(STUCK_LLM_CORE_DIR)/stuck_llm.c
CFLAGS_USERMOD += -I$(STUCK_LLM_MOD_DIR) -I$(STUCK_LLM_CORE_DIR) -O3

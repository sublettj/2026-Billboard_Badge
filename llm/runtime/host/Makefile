CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
LDLIBS = -lm

stuck_host: main.c ../stuck_llm.c ../stuck_llm.h
	$(CC) $(CFLAGS) -I.. -o $@ main.c ../stuck_llm.c $(LDLIBS)

# Catches undefined behaviour and bad pointer math in the core before it ever
# runs on a chip with no MMU to complain.
asan: main.c ../stuck_llm.c ../stuck_llm.h
	$(CC) -O1 -g -std=c11 -Wall -fsanitize=address,undefined -I.. -o stuck_host_asan main.c ../stuck_llm.c $(LDLIBS)

clean:
	rm -f stuck_host stuck_host_asan

.PHONY: asan clean

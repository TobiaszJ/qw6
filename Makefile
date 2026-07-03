# qw6 Makefile
# Build targets for the Qwen 3.6-35B-A3B inference engine

CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -g
LDFLAGS ?= -lm

# Vulkan
VULKAN_CFLAGS ?= $(shell pkg-config --cflags vulkan 2>/dev/null || echo "")
VULKAN_LIBS ?= $(shell pkg-config --libs vulkan 2>/dev/null || echo "-lvulkan")

# Sources
SRCS = qw6.c qw6_tok.c
SERVER_SRCS = qw6.c qw6_tok.c qw6-server.c

# Output binaries
BIN = qw6
SERVER = qw6-server
TEST_BUILD_DIR = .build
TEST_TOK_BIN = $(TEST_BUILD_DIR)/test_tok

.PHONY: all cpu vulkan server test bench clean help

all: cpu

help:
	@echo "qw6 build targets:"
	@echo "  make cpu       — CPU-only reference/debug build (no Vulkan needed)"
	@echo "  make vulkan    — Vulkan build for BC-250 (requires Mesa 25.1+)"
	@echo "  make server    — Build HTTP server (Phase 4)"
	@echo "  make test      — Run regression tests"
	@echo "  make test-tokenizer — Run tokenizer regression tests"
	@echo "  make bench     — Run benchmarks"
	@echo "  make clean     — Remove build artifacts"
	@echo ""
	@echo "Current status: Phase 1 (CPU reference path — in progress)"

# CPU-only build (Phase 1)
cpu: $(BIN)

# Vulkan build (Phase 2)
vulkan: CFLAGS += -DQW6_VULKAN $(VULKAN_CFLAGS)
vulkan: LDFLAGS += $(VULKAN_LIBS)
vulkan: $(BIN)

# Server build (Phase 4)
server: $(SERVER)

$(BIN): qw6.c qw6_tok.c qw6.h qw6_tok.h
	$(CC) $(CFLAGS) -o $@ qw6.c qw6_tok.c $(LDFLAGS)

$(SERVER): qw6-server.c qw6.c qw6_tok.c qw6.h qw6_tok.h
	$(CC) $(CFLAGS) -o $@ qw6-server.c qw6.c qw6_tok.c $(LDFLAGS)

$(TEST_BUILD_DIR):
	mkdir -p $@

$(TEST_TOK_BIN): test_tok.c qw6_tok.c qw6_tok.h | $(TEST_BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ test_tok.c qw6_tok.c $(LDFLAGS)

# Test
test: $(BIN)
	./$(BIN) --self-test

test-tokenizer: $(TEST_TOK_BIN)
	./$(TEST_TOK_BIN)

# Bench
bench: $(BIN)
	@echo "Benchmarks not yet implemented — see ROADMAP.md Phase 3"

clean:
	rm -rf $(TEST_BUILD_DIR)
	rm -f $(BIN) $(SERVER) *.o *.obj

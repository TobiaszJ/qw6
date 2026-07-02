# qw6 Makefile
# Build targets for the Qwen 3.6-35B-A3B inference engine

CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror -Wpedantic -O2 -g
LDFLAGS ?=

# Vulkan
VULKAN_SDK ?=
VULKAN_CFLAGS ?= $(shell pkg-config --cflags vulkan 2>/dev/null || echo "")
VULKAN_LIBS ?= $(shell pkg-config --libs vulkan 2>/dev/null || echo "-lvulkan")

# Sources
SRCS = qw6.c
SERVER_SRCS = qw6.c qw6-server.c

# Output binaries
BIN = qw6
SERVER = qw6-server

.PHONY: all cpu vulkan test bench clean help

all: help

help:
	@echo "qw6 build targets:"
	@echo "  make cpu       — CPU-only reference/debug build (no Vulkan needed)"
	@echo "  make vulkan    — Vulkan build for BC-250 (requires Mesa 25.1+)"
	@echo "  make test      — Run regression tests"
	@echo "  make bench     — Run benchmarks"
	@echo "  make clean     — Remove build artifacts"
	@echo ""
	@echo "Current status: pre-alpha (Phase 0 — design complete, no code yet)"

# CPU-only build (Phase 1 target)
cpu: $(BIN)
	@echo "Built $(BIN) (CPU reference path — Phase 1 target, not yet implemented)"

# Vulkan build (Phase 2 target)
vulkan: CFLAGS += -DQW6_VULKAN $(VULKAN_CFLAGS)
vulkan: LDFLAGS += $(VULKAN_LIBS)
vulkan: $(BIN)
	@echo "Built $(BIN) (Vulkan backend — Phase 2 target, not yet implemented)"

$(BIN): qw6.c
	@echo "Phase 1/2 not yet started — qw6.c does not exist"
	@exit 1

# Server
server: $(SERVER)
	@echo "Built $(SERVER) — Phase 4 target, not yet implemented"

# Test
test:
	@echo "Tests not yet implemented — see ROADMAP.md Phase 1"

# Bench
bench:
	@echo "Benchmarks not yet implemented — see ROADMAP.md Phase 3"

clean:
	rm -f $(BIN) $(SERVER) *.o *.obj
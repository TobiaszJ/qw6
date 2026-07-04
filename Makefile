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
VULKAN_SRCS = qw6.c qw6_tok.c qw6_vk.c
SERVER_SRCS = qw6.c qw6_tok.c qw6-server.c
VULKAN_SHADERS = vulkan/rmsnorm.comp vulkan/rmsnorm_apply.comp vulkan/rmsnorm_full.comp vulkan/rmsnorm_heads.comp vulkan/matvec_f32.comp vulkan/matmul_q4k.comp vulkan/matmul_q5k.comp vulkan/matmul_q6k.comp vulkan/matmul_iq2xxs.comp vulkan/matmul_iq2s.comp vulkan/matmul_iq3s.comp vulkan/vec_add.comp vulkan/add.comp vulkan/vec_axpy.comp vulkan/silu_mul.comp vulkan/sigmoid_mul.comp vulkan/l2_norm_heads.comp vulkan/buf_copy.comp vulkan/argmax.comp vulkan/sampling.comp vulkan/rope_mrope.comp vulkan/attention_gqa.comp vulkan/moe_route.comp vulkan/moe_gather.comp vulkan/deltanet_conv1d.comp vulkan/deltanet_conv1d_f32.comp vulkan/deltanet_gated.comp vulkan/deltanet_alpha_beta.comp vulkan/deltanet_norm_gate.comp vulkan/deltanet_update.comp vulkan/deltanet_retrieve.comp vulkan/mtp_draft.comp
VULKAN_SPV = $(VULKAN_SHADERS:.comp=.spv)

# Output binaries
BIN = qw6
SERVER = qw6-server
TEST_BUILD_DIR = .build
TEST_TOK_BIN = $(TEST_BUILD_DIR)/test_tok

.PHONY: all cpu vulkan shaders server test bench clean help

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
vulkan: shaders
	$(CC) $(CFLAGS) -o $(BIN) $(VULKAN_SRCS) $(LDFLAGS)

shaders: $(VULKAN_SPV)

%.spv: %.comp
	glslc -O $< -o $@

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
	rm -f $(BIN) $(SERVER) *.o *.obj $(VULKAN_SPV)

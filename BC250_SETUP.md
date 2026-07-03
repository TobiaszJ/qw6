# BC-250 Hardware Setup Guide

This document describes the hardware and software setup required to run qw6 on
the AMD BC-250. Most of this is already documented by the community — this guide
consolidates everything into a single reference and adds qw6-specific notes.

---

## 1. Hardware

### 1.1 AMD BC-250 Board

| Component | Details |
|---|---|
| Board | ASRock Rack BC-250 (crypto-mining board) |
| APU | AMD Cyan Skillfish (Zen 2 + GFX1013 GPU) |
| CPU | 6c/12t Zen 2 |
| GPU | GFX1013 ("RDNA 1.5") — 40 CUs (24 stock, 40 with unlock) |
| Memory | 16 GB GDDR6, unified (CPU+GPU), 256-bit bus |
| Storage | NVMe (475 GB typical) |
| TDP | 220 W nameplate, ~52–116 W during inference |
| BIOS | Community-patched (unlocks VRAM allocation + chipset menus) |
| Price | ~$140 (AliExpress, as of 2026) |

### 1.2 PS5 Lineage

The Cyan Skillfish silicon is widely associated with Sony's PS5 APU (Oberon).
GFX1013 is informally called "RDNA 1.5" — a GFX10.1-era ISA with ray tracing
extensions more typical of RDNA 2. This is community shorthand, not an official
AMD designation.

---

## 2. OS and Kernel

### 2.0 Current Development Board Snapshot

The active development board used for the current native loader work is reachable
as `teejay@192.168.1.150` and has been verified with:

- Vulkan RADV device: `AMD BC-250 (RADV GFX1013)`
- TTM pages limit: `4194304`
- TTM page pool size: `4194304`
- 16 GiB swap file enabled
- CPU governor set to `performance`
- default boot target set to `multi-user.target`

The 40-CU unlock is not currently active on this board (`active_cu_number 24`).
This does not block CPU reference work or native loader development, but it will
matter for Vulkan performance validation.

### 2.1 Recommended Stack

| Component | Version | Notes |
|---|---|---|
| OS | Fedora 43 (headless) | Best-documented bc250 setup |
| Kernel | 6.18+ | TTM tuning, amdgpu support |
| Mesa | 25.3.4+ | BC-250 support added in 25.1.0 |
| Vulkan | 1.4.328 | Via RADV (Mesa) |
| Display | `multi-user.target` (no GUI) | Saves ~1 GB RAM |

Alternative distributions (Debian/Arch) work with manual Mesa 25.1+ install.

### 2.2 CPU Governor

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 'w /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor - - - - performance' | \
  sudo tee /etc/tmpfiles.d/cpu-governor.conf
```

Stock `schedutil` causes LLM latency spikes.

### 2.3 GPU Governor (oberon)

The `oberon` governor is the **only working clock control** on Cyan Skillfish APUs.
It caps the GPU core at 1500 MHz.

```bash
# Verify governor is active
cat /sys/module/amdgpu/parameters/bc250_cc_write_mode
```

`force_performance_level` accepts only `auto`; manual tier pins return `EINVAL`.

---

## 3. Driver Stack

### 3.1 What Works

| Layer | Status | Notes |
|---|---|---|
| `amdgpu` kernel driver | ✅ | Auto-detected, firmware loaded |
| **Vulkan (RADV/Mesa)** | ✅ | **Primary compute path for qw6** |
| ROCm / HIP | ❌ | `rocblas_abort()` — no GFX1013 solution libraries |
| OpenCL (rusticl) | ⚠️ | Not usable in this configuration |
| AMDVLK | ❌ | Does not support GFX1013 |
| AMDGPU-PRO | ❌ | Does not support BC-250 |

### 3.2 Vulkan Verification

```bash
vulkaninfo --summary
# → GPU0: AMD BC-250 (RADV GFX1013), Vulkan 1.4.328, INTEGRATED_GPU

vulkaninfo | grep -A 20 "VkPhysicalDeviceProperties"
# → deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
# → vendorID = 0x1002 (AMD)
# → driverName = radv

# Ensure NOT using software rendering
vulkaninfo | grep -i llvmpipe
# Should return nothing
```

### 3.3 Why Not ROCm?

GFX1013 is listed in LLVM as supporting `rocm-amdhsa`, but AMD's ROCm userspace
(`rocBLAS`/`Tensile`) does not ship GFX1013 solution libraries. Setting
`HSA_OVERRIDE_GFX_VERSION=10.3.0` to masquerade as `gfx1030` is **not advisable**:
GFX10.1 ISA vs GFX10.3 ISA differences risk silent compute errors. Vulkan already
works and caches compiled shaders to disk. **qw6 uses Vulkan exclusively.**

---

## 4. Memory Tuning

### 4.1 TTM Pages Limit (Critical)

Without this, 14B+ models load but return errors during inference when the KV
cache tries to extend past the TTM ceiling.

```bash
# Runtime (immediate)
echo 4194304 | sudo tee /sys/module/ttm/parameters/pages_limit
echo 4194304 | sudo tee /sys/module/ttm/parameters/page_pool_size

# Persistent
echo "options ttm pages_limit=4194304 page_pool_size=4194304" | \
  sudo tee /etc/modprobe.d/ttm-gpu-memory.conf
printf "w /sys/module/ttm/parameters/pages_limit - - - - 4194304\n\
w /sys/module/ttm/parameters/page_pool_size - - - - 4194304\n" | \
  sudo tee /etc/tmpfiles.d/gpu-ttm-memory.conf
sudo dracut -f
```

### 4.2 Verify (Must Check!)

```bash
cat /sys/module/ttm/parameters/pages_limit
# MUST read 4194304 (16 GiB). 3145728 = 12 GiB ✗
```

**Silent cap gotcha:** `systemd-tmpfiles` runs after boot and can silently
overwrite `modprobe.d` values. If `tmpfiles.d` was ever set to `3145728`,
it wins last. An accidental 12 GB cap breaks things in confusing, misattributed
ways — the model looks "short-context only" or timing out.

### 4.3 GTT

The deprecated `amdgpu.gttsize` parameter should be **removed** from the kernel
cmdline. With `ttm.pages_limit=4194304` alone, GTT allocates 16 GiB. Removing
`amdgpu.gttsize` actually *increased* GTT.

### 4.4 Memory Layout After Tuning

| Region | Size | Notes |
|---|---|---|
| VRAM carveout | 512 MB | BIOS-reserved |
| GTT | 16 GiB | `ttm.pages_limit=4194304` |
| Vulkan logical heaps | 16.5 GiB | Two overlapping views of same pool |

### 4.5 Swap (NVMe Safety Net)

With models consuming 10+ GB on a 16 GB system, NVMe swap is required for
surviving inference peaks.

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=16384 status=progress
sudo chattr +C /swapfile   # disable btrfs copy-on-write
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon -p 10 /swapfile
echo '/swapfile none swap sw,pri=10 0 0' | sudo tee -a /etc/fstab
```

### 4.6 Reduce zram

zram compresses pages in *physical* RAM, competing with the model:

```bash
sudo mkdir -p /etc/systemd/zram-generator.conf.d
echo -e '[zram0]\nzram-size = 2048' | sudo tee /etc/systemd/zram-generator.conf.d/small.conf
```

### 4.7 Disable GUI (Saves ~1 GB)

```bash
sudo systemctl set-default multi-user.target && sudo reboot
```

---

## 5. 40-CU Unlock (Recommended for qw6)

### 5.1 Background

The GFX1013 die physically has 40 Compute Units; the factory firmware masks 16
of them (24 active stock). S. Duggan's community kernel patch re-enables all 40
by writing two GFX10 configuration registers during driver initialisation.

### 5.2 Measured Impact (from akandr/bc250, n=3, 11 models)

| Metric | 24-CU Stock | 40-CU Unlocked | Speedup |
|---|---|---|---|
| Median generation (tok/s) | baseline | 1.32× median | +32% |
| Median prefill (tok/s) | baseline | 1.50× median | +50% |
| Qwen3.6-35B-A3B IQ2_M gen | 59.6 tok/s | 78.0 tok/s | +31% |
| Qwen3.6-35B-A3B IQ2_M prefill | 170.8 tok/s | 250.2 tok/s | +46% |

Prefill gains more because it's more compute-bound. Generation is
bandwidth-bound (357 GB/s), so extra CUs help less but still significant.

### 5.3 Apply

```bash
# Build patched AMDGPU module (follow instructions in duggasco/bc250-40cu-unlock)
# Then set module parameter:
echo "options amdgpu bc250_cc_write_mode=3" | sudo tee /etc/modprobe.d/bc250-40cu.conf
sudo dracut -f && sudo reboot

# Verify:
cat /sys/module/amdgpu/parameters/bc250_cc_write_mode    # → 3
journalctl -k -b 0 | grep active_cu_number               # → active_cu_number 40
# CU topology: SE2×SH2×CU10 (2 shader engines × 2 shader arrays × 10 CUs = 40)
```

### 5.4 Thermal Considerations

- 40-CU draw: ~116 W median (vs. 101 W at 24-CU) — +15 W
- Temperature: ~5°C hotter at peak (57–95°C sustained)
- oberon governor self-throttles to maintain thermal bounds
- Better cooling → better gains (the speedup is conservative due to throttling)

The unlock is gated to PCI ID `0x13fe` (BC-250 only) and produces no persistent
state — the board reverts to stock 24-CU on reboot if the module parameter is removed.

---

## 6. qw6 Memory Budget

With headless CachyOS (~2 GB system), ~14 GB available for qw6:

```
Weights (IQ2_M experts + Q6_K critical + Q8_0 router):
  IQ2_M experts (40 layers):       10.47 GB
  Shared expert (Q6_K):              0.09 GB
  Full attention (Q6_K):             0.13 GB
  Linear attention (Q6_K):          0.64 GB
  Embeddings (Q6_K):                 0.37 GB
  Output projection (Q6_K):         0.37 GB
  Router (Q8_0):                     0.02 GB
  MTP (Q4_K_M):                      0.50 GB
  Norms + conv1d (FP32/FP16):      <0.01 GB

Overhead:
  DeltaNet state (FP16, fixed):     0.50 GB
  KV cache Q4_0 @ 64K:              0.34 GB  (10 full-attn layers only)
  Compute scratch:                  0.50 GB
───────────────────────────────────────────────
Total:                             13.94 GB
Headroom:                          +0.06 GB
```

Fits in 14 GB — no SSD streaming, no disk KV, no context truncation.
Q6_K for critical components: near-lossless quality, +0.37 GB vs Q4_K_M.
IQ3/Q4 for all experts does NOT fit (12.34 GB for experts alone at IQ3_XXS).

---

## 7. SSH Access (for Remote Development)

The BC-250 is headless and accessed via SSH. Typical setup:

```bash
# On the BC-250 (one-time):
sudo systemctl enable sshd
sudo systemctl start sshd

# From development machine:
ssh teejay@<bc250-ip>

# Add to ~/.ssh/config for convenience:
Host bc250
    HostName <bc250-ip>
    User teejay
    IdentityFile ~/.ssh/id_ed25519
```

Development workflow:
1. Code on the dev machine (RK3588 or laptop)
2. Push to GitHub
3. `ssh bc250` → `git pull` → `make vulkan` → `./qw6 --bench`
4. Iterate

---

## 8. References

- [akandr/bc250](https://github.com/akandr/bc250) — Full setup guide, benchmarks, roofline
- [elektricm/amd-bc250-docs](https://elektricm.github.io/amd-bc250-docs/) — Comprehensive docs
- [duggasco/bc250-40cu-unlock](https://github.com/duggasco/bc250-40cu-unlock) — 40-CU patch
- [BC-250 on Phoronix](https://www.phoronix.com/news/AMD-RADV-PS5-BC-250) — RADV support
- [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) — Hybrid attention

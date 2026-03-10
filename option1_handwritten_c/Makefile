# =============================================================================
# Makefile — LSTM inference engine for STM32F746G-Discovery
#
# Targets:
#   host    — native GCC build for functional testing on the development machine
#   stm32   — cross-compiled ELF for STM32F746G (arm-none-eabi-gcc required)
#   clean   — remove all build artefacts
#
# Usage:
#   make host              # build and produce ./lstm_host
#   make stm32             # build and produce ./lstm_stm32.elf
#   make stm32 CMSIS_DIR=/path/to/CMSIS/DSP/Include
#   make clean
# =============================================================================

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
SRCS   := lstm_model.c main_benchmark.c
HDRS   := lstm_model.h lstm_weights.h

# -----------------------------------------------------------------------------
# CMSIS-DSP include path (optional — only needed if USE_CMSIS is defined)
#
# Set this to the directory containing arm_math.h, e.g.:
#   CMSIS_DIR = /opt/CMSIS/CMSIS/DSP/Include
#
# Leave empty to build without CMSIS (uses standard math.h only).
# -----------------------------------------------------------------------------
CMSIS_DIR ?=

# Append CMSIS include if the variable is set
ifneq ($(CMSIS_DIR),)
CMSIS_FLAGS := -DUSE_CMSIS -I$(CMSIS_DIR)
else
CMSIS_FLAGS :=
endif

# =============================================================================
# Host build — native GCC
# =============================================================================
HOST_CC      := gcc
HOST_BINARY  := lstm_host
HOST_OBJDIR  := obj_host

# -march=native:  use all ISA extensions available on the build machine
#                 (AVX2/FMA on modern x86-64 gives a useful speed reference)
# -O3:            full optimisation (matches the STM32 build)
# -ffast-math:    allows reassociation and uses fast FP sqrt/exp intrinsics
#                 (matches the embedded build; do NOT use for numerically
#                 sensitive code that relies on IEEE 754 exact rounding)
# -fno-omit-frame-pointer: keep frame pointers for profiling with perf/gprof
HOST_CFLAGS  := -O3 -march=native -ffast-math \
                -fno-omit-frame-pointer \
                -Wall -Wextra -Wpedantic \
                -std=c11 \
                -I. \
                $(CMSIS_FLAGS)

HOST_LDFLAGS := -lm

HOST_OBJS    := $(patsubst %.c,$(HOST_OBJDIR)/%.o,$(SRCS))

.PHONY: host
host: $(HOST_BINARY)

$(HOST_BINARY): $(HOST_OBJS)
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $^ $(HOST_LDFLAGS)
	@echo "Built host binary: $@"

$(HOST_OBJDIR)/%.o: %.c $(HDRS) | $(HOST_OBJDIR)
	$(HOST_CC) $(HOST_CFLAGS) -c -o $@ $<

$(HOST_OBJDIR):
	mkdir -p $@

# =============================================================================
# STM32 build — arm-none-eabi-gcc cross-compiler
#
# Target device: STM32F746G-Discovery
#   MCU:     STM32F746NG
#   Core:    ARM Cortex-M7 (r0p1)
#   FPU:     FPv5-D16 (32 × 64-bit VFP registers, single + double precision
#            hardware — but we use single-precision only for speed)
#   HCLK:    216 MHz
#   Flash:   1 MB  (weights live here)
#   SRAM:    320 KB (LSTMState + stack live here)
#
# Compiler flags explained:
#   -mcpu=cortex-m7       : enable Cortex-M7 instruction set and pipeline
#                           scheduling (dual-issue, 6-stage FP pipeline)
#   -mthumb               : Thumb-2 ISA (16/32-bit mixed encoding, required
#                           for Cortex-M)
#   -mfpu=fpv5-d16        : select the FPv5-D16 VFP unit (VMLA.F32, VCVT,
#                           VSQRT.F32, etc.)
#   -mfloat-abi=hard      : pass float arguments in VFP registers (ABI must
#                           match the C library and startup code)
#   -O3                   : full loop/inlining optimisation
#   -ffast-math           : relax IEEE 754 rules; enables fused multiply-add
#                           (VMLA.F32) and expf() intrinsic from libm
#   -ffunction-sections   : place each function in its own ELF section so the
#                           linker can remove unused code (--gc-sections)
#   -fdata-sections       : same for data (removes unused const tables)
#   -DSTM32               : selects DWT timing in main_benchmark.c
# =============================================================================
STM32_CC      := arm-none-eabi-gcc
STM32_BINARY  := lstm_stm32.elf
STM32_OBJDIR  := obj_stm32

STM32_ARCH    := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard

STM32_CFLAGS  := $(STM32_ARCH) \
                 -O3 -ffast-math \
                 -ffunction-sections -fdata-sections \
                 -Wall -Wextra \
                 -std=c11 \
                 -DSTM32 \
                 -I. \
                 $(CMSIS_FLAGS)

# --gc-sections: discard unreferenced sections (reduces flash usage)
# --specs=nano.specs: link against newlib-nano (smaller printf/malloc)
# --specs=nosys.specs: stub out OS calls (semihosting disabled in production)
#
# Replace nosys.specs with rdimon.specs if you want semihosted printf output
# via the debugger (OpenOCD / J-Link).
STM32_LDFLAGS := $(STM32_ARCH) \
                 -Wl,--gc-sections \
                 --specs=nano.specs \
                 --specs=nosys.specs \
                 -lm

STM32_OBJS    := $(patsubst %.c,$(STM32_OBJDIR)/%.o,$(SRCS))

.PHONY: stm32
stm32: $(STM32_BINARY)

$(STM32_BINARY): $(STM32_OBJS)
	$(STM32_CC) $(STM32_LDFLAGS) -o $@ $^
	$(STM32_CC) -mcpu=cortex-m7 -mthumb \
	            -print-memory-usage $(STM32_LDFLAGS) -o /dev/null $@ 2>/dev/null || true
	arm-none-eabi-size $@
	@echo "Built STM32 ELF: $@"

$(STM32_OBJDIR)/%.o: %.c $(HDRS) | $(STM32_OBJDIR)
	$(STM32_CC) $(STM32_CFLAGS) -c -o $@ $<

$(STM32_OBJDIR):
	mkdir -p $@

# =============================================================================
# Utility targets
# =============================================================================

# Disassemble the STM32 binary to verify FPU instruction usage
.PHONY: disasm
disasm: $(STM32_BINARY)
	arm-none-eabi-objdump -d --no-show-raw-insn $< | \
	    grep -E "vmla|vmul|vadd|vsub|vdiv|vldr|vstr|vcvt" | head -40
	@echo "(showing first 40 FPU instructions)"

# Print section sizes for flash/RAM budgeting
.PHONY: size
size: $(STM32_BINARY)
	arm-none-eabi-size --format=berkeley $<

# Run the host benchmark
.PHONY: run
run: $(HOST_BINARY)
	./$(HOST_BINARY)

# Remove all build artefacts
.PHONY: clean
clean:
	rm -rf $(HOST_OBJDIR) $(STM32_OBJDIR) \
	       $(HOST_BINARY) $(STM32_BINARY)
	@echo "Clean complete."

# Default target
.DEFAULT_GOAL := host

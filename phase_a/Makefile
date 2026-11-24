# Phase A build. Every benchmark is one standalone .cu -> one binary.
#
#   make                 build all four
#   make vector_add      build one
#   make run             build and run everything, appending to ../results/*.csv
#   make clean
#
# ARCH is intentionally explicit. On Colab's T4 it is sm_75; pass
# ARCH=native to let nvcc detect the local card instead.

NVCC   ?= nvcc
ARCH   ?= sm_75
BUILD  := ../build
INCLUDE := -I../include
# -lineinfo keeps source correlation for profilers without hurting optimisation.
# No --use_fast_math: it changes floating-point semantics, and these kernels are
# checked element-by-element against a CPU reference.
NVFLAGS := -O3 -std=c++17 -lineinfo -Xcompiler -Wall $(INCLUDE) -arch=$(ARCH)
LDLIBS  := -lcublas

BINS := $(BUILD)/vector_add $(BUILD)/reduction $(BUILD)/matmul $(BUILD)/streams

.PHONY: all clean run

all: $(BINS)

# One pattern rule for every benchmark; the header is a dependency so a change
# to the timing/CSV helpers rebuilds everything that includes it.
$(BUILD)/%: %.cu ../include/bench_common.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDLIBS)

run: all
	$(BUILD)/vector_add --csv ../results/vector_add.csv
	$(BUILD)/reduction  --csv ../results/reduction.csv
	$(BUILD)/matmul     --csv ../results/matmul.csv
	$(BUILD)/streams    --csv ../results/streams.csv

clean:
	rm -rf $(BUILD)/vector_add $(BUILD)/reduction $(BUILD)/matmul $(BUILD)/streams

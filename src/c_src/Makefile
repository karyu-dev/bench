CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fPIC -shared -fopenmp -fno-tree-loop-distribute-patterns

TARGETS = libstream.so libfma.so liblatency.so libomp.so libbranch_alu.so

all: $(TARGETS)

libstream.so: stream_kernels.c
	$(CC) $(CFLAGS) $< -o $@

libfma.so: fma_kernel.c
	$(CC) $(CFLAGS) $< -o $@

liblatency.so: latency_kernel.c
	$(CC) $(CFLAGS) $< -o $@

libomp.so: omp_kernels.c
	$(CC) $(CFLAGS) $< -o $@

libbranch_alu.so: branch_alu_kernel.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fPIC -shared -fno-tree-loop-distribute-patterns

TARGET  = libstream.so
SRC     = stream_kernels.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean

######################################################################
# Compiler selection
CC = nvcc

######################################################################
# Flags and definitions
FLAG1 = -DNDEBUG
# FLAG2 = -DKLT_USE_QSORT   # Uncomment if you want to use standard qsort
CFLAGS = $(FLAG1) $(FLAG2) -pg

# CUDA flags... Limit to 48 registers per thread
CUFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -maxrregcount=48

######################################################################
# Source files
EXAMPLES = example1 example2 example3 example4 example5
ARCH_C = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
          storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
ARCH_CU = Horizontal_GPU.cu

LIB = -L/usr/local/lib -L/usr/lib

######################################################################
# Compile rules
.SUFFIXES: .c .o .cu

.c.o:
	@echo "Compiling $< to $@"
	$(CC) -c $(CFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created successfully"; else echo "Failed to create $@"; exit 1; fi

%.o: %.cu
	@echo "Compiling CUDA source $< to $@"
	nvcc -c -O3 $(CUFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created successfully"; else echo "Failed to create $@"; exit 1; fi

######################################################################
# Build library
libklt.a: $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "Creating libklt.a with object files..."
	rm -f libklt.a
	ar ruv libklt.a $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "Library libklt.a created successfully."

lib: libklt.a

######################################################################
# Build all examples
all: libklt.a $(EXAMPLES)

$(EXAMPLES): %: %.c libklt.a
	@echo "Linking $@ ..."
	nvcc -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm -lcudart -pg
	@if [ -f $@ ]; then echo "$@ built successfully"; else echo "Failed to build $@"; exit 1; fi

######################################################################
# Dependencies and cleanup
depend:
	makedepend $(ARCH_C) $(ARCH_CU) $(EXAMPLES:=.c)

clean:
	rm -f *.o *.a $(EXAMPLES) *.tar *.tar.gz libklt.a \
	      images/set*/feat*.ppm features.ft features.txt gmon.out p.dot finalProfile.pdf profile_output.txt
	rm -f $(EXEC) $(OBJS) *~ gmon.out
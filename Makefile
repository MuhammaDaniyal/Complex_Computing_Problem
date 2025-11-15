######################################################################
# Compiler selection
NVCC = nvcc
ACC_CC = nvc

######################################################################
# Flags
CFLAGS = -DNDEBUG -pg
NVCCFLAGS = -std=c++11 -O3 -arch=sm_75
CUFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -maxrregcount=48
ACCFLAGS = -acc -Minfo=accel -fast

LIBS = -L/usr/local/lib -L/usr/lib -lm -lcudart

######################################################################
# Source files
EXAMPLES = example3
ARCH_C = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
          storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
ARCH_CU = Horizontal_GPU.cu

######################################################################
# Suffixes
.SUFFIXES: .c .o .cu

######################################################################
# Compile rules

# Normal C compilation (CPU-side)
.c.o:
	@echo "Compiling $< to $@ (CPU)..."
	$(NVCC) -c $(CFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created successfully"; else echo "Failed to create $@"; exit 1; fi

# CUDA compilation
%.o: %.cu
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) $(CUFLAGS) -c $< -o $@
	@if [ -f $@ ]; then echo "$@ created successfully"; else echo "Failed to create $@"; exit 1; fi

# OpenACC compilation for files with directives (e.g., convolve.c)
convolve_acc.o: convolve.c
	@echo "Compiling convolve.c with OpenACC using nvc..."
	$(ACC_CC) $(ACCFLAGS) -fPIC -c convolve.c -o convolve_acc.o
	@if [ -f convolve_acc.o ]; then echo "convolve_acc.o created successfully"; else echo "Failed to create convolve_acc.o"; exit 1; fi




######################################################################
# Build libraries

# CPU/CUDA library (all except convolve)
libklt.a: $(filter-out convolve.o,$(ARCH_C:.c=.o)) $(ARCH_CU:.cu=.o)
	@echo "Creating libklt.a..."
	rm -f libklt.a
	ar ruv libklt.a $(filter-out convolve.o,$(ARCH_C:.c=.o)) $(ARCH_CU:.cu=.o)
	@echo "libklt.a created."

# OpenACC-enabled library (replace convolve.o with convolve_acc.o)
libklt_acc.a: convolve_acc.o $(filter-out convolve.o,$(ARCH_C:.c=.o)) $(ARCH_CU:.cu=.o)
	@echo "Creating libklt_acc.a..."
	rm -f libklt_acc.a
	ar ruv libklt_acc.a convolve_acc.o $(filter-out convolve.o,$(ARCH_C:.c=.o)) $(ARCH_CU:.cu=.o)
	@echo "libklt_acc.a created."

######################################################################
# Build example executable (link CUDA + OpenACC)
all: libklt.a libklt_acc.a $(EXAMPLES)

# Build example executable (link CUDA + OpenACC)
$(EXAMPLES): %: %.c libklt_acc.a
	@echo "Building $@ with NVCC + OpenACC library..."
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $@.c -L. -lklt_acc $(LIBS) -no-pie -o $@
	@if [ -f $@ ]; then echo "$@ built successfully"; else echo "Failed to build $@"; exit 1; fi

######################################################################
# Dependencies and cleanup
depend:
	makedepend $(ARCH_C) $(ARCH_CU) $(EXAMPLES:=.c)

clean:
	rm -f *.o *.a $(EXAMPLES) *.tar *.tar.gz libklt.a libklt_acc.a \
	      images/set*/feat*.ppm features.ft features.txt gmon.out p.dot finalProfile.pdf profile_output.txt
	rm -f *~ gmon.out

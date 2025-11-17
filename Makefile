######################################################################
# Compilers (NVIDIA HPC SDK)
######################################################################

ACC_CC   = nvc          # OpenACC C compiler
CUDA_CC  = nvc++        # CUDA/C++ compiler (replaces nvcc)

######################################################################
# Flags
######################################################################

FLAG1   = -DNDEBUG
CFLAGS  = $(FLAG1) $(FLAG2) -pg

# CUDA flags for nvc++ (correct form — nvc++ does NOT support -maxrregcount)
CUFLAGS = -cuda -O3 -gpu=maxregcount:48

# OpenACC flags
ACCFLAGS = -acc -Minfo=accel -fast

######################################################################
# Source files
######################################################################

EXAMPLES = example3

# All C files
ARCH_C = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# CUDA files
ARCH_CU = Horizontal_GPU.cu

# Which C files contain OpenACC?
ACC_SOURCES = convolve.c

# Regular CPU-only C files
CPU_SOURCES = $(filter-out $(ACC_SOURCES),$(ARCH_C))

# No need for CUDA toolkit paths — HPC SDK provides all required libs internally
LIB = -L/usr/local/lib -L/usr/lib

######################################################################
# Rules
######################################################################

.SUFFIXES: .c .o .cu

######################################################################
# Compile C files without OpenACC
######################################################################
$(CPU_SOURCES:.c=.o): %.o: %.c
	@echo "Compiling $< (C, CPU only)..."
	$(ACC_CC) -c $(CFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created"; else echo "Failed: $@"; exit 1; fi

######################################################################
# Compile OpenACC C files
######################################################################
$(ACC_SOURCES:.c=.o): %.o: %.c
	@echo "Compiling $< with OpenACC..."
	$(ACC_CC) $(ACCFLAGS) -c $< -o $@
	@if [ -f $@ ]; then echo "$@ created"; else exit 1; fi

######################################################################
# CUDA compilation using nvc++
######################################################################
%.o: %.cu
	@echo "Compiling CUDA source $< using nvc++..."
	$(CUDA_CC) -c $(CUFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created"; else echo "Failed: $@"; exit 1; fi

######################################################################
# Build libraries
######################################################################

libklt.a: $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "Creating libklt.a..."
	rm -f libklt.a
	ar ruv libklt.a $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "libklt.a created."

# OpenACC version: convolve.o replaced correctly
libklt_acc.a: $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "Creating libklt_acc.a..."
	rm -f libklt_acc.a
	ar ruv libklt_acc.a $(ARCH_C:.c=.o) $(ARCH_CU:.cu=.o)
	@echo "libklt_acc.a created."

######################################################################
# Build examples using OpenACC-enabled library
######################################################################

all: libklt.a libklt_acc.a $(EXAMPLES)

$(EXAMPLES): %: %.c libklt_acc.a
	@echo "Linking $@ with OpenACC + CUDA library..."
	$(ACC_CC) $(ACCFLAGS) $(CFLAGS) -o $@ $@.c -L. -lklt_acc $(LIB) -lm
	@if [ -f $@ ]; then echo "$@ built"; else echo "Failed: $@"; exit 1; fi

######################################################################
# Cleanup
######################################################################

clean:
	rm -f *.o *.a $(EXAMPLES) *.tar *.tar.gz libklt.a libklt_acc.a \
	      images/set*/feat*.ppm features.ft features.txt gmon.out p.dot finalProfile.pdf profile_output.txt
	rm -f $(EXEC) $(OBJS) *~ gmon.out


######################################################################
# Compilers (NVIDIA HPC SDK)
######################################################################

ACC_CC  = nvc             # OpenACC C compiler
CUDA_CC = nvc++           # CUDA/C++ compiler (replaces nvcc)

######################################################################
# Flags
######################################################################

FLAG1   = -DNDEBUG
CFLAGS  = $(FLAG1) $(FLAG2) -pg

# CUDA flags for nvc++ 
CUFLAGS = -cuda -O3 -gpu=maxregcount:48

# OpenACC flags
ACCFLAGS = -acc -Minfo=accel -fast

######################################################################
# Source files and Object lists
######################################################################

EXAMPLES = example3

# All C files
ARCH_C = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
         storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

# CUDA files
ARCH_CU = Horizontal_GPU.cu
ARCH_CU_OBJS = $(ARCH_CU:.cu=.o)

# Which C files contain OpenACC?
ACC_SOURCES = convolve.c
ACC_OBJS = $(ACC_SOURCES:.c=_acc.o)

# C files compiled for CPU (all C files MINUS the OpenACC files)
CPU_ONLY_SOURCES = $(filter-out $(ACC_SOURCES),$(ARCH_C))
CPU_ONLY_OBJS = $(CPU_ONLY_SOURCES:.c=.o)

# All object files for the standard library
STANDARD_OBJS = $(ARCH_C:.c=.o) $(ARCH_CU_OBJS)
# All object files for the OpenACC library
ACC_LIBRARY_OBJS = $(CPU_ONLY_OBJS) $(ACC_OBJS) $(ARCH_CU_OBJS)


LIB = -L/usr/local/lib -L/usr/lib

######################################################################
# Rules
######################################################################

.SUFFIXES: .c .o .cu

######################################################################
# Compile C files without OpenACC (For all files *except* convolve.c)
######################################################################
$(CPU_ONLY_OBJS): %.o: %.c
	@echo "Compiling $< (C, CPU only)..."
	$(ACC_CC) -c $(CFLAGS) $< -o $@
	@if [ -f $@ ]; then echo "$@ created"; else echo "Failed: $@"; exit 1; fi

######################################################################
# Compile OpenACC C files (Only convolve.c)
######################################################################
$(ACC_OBJS): %_acc.o: %.c
	@echo "Compiling $< with OpenACC to $@..."
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

libklt.a: $(STANDARD_OBJS)
	@echo "Creating libklt.a (Standard Library)..."
	rm -f libklt.a
	ar ruv libklt.a $(STANDARD_OBJS)
	@echo "libklt.a created."

libklt_acc.a: $(ACC_LIBRARY_OBJS)
	@echo "Creating libklt_acc.a (OpenACC Library)..."
	rm -f libklt_acc.a
	ar ruv libklt_acc.a $(ACC_LIBRARY_OBJS)
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

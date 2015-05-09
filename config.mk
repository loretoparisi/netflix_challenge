#	config.mk (Makefile configuration)
#
#		Supports Cython, C++
#


# Default shell
SHELL = /bin/sh
# Default C++ compiler
CXX = g++
# Default Cython compiler (pyx -> cpp)
CYTHON = cython

# Base source directory
srcdir = src
# Directories containing source files
modules = $(filter-out %__pycache__,$(shell find $(srcdir) -type d))
# Directory containing header files
incdir = src
# Destination directory for object files and libraries
libdir = lib
# Destination directory for binaries
bindir = bin


# Default compiler flags (for C and C++)
override CFLAGS += -Wall -Wextra -I$(incdir) -march=core-avx2 -m64 -O3 \
-flto -fomit-frame-pointer -pipe -Wno-reorder -Wno-unused-function \
-Wno-parentheses
# Default compiler flags for C++
override CXXFLAGS += $(CFLAGS) -std=c++11 -Wno-write-strings
# Default linker flags
override LD_FLAGS += -L$(libdir) -Wl,-O3,--sort-common,--as-needed,-z,relro \
-flto -std=c++11

# Extra linker flags for Armadillo.
ARMA_LDFLAGS = -larmadillo

# Compiler flags for compiling with Intel math kernel library
MKL_CFLAGS =  -ffast-math -ftree-vectorize -fopt-info-vec -mveclibabi=svml
# Extra linker flags for Intel math kernel library
MKL_LDFLAGS = -L/opt/intel/lib -lsvml -ffast-math

# Default Cython flags
CYTHON_FLAGS = -2

# Compiler flags for compiling with Cython
CYTHON_CFLAGS = -I/usr/include/python3.4m -Wno-strict-aliasing
# Linker flags for linking object files compiled from Cython-generated code
CYTHON_LDFLAGS = -I/usr/include/python3.4m -lpython3.4m

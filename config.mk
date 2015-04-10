#	config.mk (Makefile configuration)
#
#		Supports Cython, C, C++
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
modules = $(shell find $(srcdir) -type d)
# Directory containing header files
includedir = src
# Destination directory for object files and libraries
libdir = lib
# Destination directory for binaries
bindir = bin


# Default compiler flags (for C and C++)
override CFLAGS += -Wall -O3 -march=native -mtune=generic -Wno-reorder
# -pipe -fstack-protector --param=ssp-buffer-size=4 -Wno-sign-compare -Wno-unused-function
# Default compiler flags for C++
override CXXFLAGS += $(CFLAGS) -std=c++11 -Wno-write-strings
# Default linker flags
override LD_FLAGS += -L$(libdir) -Wl,-O1,--sort-common,--as-needed,-z,relro -std=c++11 

# Linker flags for Armadillo
EXTRA_LDFLAGS = -larmadillo

# Default Cython flags
CYTHON_FLAGS = -2

# Compiler flags for compiling with Cython
CYTHON_CFLAGS = -I/usr/include/python3.4m -Wno-strict-aliasing
# Linker flags for linking object files compiled from Cython-generated code
CYTHON_LDFLAGS = -lpython3.4m

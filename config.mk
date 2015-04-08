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

# Directory containing source files
srcdir = $(shell find src -type d)
# Directory containing header files
includedir = src
# Destination directory for object files and libraries
libdir = lib
# Destination directory for binaries
bindir = bin


# Default compiler flags (for C and C++)
override CFLAGS += -O3 -march=native -mtune=generic
# -pipe -fstack-protector --param=ssp-buffer-size=4 -Wno-sign-compare -Wno-unused-function
# Default compiler flags for C++
override CXXFLAGS += $(CFLAGS) -std=c++11 -Wno-write-strings
# Default linker flags
override LD_FLAGS += -L$(libdir) -Wl,-O1,--sort-common,--as-needed,-z,relro 

# Default Cython flags
CYTHON_FLAGS = -2

# Compiler flags for compiling with Cython
CYTHON_CFLAGS = -I/usr/include/python3.4m -Wno-strict-aliasing
# Linker flags for linking object files compiled from Cython-generated code
CYTHON_LDFLAGS = -lpython3.4m

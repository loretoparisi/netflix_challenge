# 	Makefile 
#
#		Supports Cython, C++
#

include config.mk

include $(addsuffix /module.mk, $(modules))

vpath %.cc $(modules)
vpath %.pyx $(modules)

# Absolute file paths for dynamic libraries
LIB_FILES = $(foreach lib, $(LIBS), $(libdir)/$(lib))
# Absolute file paths for Cython extensions
EXT_FILES = $(foreach ext, $(EXTS), $(libdir)/$(ext))
EXT_LINKS = $(foreach ext, $(EXTS), $(srcdir)/$(ext))
# Absolute file paths for binary executables
BIN_FILES = $(foreach bin, $(BINS), $(bindir)/$(bin))


# Clear allowed target suffixes
.SUFFIXES:

# Set allowed target suffixes
.SUFFIXES: .py .pyx .pxd .cc .hh .o .so

# Declare all phony targets
.PHONY: all clean mklib mkbin


# All (final) targets
all: $(LIB_FILES) $(EXT_FILES) $(BIN_FILES)

# Clean up all files
clean:
	@# Remove all make-generated files
	$(RM) -f $(libdir)/*.o $(LIB_FILES) $(EXT_FILES) $(EXT_LINKS) $(BIN_FILES)
	@# Remove Python btyecode
	for i in `find . -type f -iname "*.pyc"`; do \
		$(RM) "$$i"; \
	done
	@# If libdir is empty, remove it
	if [[ -d $(libdir) && -z "`ls -A $(libdir)`" ]]; then \
		$(RM) -rf $(libdir); \
	fi
	@# If bindir is empty, remove it
	if [[ -d $(bindir) && -z "`ls -A $(bindir)`" ]]; then \
		$(RM) -rf $(bindir); \
	fi

# Create directory for containing object files and libraries
mklib:
	@# Make libdir if it doesn't already exist
	test -d $(libdir) || mkdir $(libdir)

# Create directory for containing binaries
mkbin:
	@# Make bindir if it doesn't already exist
	test -d $(bindir) || mkdir $(bindir)


# Generate a C++ file from a Cython file
%.cc: %.pyx
	$(CYTHON) $(CYTHON_FLAGS) --cplus $< -o $@

# Additional compiler flags for all object files go here (using EXTRA_CFLAGS)
$(libdir)/svdpp.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/svdpp_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/timesvdpp.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/timesvdpp_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/globals.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/globals_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/knn.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/knn_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/svd.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/svd_only_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
$(libdir)/interface.o: private EXTRA_CFLAGS += $(CYTHON_CFLAGS)

# Implicit rule to generate object files
$(libdir)/%.o: %.cc | mklib
	@# Everything is compiled with -fPIC so they can be dynamically linked
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -fPIC -c $< -o $@

# Dependencies for all library targets go here
$(libdir)/interface.so: $(libdir)/interface.o $(libdir)/svdpp.o \
$(libdir)/netflix.o

# Additional linker flags for all library targets go here (using EXTRA_LDFLAGS)
$(libdir)/interface.so: private EXTRA_LDFLAGS += $(CYTHON_LDFLAGS) \
$(ARMA_LDFLAGS)

# Implicit rule matching the C/C++ dynamic library naming convention
$(libdir)/lib%.so: $(libdir)/%.o | mklib
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $^ -o $@ $(EXTRA_LDFLAGS)

# Any shared object not matching the C/C++ library naming convention is
# assumed to be a Cython extension; shared objects must match the name of the
# Cython interfaces to avoid runtime errors
$(libdir)/%.so: $(libdir)/%.o
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $^ -o $@ $(EXTRA_LDFLAGS)
	@# Remove any existing link to an old extension, and the generated source
	$(RM) $(@:$(libdir)%=$(srcdir)%) $(@:$(libdir)/%.so=%.cc)
	@# Link the extension back to our source directory for use
	ln -s $@ $(srcdir)

# Dependencies for all binary targets go here
$(bindir)/rbm_test: $(libdir)/rbm.o $(libdir)/netflix.o
$(bindir)/svdpp_test: $(libdir)/svdpp.o $(libdir)/netflix.o
$(bindir)/timesvdpp_test: $(libdir)/timesvdpp.o $(libdir)/netflix.o
$(bindir)/globals_test: $(libdir)/globals.o $(libdir)/netflix.o
$(bindir)/knn_test: $(libdir)/knn.o $(libdir)/netflix.o
$(bindir)/svd_only_test: $(libdir)/knn.o $(libdir)/netflix.o
$(bindir)/binarize_data: $(libdir)/netflix.o

# Additional linker flags for all binary targets go here (using EXTRA_LDFLAGS)
$(bindir)/rbm_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
$(bindir)/svdpp_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
$(bindir)/timesvdpp_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
$(bindir)/globals_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
$(bindir)/knn_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
$(bindir)/svd_only_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)

# Default rule for compiling binaries
$(bindir)/%: $(libdir)/%.o | mkbin
	$(CXX) $(LD_FLAGS) $(filter-out mkbin,$^) -o $@ $(EXTRA_LDFLAGS)


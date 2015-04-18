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
all: $(LIBS) $(EXTS) $(BINS)

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
%.cc: %.pyx mklib
	$(CYTHON) $(CYTHON_FLAGS) --cplus $< -o $@

# Additional compiler flags for all object files go here (using EXTRA_CFLAGS)
lib/svdpp.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
lib/svdpp_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
lib/knn.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
lib/knn_test.o: private EXTRA_CFLAGS += -DARMA_NO_DEBUG
lib/interface.o: private EXTRA_CFLAGS += $(CYTHON_CFLAGS)

# Implicit rule to generate object files
$(libdir)/%.o: %.cc mklib
	@# Everything is compiled with -fPIC so they can be dynamically linked
	$(CXX) $(CXXFLAGS) $(EXTRA_CFLAGS) -fPIC -c $< -o $@

# Dependencies for all library targets go here
interface.so: src/interface.cc lib/interface.o lib/svdpp.o lib/netflix.o

# Additional linker flags for all library targets go here (using EXTRA_LDFLAGS)
interface.so: private EXTRA_LDFLAGS += $(CYTHON_LDFLAGS) $(ARMA_LDFLAGS)

# Implicit rule matching the C/C++ dynamic library naming convention
lib%.so: $(libdir)/%.o
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $^ -o $(libdir)/$@ \
	$(EXTRA_LDFLAGS)

# Any shared object not matching the C/C++ library naming convention is
# assumed to be a Cython extension; shared objects must match the name of the
# Cython interfaces to avoid runtime errors
%.so: $(libdir)/%.o
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $(filter-out %.cc,$^) \
	-o $(libdir)/$@ $(EXTRA_LDFLAGS)
	@# Link the extension back to our source directory for use
	@# The .. gets directories right (for now)
	$(RM) $(srcdir)/$@
	ln -s ../$(libdir)/$@ $(srcdir)

# Dependencies for all binary targets go here
rbm_test: lib/rbm.o lib/netflix.o
svdpp_test: lib/svdpp.o lib/netflix.o
knn_test: lib/knn.o lib/netflix.o
binarize_data: lib/netflix.o

# Additional linker flags for all binary targets go here (using EXTRA_LDFLAGS)
rbm_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
svdpp_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)
knn_test: private EXTRA_LDFLAGS += $(ARMA_LDFLAGS)

# Default rule for compiling binaries
$(BINS): %: $(libdir)/%.o mkbin
	$(CXX) $(LD_FLAGS) $(filter-out mkbin,$^) -o $(bindir)/$@ $(EXTRA_LDFLAGS)


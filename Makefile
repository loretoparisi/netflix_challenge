# 	Makefile 
#
#		Supports Cython, C++
#

include config.mk

include $(addsuffix /module.mk, $(srcdir))

vpath %.cc $(srcdir)
vpath %.pyx $(srcdir)

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
	$(CYTHON) $(CYTHON_FLAGS) --cplus $(srcdir)/$< -o $(srcdir)/$@

# Implicit rule to generate object files
lib/%.o: %.cc mklib
	@# Everything is compiled with -fPIC so they can be dynamically linked
	$(CXX) $(ALL_CXXFLAGS) $(EXTRA_CFLAGS) -fPIC -c $< -o $@

# Implicit rule matching the C/C++ dynamic library naming convention
lib%.so: lib/%.o
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $(libdir)/$< -o $(libdir)/$@ \
	$(EXTRA_LDFLAGS)

# Any shared object not matching the C/C++ library naming convention is
# assumed to be a Cython extension; shared objects must match the name of the
# Cython interfaces to avoid runtime errors
%.so: lib/%.o
	$(CXX) $(LD_FLAGS) -shared -Wl,-soname,$@ $(libdir)/$< -o $(libdir)/$@ \
	$(EXTRA_LDFLAGS)
	@# Link the extension back to our source directory for use
	ln -s $(libdir)/$@ $(srcdir)

# Default rule for compiling binaries
$(BINS): %: lib/%.o mkbin
	$(CXX) $(LD_FLAGS) $< -o $(bindir)/$@ $(EXTRA_LDFLAGS)

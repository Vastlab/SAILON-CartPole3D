# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tboult/WORK/bullet3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tboult/WORK/bullet3/build_cmake

# Include any dependencies generated for this target.
include Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/depend.make

# Include the progress variables for this target.
include Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/progress.make

# Include the compile flags for this target's objects.
include Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/flags.make

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/flags.make
Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o: ../Extras/GIMPACTUtils/btGImpactConvexDecompositionShape.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tboult/WORK/bullet3/build_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o"
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o -c /home/tboult/WORK/bullet3/Extras/GIMPACTUtils/btGImpactConvexDecompositionShape.cpp

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.i"
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tboult/WORK/bullet3/Extras/GIMPACTUtils/btGImpactConvexDecompositionShape.cpp > CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.i

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.s"
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tboult/WORK/bullet3/Extras/GIMPACTUtils/btGImpactConvexDecompositionShape.cpp -o CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.s

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.requires:

.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.requires

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.provides: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.requires
	$(MAKE) -f Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/build.make Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.provides.build
.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.provides

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.provides.build: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o


# Object files for target GIMPACTUtils
GIMPACTUtils_OBJECTS = \
"CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o"

# External object files for target GIMPACTUtils
GIMPACTUtils_EXTERNAL_OBJECTS =

Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o
Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/build.make
Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: Extras/ConvexDecomposition/libConvexDecomposition.so.3.19
Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: src/BulletCollision/libBulletCollision.so.3.19
Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: src/LinearMath/libLinearMath.so.3.19
Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tboult/WORK/bullet3/build_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libGIMPACTUtils.so"
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GIMPACTUtils.dir/link.txt --verbose=$(VERBOSE)
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && $(CMAKE_COMMAND) -E cmake_symlink_library libGIMPACTUtils.so.3.19 libGIMPACTUtils.so.3.19 libGIMPACTUtils.so

Extras/GIMPACTUtils/libGIMPACTUtils.so: Extras/GIMPACTUtils/libGIMPACTUtils.so.3.19
	@$(CMAKE_COMMAND) -E touch_nocreate Extras/GIMPACTUtils/libGIMPACTUtils.so

# Rule to build all files generated by this target.
Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/build: Extras/GIMPACTUtils/libGIMPACTUtils.so

.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/build

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/requires: Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/btGImpactConvexDecompositionShape.o.requires

.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/requires

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/clean:
	cd /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils && $(CMAKE_COMMAND) -P CMakeFiles/GIMPACTUtils.dir/cmake_clean.cmake
.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/clean

Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/depend:
	cd /home/tboult/WORK/bullet3/build_cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tboult/WORK/bullet3 /home/tboult/WORK/bullet3/Extras/GIMPACTUtils /home/tboult/WORK/bullet3/build_cmake /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils /home/tboult/WORK/bullet3/build_cmake/Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Extras/GIMPACTUtils/CMakeFiles/GIMPACTUtils.dir/depend


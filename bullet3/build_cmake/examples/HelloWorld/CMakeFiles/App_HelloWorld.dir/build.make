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
include examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/depend.make

# Include the progress variables for this target.
include examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/progress.make

# Include the compile flags for this target's objects.
include examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/flags.make

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/flags.make
examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o: ../examples/HelloWorld/HelloWorld.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tboult/WORK/bullet3/build_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o"
	cd /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/App_HelloWorld.dir/HelloWorld.o -c /home/tboult/WORK/bullet3/examples/HelloWorld/HelloWorld.cpp

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/App_HelloWorld.dir/HelloWorld.i"
	cd /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tboult/WORK/bullet3/examples/HelloWorld/HelloWorld.cpp > CMakeFiles/App_HelloWorld.dir/HelloWorld.i

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/App_HelloWorld.dir/HelloWorld.s"
	cd /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tboult/WORK/bullet3/examples/HelloWorld/HelloWorld.cpp -o CMakeFiles/App_HelloWorld.dir/HelloWorld.s

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.requires:

.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.requires

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.provides: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.requires
	$(MAKE) -f examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/build.make examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.provides.build
.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.provides

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.provides.build: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o


# Object files for target App_HelloWorld
App_HelloWorld_OBJECTS = \
"CMakeFiles/App_HelloWorld.dir/HelloWorld.o"

# External object files for target App_HelloWorld
App_HelloWorld_EXTERNAL_OBJECTS =

examples/HelloWorld/App_HelloWorld: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o
examples/HelloWorld/App_HelloWorld: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/build.make
examples/HelloWorld/App_HelloWorld: src/BulletDynamics/libBulletDynamics.so.3.19
examples/HelloWorld/App_HelloWorld: src/BulletCollision/libBulletCollision.so.3.19
examples/HelloWorld/App_HelloWorld: src/LinearMath/libLinearMath.so.3.19
examples/HelloWorld/App_HelloWorld: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tboult/WORK/bullet3/build_cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable App_HelloWorld"
	cd /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/App_HelloWorld.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/build: examples/HelloWorld/App_HelloWorld

.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/build

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/requires: examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/HelloWorld.o.requires

.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/requires

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/clean:
	cd /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld && $(CMAKE_COMMAND) -P CMakeFiles/App_HelloWorld.dir/cmake_clean.cmake
.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/clean

examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/depend:
	cd /home/tboult/WORK/bullet3/build_cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tboult/WORK/bullet3 /home/tboult/WORK/bullet3/examples/HelloWorld /home/tboult/WORK/bullet3/build_cmake /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld /home/tboult/WORK/bullet3/build_cmake/examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/HelloWorld/CMakeFiles/App_HelloWorld.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rodrygojose/opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rodrygojose/opencv/release

# Include any dependencies generated for this target.
include modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/depend.make

# Include the progress variables for this target.
include modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/progress.make

# Include the compile flags for this target's objects.
include modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/flags.make

modules/legacy/opencv_legacy_pch_dephelp.cxx: ../modules/legacy/src/precomp.hpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrygojose/opencv/release/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating opencv_legacy_pch_dephelp.cxx"
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo \#include\ \"/home/rodrygojose/opencv/modules/legacy/src/precomp.hpp\" > /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo int\ testfunction\(\)\; >> /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo int\ testfunction\(\) >> /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo { >> /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo \ \ \ \ \return\ 0\; >> /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/cmake -E echo } >> /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/flags.make
modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o: modules/legacy/opencv_legacy_pch_dephelp.cxx
	$(CMAKE_COMMAND) -E cmake_progress_report /home/rodrygojose/opencv/release/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o"
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o -c /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.i"
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx > CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.i

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.s"
	cd /home/rodrygojose/opencv/release/modules/legacy && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/rodrygojose/opencv/release/modules/legacy/opencv_legacy_pch_dephelp.cxx -o CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.s

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.requires:
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.requires

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.provides: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.requires
	$(MAKE) -f modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/build.make modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.provides.build
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.provides

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.provides.build: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o

# Object files for target opencv_legacy_pch_dephelp
opencv_legacy_pch_dephelp_OBJECTS = \
"CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o"

# External object files for target opencv_legacy_pch_dephelp
opencv_legacy_pch_dephelp_EXTERNAL_OBJECTS =

lib/libopencv_legacy_pch_dephelp.a: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o
lib/libopencv_legacy_pch_dephelp.a: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/build.make
lib/libopencv_legacy_pch_dephelp.a: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../../lib/libopencv_legacy_pch_dephelp.a"
	cd /home/rodrygojose/opencv/release/modules/legacy && $(CMAKE_COMMAND) -P CMakeFiles/opencv_legacy_pch_dephelp.dir/cmake_clean_target.cmake
	cd /home/rodrygojose/opencv/release/modules/legacy && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_legacy_pch_dephelp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/build: lib/libopencv_legacy_pch_dephelp.a
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/build

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/requires: modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/opencv_legacy_pch_dephelp.cxx.o.requires
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/requires

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/clean:
	cd /home/rodrygojose/opencv/release/modules/legacy && $(CMAKE_COMMAND) -P CMakeFiles/opencv_legacy_pch_dephelp.dir/cmake_clean.cmake
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/clean

modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/depend: modules/legacy/opencv_legacy_pch_dephelp.cxx
	cd /home/rodrygojose/opencv/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rodrygojose/opencv /home/rodrygojose/opencv/modules/legacy /home/rodrygojose/opencv/release /home/rodrygojose/opencv/release/modules/legacy /home/rodrygojose/opencv/release/modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : modules/legacy/CMakeFiles/opencv_legacy_pch_dephelp.dir/depend


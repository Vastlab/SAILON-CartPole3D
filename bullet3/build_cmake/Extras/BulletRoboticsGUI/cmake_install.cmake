# Install script for directory: /home/tboult/WORK/bullet3/Extras/BulletRoboticsGUI

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libBulletRoboticsGUI.so.3.19"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libBulletRoboticsGUI.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/tboult/WORK/bullet3/build_cmake/Extras/BulletRoboticsGUI/libBulletRoboticsGUI.so.3.19"
    "/home/tboult/WORK/bullet3/build_cmake/Extras/BulletRoboticsGUI/libBulletRoboticsGUI.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libBulletRoboticsGUI.so.3.19"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libBulletRoboticsGUI.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/tboult/WORK/bullet3/build_cmake/examples/ExampleBrowser:/home/tboult/WORK/bullet3/build_cmake/Extras/BulletRobotics:/home/tboult/WORK/bullet3/build_cmake/Extras/InverseDynamics:/home/tboult/WORK/bullet3/build_cmake/Extras/Serialize/BulletWorldImporter:/home/tboult/WORK/bullet3/build_cmake/Extras/Serialize/BulletFileLoader:/home/tboult/WORK/bullet3/build_cmake/src/BulletSoftBody:/home/tboult/WORK/bullet3/build_cmake/src/BulletDynamics:/home/tboult/WORK/bullet3/build_cmake/src/BulletCollision:/home/tboult/WORK/bullet3/build_cmake/src/BulletInverseDynamics:/home/tboult/WORK/bullet3/build_cmake/src/LinearMath:/home/tboult/WORK/bullet3/build_cmake/examples/OpenGLWindow:/home/tboult/WORK/bullet3/build_cmake/examples/ThirdPartyLibs/Gwen:/home/tboult/WORK/bullet3/build_cmake/examples/ThirdPartyLibs/BussIK:/home/tboult/WORK/bullet3/build_cmake/src/Bullet3Common:"
           NEW_RPATH "")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/tboult/WORK/bullet3/build_cmake/Extras/BulletRoboticsGUI/bullet_robotics_gui.pc")
endif()


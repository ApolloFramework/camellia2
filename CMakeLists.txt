# Project information
project(Camellia)
set(VERSION_MAJOR "1")
set(VERSION_MINOR "0")
set(VERSION_PATCH "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

# Required version
cmake_minimum_required(VERSION 3.1)
cmake_policy(SET CMP0004 OLD)
# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

# Various ways of using scimake
if (EXISTS ${CMAKE_SOURCE_DIR}/scimake)
   set(SCIMAKE_DIR ${CMAKE_SOURCE_DIR}/scimake)
elseif (EXISTS ${CMAKE_SOURCE_DIR}/../scimake)
   set(SCIMAKE_DIR ${CMAKE_SOURCE_DIR}/../scimake)
else ()
   # This is useful for spack
   set(SCIMAKE_DIR $ENV{APOLLO_ROOT}/scimake)
endif()
include(${SCIMAKE_DIR}/SciInit.cmake)

#----------------------------------------------------------
#   Various properties
#----------------------------------------------------------
# Set TARGET_SUPPORTS_SHARED_LIBS -- otherwise problems building shared libs on Macs (at least)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

set(CMDEPS_STATE REQUIRED)
if (BASIC_DEPS_REQUIRED STREQUAL "False") # If set to false, change to empty
  message(STATUS "Unsetting REQUIRED in find_package calls for dependencies.")
  set(CMDEPS_STATE)
endif ()

set (CMAKE_CXX_STANDARD 11)

# Camellia comments this out
SET(BUILD_SHARED_LIBS OFF)
MESSAGE("Setting BUILD_SHARED_LIBS to ${BUILD_SHARED_LIBS}")
#IF(NOT BUILD_SHARED_LIBS)
#  MESSAGE("Setting BUILD_SHARED_LIBS to ${Trilinos_BUILD_SHARED_LIBS}")
#  SET(BUILD_SHARED_LIBS ${Trilinos_BUILD_SHARED_LIBS})
#ENDIF()

#----------------------------------------------------------
#   Dependencies
#----------------------------------------------------------
  if (ENABLE_PARALLEL)
   set(TRILINOS_BUILD trilinos-par)
  else ()
   set(TRILINOS_BUILD trilinos-ser)
  endif ()
  set(USE_TRILINOS_CONFIG_CMAKE 1)
  find_package(SciTrilinos REQUIRED) # Finds lapack and blas
  if ("${Trilinos_USE_VENDOR_LINALG}" STREQUAL "mkl")
    message(STATUS "Adding -mkl to cxx flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=cluster")
    set(Trilinos_LINALG_LIBRARIES)
  endif()

  if (ENABLE_PARALLEL)
    set(MOAB_COMPONENTS MOAB iMesh)
  else ()
    set(MOAB_COMPONENTS MOAB iMesh)
  endif ()
  find_package(SciMoab REQUIRED)
  ADD_DEFINITIONS(-DHAVE_MOAB)
  #find_package(SciNetcdf REQUIRED)
  
#endif ()

# Set TARGET_SUPPORTS_SHARED_LIBS -- otherwise problems building shared libs on Macs (at least)
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

option(INCLUDE_DRIVERS_IN_ALL "Include drivers in make all (set to ON for IDE project generation)" OFF)
option(BUILD_FOR_INSTALL "Build for installation (including examples and tests)" OFF)

IF(INCLUDE_DRIVERS_IN_ALL)
  SET(EXCLUDE_DRIVERS_FROM_ALL "")
ELSE()
  SET(EXCLUDE_DRIVERS_FROM_ALL "EXCLUDE_FROM_ALL")
ENDIF(INCLUDE_DRIVERS_IN_ALL)

# Set Camellia directories
set(CAMELLIA_SOURCE_DIR src)
set(DRIVER_DIR drivers)

if (EXTRA_INCLUDE)
  include_directories(${EXTRA_INCLUDE})
endif()

if (OMP_INCLUDE)
  include_directories(${OMP_INCLUDE})
endif()

if(SCALAPACK_LIB)
  link_libraries(${SCALAPACK_LIB})
endif()

# Camellia comments this out
SET(BUILD_SHARED_LIBS OFF)
MESSAGE("Setting BUILD_SHARED_LIBS to ${BUILD_SHARED_LIBS}")
#IF(NOT BUILD_SHARED_LIBS)
#  MESSAGE("Setting BUILD_SHARED_LIBS to ${Trilinos_BUILD_SHARED_LIBS}")
#  SET(BUILD_SHARED_LIBS ${Trilinos_BUILD_SHARED_LIBS})
#ENDIF()

# Find all library source files
FILE(GLOB_RECURSE LIB_SOURCES "${CAMELLIA_SOURCE_DIR}/*.cpp" "${CAMELLIA_SOURCE_DIR}/include/*.h")
set(HEADERS 
  ${CAMELLIA_SOURCE_DIR}/include
)
if (ENABLE_PARALLEL)
 set(TRILINOS_BUILD trilinos-par)
else ()
 set(TRILINOS_BUILD trilinos-ser)
endif ()
set(USE_TRILINOS_CONFIG_CMAKE 1)
find_package(SciTrilinos REQUIRED) # Finds lapack and blas
if ("${Trilinos_USE_VENDOR_LINALG}" STREQUAL "mkl")
  message(STATUS "Adding -mkl to cxx flags")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=cluster")
  set(Trilinos_LINALG_LIBRARIES)
endif()

if (ENABLE_PARALLEL)
  set(MOAB_COMPONENTS MOAB iMesh)
else ()
  set(MOAB_COMPONENTS MOAB iMesh)
endif ()
find_package(SciMoab REQUIRED)
ADD_DEFINITIONS(-DHAVE_MOAB)
>>>>>>> Adding cmake config-file functionality for downstream convenience

#Clean up the Trilinos Libraries by removing duplicates
#This is need to avoid CMAKE linking errors
#due to excessively long link lists with Spack

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)

list(REVERSE Trilinos_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_LIBRARIES)
list(REVERSE Trilinos_LIBRARIES)

list(REVERSE Trilinos_TPL_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)
list(REVERSE Trilinos_TPL_INCLUDE_DIRS)

#----------------------------------------------------------
# Includes
#----------------------------------------------------------
include_directories(
  src/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SCIMAKE_DIR}/include
)

if (EXTRA_INCLUDE)
  include_directories(${EXTRA_INCLUDE})
endif()

if (OMP_INCLUDE)
  include_directories(${OMP_INCLUDE})
endif()

set(Camellia_TPL_INCLUDE_DIRS
  ${Trilinos_TPL_INCLUDE_DIRS}
  ${Trilinos_INCLUDE_DIRS}
  ${Moab_INCLUDE_DIRS}
  ${Mumps_INCLUDE_DIRS}
)
include_directories(${Camellia_TPL_INCLUDE_DIRS})

#----------------------------------------------------------
# Library
#----------------------------------------------------------
# Set Camellia directories
set(CAMELLIA_SOURCE_DIR src)
set(DRIVER_DIR drivers)

# Find all library source files
FILE(GLOB_RECURSE LIB_SOURCES "${CAMELLIA_SOURCE_DIR}/*.cpp" "${CAMELLIA_SOURCE_DIR}/include/*.h")
set(HEADERS 
  ${CAMELLIA_SOURCE_DIR}/include
)

add_library(Camellia ${LIB_SOURCES} ${HEADERS})

# Most dependencies handled by trilinos dependencies
target_link_libraries(
 Camellia
  ${Moab_LIBRARIES}
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  ${Trilinos_LINALG_LIBRARIES}
  ${ADDITIONAL_LIBRARIES}
  ${Fortran_IMPLICIT_LIBRARIES}
)

FILE(GLOB ALL_HEADER_FILES "${HEADERS}/*.h" "${HEADERS}/*.hpp")
install (FILES ${ALL_HEADER_FILES} DESTINATION include/Camellia)

install (TARGETS Camellia 
         EXPORT camellia-lib
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib
         RUNTIME DESTINATION bin
         INCLUDES DESTINATION include
         )

#----------------------------------------------------------
# Drivers, examples, etc.
#----------------------------------------------------------
# Enable testing (see unit_tests directory)
include(CTest)
enable_testing()

option(INCLUDE_DRIVERS_IN_ALL "Include drivers in make all (set to ON for IDE project generation)" OFF)
option(BUILD_FOR_INSTALL "Build for installation (including examples and tests)" ON)

IF(INCLUDE_DRIVERS_IN_ALL)
  SET(EXCLUDE_DRIVERS_FROM_ALL "")
ELSE()
  SET(EXCLUDE_DRIVERS_FROM_ALL "EXCLUDE_FROM_ALL")
ENDIF(INCLUDE_DRIVERS_IN_ALL)

# Build Drivers

# Build for install (as spack install does) should include examples, but exclude drivers
IF(BUILD_FOR_INSTALL)
  SET(EXCLUDE_EXAMPLES_FROM_ALL "")
ELSE()
  SET(EXCLUDE_EXAMPLES_FROM_ALL ${EXCLUDE_DRIVERS_FROM_ALL})
ENDIF(BUILD_FOR_INSTALL)

add_subdirectory(drivers ${EXCLUDE_DRIVERS_FROM_ALL})
add_subdirectory(examples ${EXCLUDE_EXAMPLES_FROM_ALL})
add_subdirectory(manual-examples ${EXCLUDE_EXAMPLES_FROM_ALL})
add_subdirectory(unit_tests)
#add_subdirectory(slow_tests)

#----------------------------------------------------------
#  Documentation
#----------------------------------------------------------
# Find/Setup Doxygen
option(ENABLE_DOCS "Build the documents with doxygen or not" OFF)
IF(ENABLE_DOCS)
  find_package(SciDoxygen)
  if (DOXYGEN_FOUND)
     message("DOXYGEN_EXECUTABLE found.")
     message("DOXYGEN_EXECUTABLE = ${DOXYGEN_EXECUTABLE}")
  endif ()
  # Put docs into build dir
  set(DOXYGEN_DOCSDIR ${CMAKE_BINARY_DIR}/docs)

  configure_file(camellia.dox.in camellia.dox)

  # Convenience
  add_custom_target(docs)
  add_dependencies(docs apidocs)
  add_custom_target(install-docs)
  add_dependencies(install-docs install-apidocs)

  # Doxygen documentation
  ##

  add_custom_target(apidocs
          command ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/camellia.dox)
  add_custom_target(install-apidocs COMMAND $(CMAKE_COMMAND) -P ${BLDDIR}/cmake_install.cmake)
  add_dependencies(install-apidocs apidocs)
ENDIF()


#----------------------------------------------------------
# Finish up -- include cmake files to make downstream packages
# using CMake configure easily
#----------------------------------------------------------
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR include)
set(INCLUDE_INSTALL_DIR include)
set(Camellia_INCLUDE_DIRS ${INCLUDE_INSTALL_DIR})

configure_package_config_file(CamelliaConfig.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/CamelliaConfig.cmake
      INSTALL_DESTINATION lib/Camellia/cmake
      PATH_VARS INCLUDE_INSTALL_DIR)
write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/CamelliaVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMajorVersion )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CamelliaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/CamelliaVersion.cmake
              DESTINATION lib/Camellia/cmake )
install (EXPORT camellia-lib 
         DESTINATION lib/Camellia/cmake 
         FILE "CamelliaTargets.cmake" )

set(INSTALL_CONFIG_HEADERS FALSE)
include(${SCIMAKE_DIR}/SciFinalize.cmake)

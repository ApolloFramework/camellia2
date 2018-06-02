#!/bin/bash

set -e

# A Camellia do-config script that uses Spack-built dependencies

export INSTALL_PREFIX=${HOME}/lib/Camellia

COMPILER=clang@9.0.0-apple # or gcc
BUILD_TYPE=Release

CAMELLIA_PATH=../..

SPACK_EXE=${HOME}/spack/bin/spack
MPI_DIR=$(${SPACK_EXE} location -i openmpi %${COMPILER})
TRILINOS_DIR=$(${SPACK_EXE} location -i trilinos %${COMPILER} build_type=${BUILD_TYPE})
MOAB_DIR=$(${SPACK_EXE} location -i moab %${COMPILER})

export PATH=$(${SPACK_EXE} location -i cmake %${COMPILER})/bin:${PATH}
export PATH=$(${SPACK_EXE} location -i openmpi %${COMPILER})/bin:${PATH}

SPACK_EXE=${HOME}/spack/bin/spack

cmake \
  -D TRILINOS_PATH:PATH=${TRILINOS_DIR} \
  -D CAMELLIA_SOURCE_DIR:PATH=${CAMELLIA_PATH}/src \
  -D CMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -D MPI_DIR:PATH=${MPI_DIR} \
  -D ENABLE_MOAB:BOOL=ON \
  -D MOAB_PATH:PATH=${MOAB_DIR} \
  -D BUILD_FOR_INSTALL:BOOL=ON \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX} \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH="ON" \
  ${CAMELLIA_PATH}

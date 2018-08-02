Camellia: A Software Toolbox for Discontinuous Petrov-Galerkin (DPG) Methods
-----------------------------------------------------------------------------
(with support for other formulations)

by Nathan V. Roberts

We have recently added support for building Camellia using Spack, a cross-platform package manager designed for HPC libraries.  This is likely the easiest way to build Camellia.  We provide instructions for building using Spack below.  We also include some instructions for manual builds, which provide more control.

Details about using Camellia can be found in the v1.0 Manual, available at

	http://www.ipd.anl.gov/anlpubs/2016/11/130782.pdf

******************************** Instructions for Spack Builds *********************************

Spack has the following prerequisites:
- Python 2 (2.6 or 2.7) or 3 (3.3 - 3.6)
- A C/C++ compiler
- The git and curl commands

To install Spack, do the following:

	git clone https://github.com/spack/spack.git

For bash or zsh, in your .bashrc or .zshrc, add the following:

	export SPACK_ROOT=/path/to/spack
	. $SPACK_ROOT/share/spack/setup-env.sh

For csh or tcsh, add the following to your .cshrc:

	setenv SPACK_ROOT /path/to/spack
	source $SPACK_ROOT/share/spack/setup-env.csh

Start a new shell session, or execute the appropriate lines above manually (this places the spack executable in your path, among other things).  Then, installing Camellia should be as simple as running:

	spack install camellia

(If you want to maintain your own build of Camellia (as you might if you are a Camellia developer), then instead do:

	spack install camellia --only dependencies

If you want to maintain your own build, then the next step is to edit the CMake do-config.sh script found in build/spack-based in the Camellia distribution.)

This will take some time, as it downloads and builds all the dependencies.  If you did a full install, the install location can be found by executing:

	spack location -i camellia

To confirm that the build was successful, run:

	`spack location -i camellia`/bin/tests/runTests

All tests should pass.

Further details about Spack can be found at

	https://spack.readthedocs.io/en/latest/getting_started.html

******************************** Instructions for Manual Builds *********************************

******** PREREQUISITES ********
Trilinos is required for all builds of Camellia.  A couple of sample do-configure scripts for Trilinos can be found in the Camellia distribution directory, under build/trilinos-do-configure-samples.  These include the packages within Trilinos that Camellia requires.

Building Trilinos (specifically Epetra) with HDF5 is not absolutely required, but without this nearly all Camellia's data I/O facilities (mesh and solution serialization, as well as visualization output) will fail with an exception.  This will also result in test failures for tests related to mesh and solution I/O.  Note that there is an incompatibility with HDF5 version 1.10; please use 1.8.x instead.

For an MPI build, Camellia also requires some version of the MPI libraries.  Open MPI is what we use most of the time.  Additionally, Camellia supports MUMPS and SuperLU_Dist if both Camellia and Trilinos are built with these libraries.  MUMPS also requires SCALAPACK to be installed.

Instructions for building several of these libraries follow.

CMake install:
On a Mac, our experience is that due to Apple’s requirements for code signatures it is simpler to install CMake from source than to use the prebuilt binary.  Homebrew works well for this.

OpenMPI install:
1. Download source from http://www.open-mpi.org/software/ompi/.
2. cd into source dir.
3. Configure (editing the prefix line according to where you'd like it installed):
	./configure --prefix=$HOME/lib/openmpi-1.8.3 CC=cc CXX=c++ FC=gfortran
4. Build:
	make -j6
5. Install:
	make install
6. Add the bin folder to your PATH, e.g. by adding to your .bashrc:
	export PATH=${PATH}:${HOME}/lib/openmpi-1.8.3/bin

HDF5 install (parallel build, not suitable for serial builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=mpicc ./configure --prefix=/Users/nroberts/local/hdf5
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/local/hdf5/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/local/hdf5/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

HDF5 install (serial build, not suitable for parallel builds of Trilinos):
1. Download source for hdf5-1.8.X from http://www.hdfgroup.org/HDF5/release/obtainsrc.html#conf.
2. Untar.
3. Configure:
   CC=clang ./configure --prefix=/Users/nroberts/local/hdf5-serial
4. Make and install:
   make -j6
   make install
5. In the Trilinos do-configure, you'll want to include lines like the following:
   -D TPL_ENABLE_HDF5:STRING=ON \
   -D HDF5_LIBRARY_DIRS:FILEPATH=/Users/nroberts/local/hdf5-serial/lib \
   -D HDF5_LIBRARY_NAMES:STRING="hdf5" \
   -D TPL_HDF5_INCLUDE_DIRS:FILEPATH=/Users/nroberts/local/hdf5-serial/include \
   -D EpetraExt_USING_HDF5:BOOL=ON \

Note that HDF5 1.10.x appears to be incompatible with the EpetraExt class that Camellia relies on for HDF5 output.  Therefore, 1.8.x is strongly recommended.

Scalapack install:
1. Download source from http://www.netlib.org/scalapack/
2. cd into source dir.
3. Configure:
	ccmake .
	(specify ~/lib/openmpi-1.8.3 as the MPI_BASE_DIR, and ~ as the CMAKE_INSTALL_PREFIX; the other values should be autofilled on configure.)
4. Build
	cmake .
	make -j6
5. Install
	make install

MUMPS install:
1. Download source from http://graal.ens-lyon.fr/MUMPS/
2. Copy <MUMPS dir>/Make.inc/Makefile.INTEL.PAR to <MUMPS dir>/Makefile.inc
3. Edit the following lines in Makefile.inc (some of these are specific to building on a Mac):
	FC = gfortran
	FL = gfortran
	SCALAP  = <home dir>/lib/libscalapack.a
	INCPAR = -I<home dir>/lib/openmpi-1.8.3/include
	LIBPAR = $(SCALAP)  -L<home dir>/lib/openmpi-1.8.3/lib -lmpi -lmpi_f77
	LIBBLAS = -framework vecLib # BLAS and LAPACK libraries for Mac
	OPTF    = -O3 -Dintel_ -DALLOW_NON_INIT 
	OPTL    = -O3
	OPTC    = -O3
4. make -j6
5. Copy the built libraries from <MUMPS dir>/lib to $HOME/lib/mumps-4.10.0.
6. Copy the include directory to $HOME/lib/mumps-4.10.0/include.

******** BUILDING CAMELLIA **********

Once that's done, you're ready to start on the Camellia build.

Instructions for a serial debug build:
1. Clone from repo.
	git clone https://bitbucket.org/nateroberts/camellia.git
2. Go to the serial-debug build directory:
	cd build/serial-debug
3. Edit do-configure-serial-debug in the following manner:
       - set the TRILINOS_PATH to your serial-debug Trilinos installation
       - set ZLIB_LIB to the path to the zlib library (for HDF5 support)
       - set CMAKE_INSTALL_PREFIX:PATH to your preferred install location for Camellia
4. Run the do-configure script:
	./do-configure-serial-debug
5. make
6. make test
7. make install

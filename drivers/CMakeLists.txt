#project(Drivers)

option(BUILD_DPGTESTS_DRIVER "Build DPGTests driver" OFF)
option(BUILD_BRENDAN_DRIVERS "Build drivers in Brendan directory" OFF)
option(BUILD_PRECONDITIONING_DRIVERS "Build drivers in Preconditioning directory" OFF)
option(BUILD_CONVECTION_DRIVERS "Build drivers in Convection directory" OFF)

# Include headers from DPGTests for some drivers
include_directories(DPGTests)

# Add each driver
#add_subdirectory(Burgers)
if (BUILD_BRENDAN_DRIVERS)
  add_subdirectory(Brendan)
  MESSAGE("Setting up makefiles for drivers in drivers/Brendan, because BUILD_BRENDAN_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Brendan, because BUILD_BRENDAN_DRIVERS is OFF.")  
endif(BUILD_BRENDAN_DRIVERS)

if (BUILD_CONVECTION_DRIVERS)
  add_subdirectory(Convection)
  MESSAGE("Setting up makefiles for drivers in drivers/Convection, because BUILD_CONVECTION_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Convection, because BUILD_CONVECTION_DRIVERS is OFF.")  
endif(BUILD_CONVECTION_DRIVERS)

if (BUILD_DPGTESTS_DRIVER)
  add_subdirectory(DPGTests)
  MESSAGE("Setting up makefile for legacy DPGTests driver in drivers/DPGTests, because BUILD_DPGTESTS_DRIVER is ON.")
else()
  MESSAGE("Not setting up makefile for legacy DPGTests driver in drivers/DPGTests, because BUILD_DPGTESTS_DRIVER is OFF.")  
endif(BUILD_DPGTESTS_DRIVER)

if (BUILD_PRECONDITIONING_DRIVERS)
  add_subdirectory(Preconditioning)
  MESSAGE("Setting up makefiles for drivers in drivers/Preconditioning, because BUILD_PRECONDITIONING_DRIVERS is ON.")
else()
  MESSAGE("Not setting up makefiles for drivers in drivers/Preconditioning, because BUILD_PRECONDITIONING_DRIVERS is OFF.")  
endif(BUILD_PRECONDITIONING_DRIVERS)

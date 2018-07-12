#project(DPGTests)

FILE(GLOB TEST_SOURCES "*.cpp")
set( DRIVERS_DIR ".." )

#include_directories("${DRIVERS_DIR}/Stokes") # for certain tests in DPGTests

add_executable(DPGTests ${TEST_SOURCES} ${CONFUSION_SOURCES} ${POISSON_SOURCES})
target_link_libraries(DPGTests Camellia)

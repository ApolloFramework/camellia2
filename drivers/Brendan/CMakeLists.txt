project(BDrivers)

add_executable(LinearElasticity2D LinearElasticity2D.cpp)
add_executable(LinearElasticity3D LinearElasticity3D.cpp)
add_executable(LinearElasticity3DGMG LinearElasticity3DGMG.cpp)
add_executable(OldroydBDriver OldroydBDriver.cpp)

target_link_libraries(OldroydBDriver ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia)
target_link_libraries(LinearElasticity2D ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia)
target_link_libraries(LinearElasticity3D ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia)
target_link_libraries(LinearElasticity3DGMG ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} Camellia)
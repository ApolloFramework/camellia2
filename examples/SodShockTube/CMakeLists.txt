project(SodShockTubeExamples)

set( SOD_SHOCK_TUBE_EXAMPLE "SodShockTubeExample.cpp" )

add_executable(SodShockTubeExample ${SOD_SHOCK_TUBE_EXAMPLE})
target_link_libraries(SodShockTubeExample
  Camellia
)


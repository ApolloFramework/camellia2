project(BrioWuShockTubeExamples)

set( BRIO_WU_SHOCK_TUBE_EXAMPLE "BrioWuShockTubeExample.cpp" )

add_executable(BrioWuShockTubeExample ${BRIO_WU_SHOCK_TUBE_EXAMPLE})
target_link_libraries(BrioWuShockTubeExample
  Camellia
)

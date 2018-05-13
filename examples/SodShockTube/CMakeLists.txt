project(SodShockTubeExamples)

set( SOD_SHOCK_TUBE_EXAMPLE "SodShockTubeExample.cpp" )

add_executable(SodShockTubeExample ${SOD_SHOCK_TUBE_EXAMPLE})
target_link_libraries(SodShockTubeExample
  Camellia
)

set( STANDING_SHOCK_EXAMPLE "StandingShockExample.cpp" )
add_executable(StandingShockExample ${STANDING_SHOCK_EXAMPLE})
target_link_libraries(StandingShockExample
  Camellia
)

set( STANDING_SHOCK_EXAMPLE_CONSERVATION_FORM "StandingShockConservationFormExample.cpp" )
add_executable(StandingShockConservationFormExample ${STANDING_SHOCK_EXAMPLE_CONSERVATION_FORM})
target_link_libraries(StandingShockConservationFormExample
  Camellia
)


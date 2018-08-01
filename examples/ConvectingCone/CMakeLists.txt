project(ConvectingConeExample)

add_executable(ConvectingConeExample ConvectingConeExample.cpp)
target_link_libraries(ConvectingConeExample Camellia)

install (TARGETS ConvectingConeExample DESTINATION bin/examples/)

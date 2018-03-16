project(DGAdvectionExample)

add_executable(CGAdvectionExplicitExample CGAdvectionExplicitExample.cpp)

target_link_libraries(CGAdvectionExplicitExample
  Camellia
)

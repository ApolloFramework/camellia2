project(Poisson2DTriangles)

add_executable(Poisson2DTriangles "Poisson2DTriangles.cpp")
target_link_libraries(Poisson2DTriangles 
  Camellia
)


project(ch05-Formulations)

add_executable(ch05-NavierStokesCavity ch05-NavierStokesCavity.cpp)
target_link_libraries(ch05-NavierStokesCavity Camellia)

add_executable(ch05-PoissonHomogeneous ch05-PoissonHomogeneous.cpp)
target_link_libraries(ch05-PoissonHomogeneous Camellia)

add_executable(ch05-StokesCavity ch05-StokesCavity.cpp)
target_link_libraries(ch05-StokesCavity Camellia)

install (TARGETS ch05-NavierStokesCavity DESTINATION bin/manual-examples)
install (TARGETS ch05-PoissonHomogeneous DESTINATION bin/manual-examples)
install (TARGETS ch05-StokesCavity DESTINATION bin/manual-examples)

project(ch04-NavierStokesCavity)

add_executable(ch04-NavierStokesCavity ch04-NavierStokesCavity.cpp)
target_link_libraries(ch04-NavierStokesCavity Camellia)

install (TARGETS ch04-NavierStokesCavity DESTINATION bin/manual-examples)

project(ch03-StokesCavity)

add_executable(ch03-StokesCavity ch03-StokesCavity.cpp)
target_link_libraries(ch03-StokesCavity Camellia)

install (TARGETS ch03-StokesCavity DESTINATION bin/manual-examples)

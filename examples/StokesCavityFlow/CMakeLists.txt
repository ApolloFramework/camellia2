project(StokesCavityFlowExamples)

add_executable(StokesGramOutput StokesGramOutput.cpp)
target_link_libraries(StokesGramOutput 
  Camellia
)

add_executable(StokesCavityFlowExample StokesCavityFlowExample.cpp)
target_link_libraries(StokesCavityFlowExample 
  Camellia
)

add_executable(StokesCavityFlowSpaceTimeExample StokesCavityFlowSpaceTimeExample.cpp)
target_link_libraries(StokesCavityFlowSpaceTimeExample
  Camellia
)


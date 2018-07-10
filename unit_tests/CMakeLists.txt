#project(UnitTests)

FILE(GLOB TEST_SOURCES "*.cpp")

add_executable(runTests ${TEST_SOURCES})

target_link_libraries(runTests Camellia)

add_test(NAME runTests COMMAND runTests)

install (TARGETS runTests DESTINATION bin/tests)

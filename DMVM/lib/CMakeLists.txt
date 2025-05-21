set(LIB_NAME dmvm_lib)

file(GLOB SRCFILES src/*.c)
file(GLOB HFILES include/*.h)

add_library(${LIB_NAME} ${SRCFILES})
target_include_directories(${LIB_NAME}
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PUBLIC MPI::MPI_C)
if(ENABLE_CHECK)
  target_compile_definitions(${LIB_NAME} PRIVATE CHECK)
  message(STATUS "Enabled : Checking code")
else()
  message(STATUS "Disabled : Checking code")
endif()

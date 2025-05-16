set(LIB_NAME dmvm_lib)

file(GLOB SRCFILES src/*.c)
file(GLOB HFILES include/*.h)

add_library(${LIB_NAME} ${SRCFILES})
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(ENABLE_CHECK)
     target_compile_definitions( ${LIB_NAME} PRIVATE CHECK)
     message(VERBOSE "Checking code enabled")
endif()
add_executable(dmvm main.c)
target_link_libraries(dmvm PUBLIC dmvm_lib)
target_compile_definitions(dmvm PUBLIC ARRAY_ALIGNMENT=64)
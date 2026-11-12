idf_component_register(SRCS "main.cpp" INCLUDE_DIRS ".." REQUIRES framework-arduinoespressif32 bt app_update)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3 -std=gnu++17 -ffp-contract=off -Wno-error=misleading-indentation)

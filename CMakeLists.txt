# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion RISCV_ZCB_RPI_2_2_0_2)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)

project(rick_v2_pico C CXX ASM)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Initialize the Pico SDK
pico_sdk_init()

# Tell CMake we are writing C++
add_executable(rick_v2_pico 
    main.cpp
)

# Enable USB serial output (for debugging) and disable UART
pico_enable_stdio_usb(rick_v2_pico 1)
pico_enable_stdio_uart(rick_v2_pico 0)

# Link the required hardware libraries
target_link_libraries(rick_v2_pico 
    pico_stdlib
    hardware_spi
    hardware_timer
    hardware_pwm
)

# Create the .uf2 file for dragging and dropping onto the Pico
pico_add_extra_outputs(rick_v2_pico)
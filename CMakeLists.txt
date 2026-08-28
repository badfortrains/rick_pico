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
set(PICO_BOARD pico2_w CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)

project(rick_v2_pico C CXX ASM)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    pico_btstack_ble
    pico_btstack_cyw43
    pico_cyw43_arch_none
    pico_sha256
    boot_uf2_headers
    hardware_spi
    hardware_timer
    hardware_pwm
)

target_include_directories(rick_v2_pico PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

pico_btstack_make_gatt_header(
    rick_v2_pico
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/rick_control.gatt"
)

# Embed the RP2350 A/B layout in the initial wired image. Each application
# partition is 2040 KiB. The final 16 KiB remain unpartitioned so BTstack's
# two-sector bond store and the RP2350-E10 reserved sector stay untouched.
pico_embed_pt_in_binary(
    rick_v2_pico
    "${CMAKE_CURRENT_LIST_DIR}/ota_partition.json"
)

# Hash every image so the RP2350 boot ROM verifies it before launch. TBYB
# keeps the previous partition bootable until the new firmware initializes
# its hardware and Bluetooth stack and explicitly accepts the update.
target_compile_definitions(rick_v2_pico PRIVATE
    PICO_CRT0_IMAGE_TYPE_TBYB=1
)
pico_hash_binary(rick_v2_pico)

# Create the .uf2 file for the initial wired flash and later BLE updates.
pico_add_extra_outputs(rick_v2_pico)

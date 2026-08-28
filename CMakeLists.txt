# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.3.0)
set(toolchainVersion RISCV_ZCB_RPI_2_2_0_2)
set(picotoolVersion 2.3.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2_w CACHE STRING "Board type")

cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)

project(rick_v2_pico C CXX ASM)

option(
    RICK_BOOTSTRAP_IMAGE
    "Build the first application without TBYB when both A/B slots are empty"
    OFF
)

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

# Keep the partition table out of the replaceable application image. The
# RP2350 must install it separately, reboot to activate it, and only then load
# an application into one of the A/B slots. Generate that one-time wired UF2.
pico_init_picotool()
if (picotool_FOUND)
    set(OTA_PARTITION_UF2 "${CMAKE_CURRENT_BINARY_DIR}/ota_partition.uf2")
    add_custom_command(
        OUTPUT "${OTA_PARTITION_UF2}"
        COMMAND picotool partition create --quiet
            "${CMAKE_CURRENT_LIST_DIR}/ota_partition.json"
            "${OTA_PARTITION_UF2}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ota_partition.json"
        VERBATIM
    )
    add_custom_target(rick_v2_pico_partition ALL
        DEPENDS "${OTA_PARTITION_UF2}"
    )
endif()

# Hash every image so the RP2350 boot ROM verifies it before launch. TBYB
# keeps the previous partition bootable until the new firmware initializes
# its hardware and Bluetooth stack and explicitly accepts the update.
if (RICK_BOOTSTRAP_IMAGE)
    target_compile_definitions(rick_v2_pico PRIVATE
        PICO_CRT0_IMAGE_TYPE_TBYB=0
    )
else()
    target_compile_definitions(rick_v2_pico PRIVATE
        PICO_CRT0_IMAGE_TYPE_TBYB=1
    )
endif()
pico_hash_binary(rick_v2_pico)

# Create the application-only .uf2 used for wired and later BLE updates.
pico_add_extra_outputs(rick_v2_pico)

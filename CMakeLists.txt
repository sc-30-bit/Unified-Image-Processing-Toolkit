cmake_minimum_required(VERSION 3.20)

project(Unified_IP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(Unified_IP
    main.cpp
    basemodel.cpp
    yolo.cpp
    restoration.cpp
    colorizer.cpp
)

target_include_directories(Unified_IP PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio dnn)
if(OpenCV_FOUND)
    target_link_libraries(Unified_IP PRIVATE ${OpenCV_LIBS})
else()
    if(NOT DEFINED OPENCV_ROOT OR OPENCV_ROOT STREQUAL "")
        set(OPENCV_ROOT "$ENV{OPENCV_ROOT}")
    endif()
    if(OPENCV_ROOT STREQUAL "")
        set(OPENCV_ROOT "H:/libraries/Opencv_MSVC/opencv/build")
    endif()

    find_path(OPENCV_INCLUDE_DIR
        NAMES opencv2/opencv.hpp
        HINTS "${OPENCV_ROOT}/include"
    )

    find_library(OPENCV_WORLD_RELEASE
        NAMES opencv_world4120
        HINTS "${OPENCV_ROOT}/x64/vc16/lib"
    )
    find_library(OPENCV_WORLD_DEBUG
        NAMES opencv_world4120d
        HINTS "${OPENCV_ROOT}/x64/vc16/lib"
    )

    if(NOT OPENCV_INCLUDE_DIR)
        message(FATAL_ERROR
            "OpenCV include not found. Set OpenCV_DIR or OPENCV_ROOT. Current OPENCV_ROOT=${OPENCV_ROOT}"
        )
    endif()

    if(NOT OPENCV_WORLD_RELEASE AND NOT OPENCV_WORLD_DEBUG)
        message(FATAL_ERROR
            "OpenCV library not found. Set OpenCV_DIR or OPENCV_ROOT. Current OPENCV_ROOT=${OPENCV_ROOT}"
        )
    endif()

    target_include_directories(Unified_IP PRIVATE ${OPENCV_INCLUDE_DIR})

    if(OPENCV_WORLD_RELEASE AND OPENCV_WORLD_DEBUG)
        target_link_libraries(Unified_IP PRIVATE
            $<$<CONFIG:Debug>:${OPENCV_WORLD_DEBUG}>
            $<$<NOT:$<CONFIG:Debug>>:${OPENCV_WORLD_RELEASE}>
        )
    elseif(OPENCV_WORLD_RELEASE)
        target_link_libraries(Unified_IP PRIVATE ${OPENCV_WORLD_RELEASE})
    else()
        target_link_libraries(Unified_IP PRIVATE ${OPENCV_WORLD_DEBUG})
    endif()
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(ONNXRUNTIME_ROOT STREQUAL "")
    set(ONNXRUNTIME_ROOT "H:/libraries/onnxruntime-win-x64-gpu-1.17.0/onnxruntime-win-x64-gpu-1.17.0")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNXRuntime not found. Set ONNXRUNTIME_ROOT to your onnxruntime install directory "
        "(it should contain include/ and lib/). Current ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
    )
endif()

target_include_directories(Unified_IP PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(Unified_IP PRIVATE ${ONNXRUNTIME_LIBRARY})

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Unified_IP PRIVATE OpenMP::OpenMP_CXX)
endif()

if(WIN32)
    target_link_libraries(Unified_IP PRIVATE ws2_32 crypt32)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/index.html")
        add_custom_command(TARGET Unified_IP POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/index.html"
                "$<TARGET_FILE_DIR:Unified_IP>/index.html"
        )
    endif()

    foreach(_dll
        onnxruntime.dll
        onnxruntime_providers_shared.dll
        onnxruntime_providers_cuda.dll
    )
        if(EXISTS "${ONNXRUNTIME_ROOT}/lib/${_dll}")
            add_custom_command(TARGET Unified_IP POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_ROOT}/lib/${_dll}"
                    "$<TARGET_FILE_DIR:Unified_IP>/${_dll}"
            )
        endif()
    endforeach()
endif()

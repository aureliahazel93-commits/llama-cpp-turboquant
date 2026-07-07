option(GGML_HYBRID "ggml: use hybrid CPU+iGPU backend" OFF)

if(GGML_HYBRID)
    message(STATUS "Hybrid CPU+iGPU backend enabled")

    set(HYBRID_SOURCES
        ggml-hybrid.cpp
        uma/uma-buffer.cpp
        compute/mul-mat-dispatch.cpp
        scheduler/thermal-balancer.cpp
    )

    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Hybrid backend: Vulkan iGPU compute enabled")
        list(APPEND HYBRID_SOURCES compute/dequant-vulkan.cpp)
        list(APPEND HYBRID_DEFS GGML_HYBRID_VULKAN=1)

        file(GLOB HYBRID_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
        foreach(shader ${HYBRID_SHADERS})
            get_filename_component(shader_name ${shader} NAME_WE)
            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv
                COMMAND ${Vulkan_GLSLANG_VALIDATOR} -V -o ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv ${shader}
                DEPENDS ${shader}
                COMMENT "Compiling shader ${shader_name}.comp"
            )
        endforeach()
    else()
        message(STATUS "Hybrid backend: Vulkan not found, CPU-only mode")
    endif()

    add_library(ggml-hybrid OBJECT ${HYBRID_SOURCES})
    target_link_libraries(ggml-hybrid PRIVATE ggml-base)
    target_include_directories(ggml-hybrid PRIVATE .. ${CMAKE_CURRENT_SOURCE_DIR}/..)
    if(HYBRID_DEFS)
        target_compile_definitions(ggml-hybrid PRIVATE ${HYBRID_DEFS})
    endif()
    if(Vulkan_FOUND)
        target_link_libraries(ggml-hybrid PUBLIC Vulkan::Vulkan)
        target_include_directories(ggml-hybrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endif()

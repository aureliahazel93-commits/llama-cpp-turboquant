# Auto-detect GPU backends using llama_prober.py
# Included from top-level CMakeLists.txt
# Override: -DGGML_AUTO_PROBE=OFF to disable entirely

option(GGML_AUTO_PROBE "Auto-detect GPU backends via llama_prober.py" ON)

if(GGML_AUTO_PROBE)
    # Check if user explicitly set any backend
    set(_explicit_backend FALSE)
    foreach(_backend CUDA METAL HIP HIPBLAS SYCL VULKAN BLAS MUSA CANN OPENVINO OPENCL)
        if(DEFINED GGML_${_backend})
            set(_explicit_backend TRUE)
            message(STATUS "GPU auto-probe: skipped (GGML_${_backend} explicitly set)")
            break()
        endif()
    endforeach()

    if(NOT _explicit_backend)
        find_program(PYTHON3_PROBE python3)
        if(PYTHON3_PROBE)
            execute_process(
                COMMAND ${PYTHON3_PROBE} ${CMAKE_SOURCE_DIR}/llama_prober.py
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE _probe_output
                RESULT_VARIABLE _probe_result
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_probe_result EQUAL 0 AND _probe_output)
                message(STATUS "GPU auto-probe result: ${_probe_output}")
                separate_arguments(_probe_args UNIX_COMMAND "${_probe_output}")
                foreach(_arg ${_probe_args})
                    if(_arg MATCHES "^-D(GGML_[A-Z]+)=ON$")
                        set(${CMAKE_MATCH_1} ON CACHE BOOL "Auto-detected by gpu-prober" FORCE)
                    elseif(_arg MATCHES "^-D(GGML_[A-Z]+)=OFF$")
                        set(${CMAKE_MATCH_1} OFF CACHE BOOL "Auto-detected by gpu-prober" FORCE)
                    endif()
                endforeach()
            else()
                message(STATUS "GPU auto-probe: no GPUs detected, CPU-only build")
            endif()
        else()
            message(WARNING "GPU auto-probe: python3 not found, skipping")
        endif()
    endif()
endif()

add_executable(llama-stability-runner
    stability_runner.cpp
    kl_divergence.cpp
    round_trip.cpp
)
target_include_directories(llama-stability-runner PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/tools/stability)
target_link_libraries(llama-stability-runner PRIVATE llama llama-common)
install(TARGETS llama-stability-runner RUNTIME DESTINATION bin)


add_executable(llama-calibrate
    calibrator.cpp
    kl_entropy.cpp
    aciq.cpp
    eq_refine.cpp
    scale_table.cpp
)
target_include_directories(llama-calibrate PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/tools/stability)
target_link_libraries(llama-calibrate PRIVATE llama llama-common)
install(TARGETS llama-calibrate RUNTIME DESTINATION bin)

add_executable(rotorquant-benchmark stability_runner.cpp)
target_compile_definitions(rotorquant-benchmark PRIVATE STABILITY_MODE_ROTORQUANT)
target_include_directories(rotorquant-benchmark PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ggml/include ${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/tools/stability)
target_link_libraries(rotorquant-benchmark PRIVATE llama llama-common)

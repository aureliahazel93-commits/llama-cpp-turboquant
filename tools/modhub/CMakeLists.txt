find_package(OpenSSL REQUIRED)

add_executable(llama-mod
    llama-mod.cpp
    registry.cpp
)
target_link_libraries(llama-mod PRIVATE
    llama
    llama-common
    OpenSSL::Crypto
)
target_include_directories(llama-mod PRIVATE ../../common)

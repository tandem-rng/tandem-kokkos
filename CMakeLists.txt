cmake_minimum_required(VERSION 3.25)
project(tandem_kokkos VERSION 0.1.0 LANGUAGES CXX)

# A parent project may build Kokkos in-tree, which defines the target without a package.
if(NOT TARGET Kokkos::kokkos)
    find_package(Kokkos REQUIRED)
endif()

include(GNUInstallDirs)

add_library(tandem_kokkos INTERFACE)
add_library(tandem::kokkos ALIAS tandem_kokkos)
target_include_directories(tandem_kokkos INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/tandem-cuda/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(tandem_kokkos INTERFACE Kokkos::kokkos)
target_compile_features(tandem_kokkos INTERFACE cxx_std_20)
set_target_properties(tandem_kokkos PROPERTIES EXPORT_NAME kokkos)

include(CMakePackageConfigHelpers)
install(DIRECTORY include/tandem external/tandem-cuda/include/tandem
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS tandem_kokkos EXPORT tandem-kokkos-targets)
install(EXPORT tandem-kokkos-targets NAMESPACE tandem::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tandem-kokkos)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tandem-kokkosConfig.cmake
     "include(CMakeFindDependencyMacro)\nfind_dependency(Kokkos)\n"
     "include(\${CMAKE_CURRENT_LIST_DIR}/tandem-kokkos-targets.cmake)\n")
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tandem-kokkosConfigVersion.cmake
                                 COMPATIBILITY SameMinorVersion ARCH_INDEPENDENT)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tandem-kokkosConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/tandem-kokkosConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tandem-kokkos)

option(TANDEM_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(TANDEM_BENCH "Build the benchmark" OFF)

if(TANDEM_TESTS)
    enable_testing()
    add_executable(test_tandem tests/test_tandem.cpp)
    target_link_libraries(test_tandem PRIVATE tandem::kokkos)
    add_test(NAME tandem COMMAND test_tandem ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
    set_tests_properties(tandem PROPERTIES ENVIRONMENT "OMP_PROC_BIND=false")
endif()

if(TANDEM_BENCH)
    add_executable(bench tools/bench.cpp)
    target_link_libraries(bench PRIVATE tandem::kokkos)
endif()

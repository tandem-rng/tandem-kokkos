# Recipe for a Spack package repository. It is not submitted to spack-packages yet.
from spack_repo.builtin.build_systems import cmake
from spack_repo.builtin.build_systems.cmake import CMakePackage

from spack.package import *


class TandemKokkos(CMakePackage):
    """Tandem8x32 random number generator for Kokkos, header only."""

    homepage = "https://github.com/tandem-rng/tandem-kokkos"
    git = "https://github.com/tandem-rng/tandem-kokkos.git"

    license("Apache-2.0")

    # No releases exist. A release adds
    # version("X.Y.Z", sha256="...", url="https://github.com/tandem-rng/tandem-kokkos/archive/refs/tags/vX.Y.Z.tar.gz")
    # The tarball of a release must carry external/tandem-cuda, since GitHub archives omit submodules.
    version("main", branch="main", submodules=True)

    variant("openmp", default=True, description="Build with the Kokkos OpenMP backend")
    variant("cuda", default=False, description="Build with the Kokkos CUDA backend")
    variant(
        "cuda_arch",
        default="none",
        values=("none", "70", "75", "80", "86", "89", "90"),
        multi=False,
        description="CUDA architecture of the Kokkos build",
        when="+cuda",
    )

    depends_on("cxx", type="build")
    depends_on("cmake@3.25:", type="build")
    depends_on("kokkos@4.5:")
    # The tests also run on the Serial space.
    depends_on("kokkos+serial")
    depends_on("kokkos+openmp", when="+openmp")
    depends_on("kokkos+cuda+wrapper", when="+cuda")
    for _arch in ("70", "75", "80", "86", "89", "90"):
        depends_on(f"kokkos cuda_arch={_arch}", when=f"+cuda cuda_arch={_arch}")

    def setup_build_environment(self, env):
        if self.spec.satisfies("+cuda"):
            env.set("NVCC_WRAPPER_DEFAULT_COMPILER", self.compiler.cxx)


class CMakeBuilder(cmake.CMakeBuilder):
    def cmake_args(self):
        return [
            self.define("TANDEM_TESTS", self.pkg.run_tests),
            self.define("TANDEM_BENCH", False),
        ]

    def check(self):
        # A CUDA build needs a GPU, which a build host often lacks.
        if self.spec.satisfies("+cuda"):
            return
        with working_dir(self.build_directory):
            ctest = Executable(self.spec["cmake"].prefix.bin.ctest)
            ctest("--output-on-failure", env={"OMP_PROC_BIND": "false"})

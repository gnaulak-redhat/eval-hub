import os
import platform
import sys
from setuptools import setup
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel


class PlatformSpecificWheel(bdist_wheel):
    """
    Force wheel to be platform-specific.

    Supports platform override via WHEEL_PLATFORM environment variable.
    This allows cross-platform wheel building in CI.
    """

    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # Mark this wheel as platform-specific
        self.root_is_pure = False

    def get_tag(self):
        # Get the default tag
        python, abi, plat = bdist_wheel.get_tag(self)

        # Allow platform override via environment variable (for CI cross-compilation)
        platform_override = os.environ.get("WHEEL_PLATFORM")
        if platform_override:
            plat = platform_override

        # We support all Python 3 versions, any ABI
        python, abi = "py3", "none"
        return python, abi, plat


class PostInstallCommand(install):
    """Post-installation for marking binary as executable."""

    def run(self):
        install.run(self)
        # Make binary executable on Unix-like systems
        if sys.platform != "win32":
            binary_path = self._get_binary_path()
            if binary_path and os.path.exists(binary_path):
                os.chmod(binary_path, 0o755)

    def _get_binary_path(self):
        install_lib = self.install_lib
        if not install_lib:
            return None
        binary_name = self._get_binary_name()
        return os.path.join(install_lib, "evalhub_server", "binaries", binary_name)

    def _get_binary_name(self):
        system = platform.system().lower()
        machine = platform.machine().lower()

        if system == "windows":
            return "eval-hub-windows-amd64.exe"
        elif system == "darwin":
            if machine == "arm64":
                return "eval-hub-darwin-arm64"
            return "eval-hub-darwin-amd64"
        elif system == "linux":
            if "aarch64" in machine or "arm64" in machine:
                return "eval-hub-linux-arm64"
            return "eval-hub-linux-amd64"

        raise RuntimeError(f"Unsupported platform: {system} {machine}")


setup(
    cmdclass={
        "install": PostInstallCommand,
        "bdist_wheel": PlatformSpecificWheel,
    },
)

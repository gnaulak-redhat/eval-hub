# eval-hub-server

This package provides the eval-hub server binary for multiple platforms.

It is primarily intended to be used as a dependency of `eval-hub-sdk`.

## Installation

```bash
pip install eval-hub-server
```

## Usage

```python
from evalhub_server import get_binary_path

# Get the path to the binary
binary_path = get_binary_path()

# Use it however you need (e.g., subprocess)
import subprocess
subprocess.run([binary_path, "serve"], check=True)
```

## Supported Platforms

- Linux: x86_64, arm64
- macOS: x86_64 (Intel), arm64 (Apple Silicon)
- Windows: x86_64

## For eval-hub-sdk Users

If you're using [`eval-hub-sdk`](https://github.com/eval-hub/eval-hub-sdk), you can install the server binary as an extra:

```bash
pip install eval-hub-sdk[server]
```

For more information, see the [eval-hub-sdk repository](https://github.com/eval-hub/eval-hub-sdk).

## Development

This package is automatically built and published when a new release is created in the eval-hub repository. The build process:

1. Compiles Go binaries for all supported platforms
2. Creates platform-specific Python wheels containing the appropriate binary
3. Publishes to PyPI using trusted publishing

## License

Apache-2.0

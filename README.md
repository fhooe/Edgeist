# Edgeist

A framework for deploying and continuously training neural networks on resource limited devices.

## Development

The project is set up with a devcontainer.
Open the [Visual Studio Code](https://code.visualstudio.com/)-Workspace([edgeist.code-workspace](edgeist.code-workspace)) via: `code edgeist.code-workspace`.

Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)-extensions from the `Visual Studio Code`-marketplace, then press `F1` and type `Dev Containers: Rebuild and Reopen in Container`. After pressing `Enter`, the project will load in the devcontainer.

The devcontainer contains everything required to further develop and debug this project.
This includes compilers, embedded debuggers and a fully set-up `venv`.
However, `Visual Studio Code` might not always respect the `venv` as `defaultInterpreter`.
Thus, if packages are missing, first check whether `Visual Studio Code` actually uses the `venv`.

##  Build Instructions

### C++-Projects

This project contains a number of C++-projects:
* [desktop-application](desktop_application)
* [stm32-h563zi](examples/stm32-h563zi)
* [STM32F413](examples/legacy/MC-Code/STM32F413) (legacy)
* [STM32H7A3](examples/legacy/MC-Code/STM32H7A3) (legacy)

All but the legacy projects are [CMake](https://cmake.org/)-based.
Each has its own standalone `CMakeLists.txt`.

`CMake`-projects can be built using the [build_cpp.sh](scripts/build_cpp.sh)-script.
The `build_cpp.sh` has a single argument:
The path to the directory containing the given projects `CMakeLists.txt`:

```bash
./build_cpp.sh /path/to/cmake/lists/
```

This creates a binary.

> Before being able to build `desktop-application`,`mncm_parser` must be run.
> This generates the required files, which must be copied to [generated](desktop_application/generated):
> * Modelenums.h
> * Modelstructs.h
> * Modeltypes.h

### Python-Projects

This project contains a number of python projects:
* [nmcm_checker](nmcm/checker)
* [nmcm_common](nmcm/checker)
* [nmcm_generator](nmcm/generator)
* [nmcm_packer](nmcm/packer)
* [nmcm_parser](nmcm/parser)
* [legacy tests](examples/legacy/MC-Code/Tests)

All projects but `legacy tests` are set up with a `pyproject.toml`.
Each of those projects with a `pyproject.toml` can be built by following these steps:
* Change directory to the one containing the given `pyproject.toml`
* Run `python -m build -n`

This creates an installable wheel.

> All projects but `legacy tests` and `nmcm_packer` depend on `nmcm_common`.
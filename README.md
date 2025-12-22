# Edgeist

A framework for deploying and continuously training neural networks on resource limited devices.

## Development

The project is setup with a devcontainer.
Open the [Visual Studio Code](https://code.visualstudio.com/)-Workspace([edgeist.code-workspace](edgeist.code-workspace)) via: `code edgeist.code-workspace`.

Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)-extensions from the `Visual Studio Code`-marketplace, then press `F1` and type `Dev Containers: Rebuild and Reopen in Container`. After pressing `Enter`, the project will load in the devcontainer.

The devcontainer contains everything required to further develop and debug this project.
This includes compilers, embedded debuggers and a fully set-up `venv`.
However, `Visual Studio Code` might not always respect the `venv` as `defaultInterpreter`.
Thus, if packages are missing, first check whether `Visual Studio Code` actually uses the `venv`.

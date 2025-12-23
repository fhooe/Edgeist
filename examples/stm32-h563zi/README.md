# Example Project using STM32H563ZI

This project will contain an example on how to use `Edgeist` with an STM32H563ZI. 

## Debugging

The `Visual Studio Code` workspace is fully set up.
While inside a devcontainer, use the `Run and Debug`-menu, select `STM32H563ZI Debug` and press the `Start Debugging`-button.
The firmware from this project will be flashed onto the microcontroller
The debugger will pause execution at the beginning of the `main`-function.

## Building

The firmware can be built using the `build_cpp.sh`-script, which requires a single argument:
The path to this very directory.
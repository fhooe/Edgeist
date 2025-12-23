#!/bin/sh

set -eu

if [ "$#" -ne "1" ]; then
  echo "Usage: $0 <directory containing CMakeLists.txt>"
  exit 1
fi

cd "$1" || exit 1
mkdir -p "build"
cd "build" || exit 1

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
J="$(($(nproc) / 2))" # we might run out of memory with too many threads
cmake --build . --parallel "$J" 2>&1

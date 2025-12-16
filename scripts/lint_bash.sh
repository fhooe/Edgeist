#!/bin/sh

set -eu

if [ "$#" -ne "1" ]; then
  echo "Usage: $0 <src-folder>"
  exit 1
fi

SRC="$1"
FILES=$(find "$SRC" -type f -iname \*\.sh)

# We want the paths to split here
# shellcheck disable=SC2086
shellcheck $FILES


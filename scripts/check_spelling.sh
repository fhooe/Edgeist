#!/bin/sh

set -eu

if [ "$#" -ne "1" ]; then
  echo "Usage: $0 <src-folder>"
  exit 1
fi

SRC="$1"
SUBMODULES=$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')
IGNORE_PARAM=$(printf '%s\n' "$SUBMODULES" | sed 's/[^[:space:]]\{1,\}/-o -path .\/&/g')
IGNORE_PARAM=$(echo "$IGNORE_PARAM" |  tr '\n' ' ' | cut -c4-)

if [ -n "$IGNORE_PARAM" ]; then
  # shellcheck disable=SC2086
  FILES=$(find "$SRC" \( $IGNORE_PARAM \) -prune -type f -o -iname \*\.cpp -o -iname \*\.hpp -o -iname \*\.c -o -iname \*\.h -o -iname \*\.py -o -iname \*\.sh -o -iname \*\.rst -o -iname \*\.txt -o -iname \*\.rs -o -iname \*\.md -o -iname \*\.js -o -iname "Dockerfile*" -o -iname "dockerfile*" -o -iname \*\.proto -o -iname \*\.cmake)
else
  FILES=$(find "$SRC" -type f -iname \*\.cpp -o -iname \*\.hpp -o -iname \*\.c -o -iname \*\.h -o -iname \*\.py -o -iname \*\.sh -o -iname \*\.rst -o -iname \*\.txt -o -iname \*\.rs -o -iname \*\.md -o -iname \*\.js -o -iname "Dockerfile*" -o -iname "dockerfile*" -o -iname \*\.proto -o -iname \*\.cmake)
fi

# We want the paths to split here
# shellcheck disable=SC2086
codespell -x $PWD/.codespellignorelines -I $PWD/.codespellignorewords -d $FILES

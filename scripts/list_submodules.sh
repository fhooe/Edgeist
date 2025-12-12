#!/bin/sh
REPO_ROOT=$(git rev-parse --show-toplevel)/
git submodule status | awk -v RR="$REPO_ROOT" '{ print RR$2 }'
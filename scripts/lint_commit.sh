#!/bin/sh

set -eu

# add main as origin (needed in pipelines that only checkout the feat branch)
git remote set-branches --add origin main
git fetch origin main
git fetch --tags

# lint all commit messages relative to the main branch
# (or the latest commit on the main branch itself)
COMMITS=$(git rev-list --count HEAD ^origin/main)
COMMITS=${COMMITS:-1}
if [ "$COMMITS" -eq 0 ]; then
    COMMITS=1
fi
commitlint -V --extends @commitlint/config-conventional --from=HEAD~"$COMMITS" --to HEAD

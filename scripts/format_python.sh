#!/bin/sh

set -eu

usage () {
    echo "Helper script to format Python projects."
    echo ""
	echo "Usage: $0 <base-directory>"
    echo "    base-directory:   The root folder of the project."
    echo ""
    echo "Example: $0 ./"
    echo "    All files in './' will be formatted."
}

# make sure the number of arguments is correct
if [ ! $# -eq 1 ]; then
	usage
	exit 1
fi

# check the value of each argument
BASE_DIR=$1
if [ ! -d "$BASE_DIR" ]; then
    echo "Base directory '$BASE_DIR' does not exist!"
    exit 1
fi

# do basic formatting on the base directory
isort --profile black "$BASE_DIR"
black "$BASE_DIR"

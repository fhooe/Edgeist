#!/bin/bash
set -euo pipefail

echo "Starting pipeline..."

for i in {1..10}; do
    echo "Hello World $i..."
    sleep 1
done

echo "Pipeline completed successfully."

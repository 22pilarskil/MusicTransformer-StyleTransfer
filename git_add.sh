#!/bin/bash

# Recursively find all .sh and .py files in the current directory and add them to staging
find . -type f \( -iname "*.sh" -o -iname "*.py" \) -exec git add {} +

echo "All .sh and .py files in the current directory have been added to staging."


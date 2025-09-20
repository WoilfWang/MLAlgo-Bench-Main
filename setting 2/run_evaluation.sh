#!/bin/bash

answer_folder=${1:-solution/gpt} 
cd ${answer_folder} || { echo "Directory 'script' not found."; exit 1; }

# Initialize counters
total=0
success=0
success_files=()

# Iterate through all Python files in the directory
for file in *.py; do
    if [ -f "$file" ]; then
        total=$((total + 1))
        echo "Running $file ..."

        # Run the Python file with a 10-minute timeout
        timeout 600 python3 "$file"
        status=$?

        # Check exit status
        if [ $status -eq 0 ]; then
            echo "Task succeeded: $file"
            success=$((success + 1))
            success_files+=("$file")
        elif [ $status -eq 124 ]; then
            echo "Task failed (timeout): $file"
        else
            echo "Task failed (error): $file"
        fi
    fi
done

# Calculate success rate
if [ $total -gt 0 ]; then
    success_rate=$((success * 100 / total))
else
    success_rate=0
fi

# Print summary
echo "======================================="
echo "Total tasks: $total"
echo "Successful tasks: $success"
echo "Success rate: $success_rate%"
echo "Successful files:"
for f in "${success_files[@]}"; do
    echo " - $f"
done
echo "======================================="

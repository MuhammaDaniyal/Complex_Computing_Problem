#!/bin/bash
set -e  # Stop if any command fails

NUM="$1"

if [ -z "$NUM" ]; then
    printf "No arguments passed!\nUsing default dataset number 1.\n"
    NUM=1
fi

SET_NAME="set${NUM}"

if [ ! -d "images/$SET_NAME" ]; then
    echo "? Error: Folder images/$SET_NAME does not exist!"
    echo "Available sets:"
    find images -mindepth 1 -maxdepth 1 -type d -exec basename {} \;
    SET_NAME="set1"
    echo "Using default set: $SET_NAME"
fi

NUM_FILES=$(find "images/$SET_NAME" -type f -name 'img[0-9]*' | wc -l)

printf "==========================\n"
printf "|                         |\n"
printf "|   Using dataset: $SET_NAME   |\n"
printf "|                         |\n"
printf "==========================\n"

echo "=========================="
echo "?? Cleaning old builds..."
echo "=========================="
make clean

echo "=========================="
echo "?? Building project (CUDA + OpenACC)..."
echo "=========================="
make all

# Run the example (already linked with OpenACC-enabled library)
echo "=========================="
echo "?? Running example3..."
echo "=========================="
./example3 "$SET_NAME" "$NUM_FILES"

# Profile example3 using gprof
if [ ! -f gmon.out ]; then
    echo "?? gmon.out not found (profiling might not have generated output)."
else
    echo "=========================="
    echo "?? Generating profiling report..."
    echo "=========================="
    gprof example3 gmon.out > profile_output.txt
    gprof -b example3 gmon.out

    echo "=========================="
    echo "?? Creating PDF report..."
    echo "=========================="
    ./gprof2pdf.sh profile_output.txt
    ./gprof2dot.py profile_output.txt > p.dot
    dot -Tpdf -o finalProfile.pdf p.dot
fi

echo "? Done!"

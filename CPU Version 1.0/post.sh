#!/bin/bash

#!/bin/bash
set -e  # Stop if any command fails

# Capture argument
NUM="$1"

if [ -z "$NUM" ]; then
    printf "No arguments passed!\nUsing default dataset number 1.\n"
    NUM=1
fi

# Construct dataset name
SET_NAME="set${NUM}"

# Check if folder exists
if [ ! -d "images/$SET_NAME" ]; then
    echo "❌ Error: Folder images/$SET_NAME does not exist!"
    echo "Available sets:"
    find images -mindepth 1 -maxdepth 1 -type d -exec basename {} \;
    SET_NAME="set1"
    echo "Using default set: $SET_NAME"
fi

# Number of files
NUM_FILES=$(find "images/$SET_NAME" -type f -name 'img[0-9]*' | wc -l)

printf "==========================\n"
printf "|                         |\n"
printf "|   Using dataset: $SET_NAME   |\n"
printf "|                         |\n"
printf "==========================\n"

echo "=========================="
echo "🧹  Running Makefile...  "
echo "=========================="
make

if [ ! -f ./example3 ]; then
    echo "❌ example3 build failed!"
    exit 1
fi

echo "=========================="
echo "🚀 Running example3..."
echo "=========================="
./example3 "$SET_NAME" "$NUM_FILES"
gprof example3 gmon.out > profile_output.txt
gprof -b example3 gmon.out
./gprof2pdf.sh profile_output.txt
./gprof2dot.py profile_output.txt > p.dot
dot -Tpdf -o finalProfile.pdf p.dot


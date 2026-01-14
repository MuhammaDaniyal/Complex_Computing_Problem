#!/bin/bash
set -e  # Stop if any command fails

echo "=========================="
echo "🧹 Cleaning old builds..."
echo "=========================="
make clean

echo "=========================="
echo "🧱 Building project..."
echo "=========================="
make example3

if [ ! -f ./example3 ]; then
    echo "❌ example3 build failed!"
    exit 1
fi

echo "=========================="
echo "🚀 Running example3..."
echo "=========================="
./example3

if [ ! -f gmon.out ]; then
    echo "⚠️ gmon.out not found (profiling might not have generated output)."
else
    echo "=========================="
    echo "📊 Generating profiling report..."
    echo "=========================="
    gprof example3 gmon.out > profile_output.txt
    gprof -b example3 gmon.out

    echo "=========================="
    echo "📄 Creating PDF report..."
    echo "=========================="
    ./gprof2pdf.sh profile_output.txt
    ./gprof2dot.py profile_output.txt > p.dot
    dot -Tpdf -o finalProfile.pdf p.dot
fi

echo "✅ Done!"

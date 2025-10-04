#!/bin/bash

make
./example3
gprof example3 gmon.out > profile_output.txt
gprof -b example3 gmon.out
./gprof2pdf.sh profile_output.txt
./gprof2dot.py profile_output.txt > p.dot
dot -Tpdf -o finalProfile.pdf p.dot


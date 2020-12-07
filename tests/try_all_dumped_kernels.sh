#!/bin/bash
KNOWN_EXAMPLES=$(ls *source.cl)
DUMPED_SOURCES=$(ls ~/CLIntercept_Dump/python3.8/*source.cl)

parallel -v --halt soon,fail=1 -j50% "make {1}_exe" ::: $DUMPED_SOURCES

# for prog in $DUMPED_SOURCES; do
#     echo "Running source: $prog"
#     cp $prog example.cl
#     make kernel.exe || exit
# done

#!/bin/bash
KNOWN_EXAMPLES=$(ls *source.cl)
DUMPED_SOURCES=$(ls ~/CLIntercept_Dump/python3.8/*source.cl | grep -v $KNOWN_EXAMPLES)

parallel -v --dry-run "make {1}.exe" ::: $DUMPED_SOURCES

# for prog in $DUMPED_SOURCES; do
#     echo "Running source: $prog"
#     cp $prog example.cl
#     make kernel.exe || exit
# done

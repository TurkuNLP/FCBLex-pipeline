#!/bin/bash

for f in $(find IMGs -name $1) ; do
    jpegtran -rotate $2 -outfile "$f" "$f"
done

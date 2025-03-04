#!/bin/bash

for f in $(find ${1} -name $2) ; do
    jpegtran -rotate $3 -outfile "$f" "$f"
done

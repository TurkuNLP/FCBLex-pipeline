#!/bin/bash

for f in $(find IMGs -name '*.JPG') ; do
    jpegtran -rotate 270 -outfile "$f" "$f"
done

#!/bin/bash

for f in Texts/*; do cat ${f} | egrep -v "^\s*[0-9]+\s*$" | perl -pe 's/[\r\n]+/ /g' > tmp; mv tmp ${f}; done
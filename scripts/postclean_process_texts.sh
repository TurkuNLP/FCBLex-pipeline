#!/bin/bash

for f in Texts/*; do cat ${f} | egrep -v "^[:0-9:]+$" | perl -pe 's/\n/ /g' > tmp; mv tmp ${f}; done
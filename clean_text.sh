#!/bin/bash

for f in Texts/*; do cat ${f} | perl -pe 's/^\s$//g' | perl -pe 's/-\n+$//g' | egrep -v "^[0-9]" > tmp; mv tmp ${f}; done

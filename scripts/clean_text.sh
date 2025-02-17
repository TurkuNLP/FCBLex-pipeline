#!/bin/bash

for f in Texts/*; do cat ${f} | egrep -v "^#" | perl -pe 's/-\s//g' > tmp; mv tmp ${f}; done

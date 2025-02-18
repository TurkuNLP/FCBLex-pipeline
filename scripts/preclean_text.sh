#!/bin/bash

for f in UncleanTexts/*; do cat ${f} | egrep -v "^#" | perl -pe 's/-\s//g' > tmp; mv tmp ${f}; done

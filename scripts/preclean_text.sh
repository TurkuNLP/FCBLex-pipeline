#!/bin/bash

for f in UncleanTexts/*; do cat ${f} | egrep -v "^#" > tmp; mv tmp ${f}; done

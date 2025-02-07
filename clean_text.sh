#!/bin/bash

for f in Texts/*; do cat ${f} | egrep -v "^#" > tmp; mv tmp ${f}; done

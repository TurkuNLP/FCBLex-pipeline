#!/bin/bash

#python textFromBookPDFs.py
#sh clean_text.sh
python parseBooksTrankit.py
python trankitJson2Conllu.py

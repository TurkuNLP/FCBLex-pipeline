#!/bin/bash

sh scripts/postclean_process_texts.sh
python scripts/parseBooksTrankit.py
python scripts/trankitJson2Conllu.py
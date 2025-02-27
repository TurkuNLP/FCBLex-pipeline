#!/bin/bash

sh scripts/postclean_mac2unix.sh
sleep 5
sh scripts/postclean_process_texts.sh
sleep 5
python scripts/parseBooksTrankit.py
sleep 5
python scripts/trankitJson2Conllu.py
echo "Done!"
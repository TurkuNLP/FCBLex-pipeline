#!/bin/bash

sh scripts/rotateCameraImages.sh
python scripts/processImages.py
python scripts/pdfs2LayoutGoogleDocAI.py
python scripts/layout2Text.py
sh scripts/clean_text.sh
python scripts/parseBooksTrankit.py
python scripts/trankitJson2Conllu.py

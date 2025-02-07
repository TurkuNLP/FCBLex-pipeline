#!/bin/bash

#sh rotateCameraImages.sh
#python processImages.py
#python pdfs2LayoutGoogleDocAI.py
#python layout2Text.py
#sh clean_text.sh
python parseBooksTrankit.py
python trankitJson2Conllu.py

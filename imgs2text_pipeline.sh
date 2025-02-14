#!/bin/bash

if [ -d IMGs ] && [ -f docai ]; then
    echo "Starting pipeline..."
    #sh scripts/rotateCameraImages.sh
    python scripts/processImages.py
    python scripts/pdfs2LayoutGoogleDocAI.py
    python scripts/layout2Text.py
    sh scripts/clean_text.sh
else
    echo "No folders found! Initiating folders!"
    python scripts/initfolder.py
    echo "Remember to have your images in the IMGs folder in properly named sub-folders!"    
fi

#!/bin/bash

if [ -d IMGs ] && [ -f docai ]; then
    echo "Starting pipeline..."
    python scripts/processImages.py
    sleep 5
    python scripts/pdfs2LayoutGoogleDocAI.py
    sleep 5
    python scripts/layout2Text.py
    sleep 5
    sh scripts/preclean_text.sh
    echo "All done!"
else
    echo "No folders found! Initiating folders!"
    python scripts/initfolder.py
    echo "Remember to have your images in the IMGs folder in properly named sub-folders!"    
fi

#Imports

import os

#This file does one thing: initialize the folders needed by the other programs if they do not exist
#Please run this if you haven't cloned this repo before!
def main():
    if not os.path.exists("Data"):
        os.mkdir("Data")
    if not os.path.exists("Conllus"):
        os.mkdir("Conllus")
    if not os.path.exists("Parsed"):
        os.mkdir("Parsed")
    if not os.path.exists("Texts"):
        os.mkdir("Texts")
    if not os.path.exists("UncleanTexts"):
        os.mkdir("UncleanTexts")
    if not os.path.exists("VRT"):
        os.mkdir("VRT")
    if not os.path.exists("PDFs"):
        os.mkdir("PDFs")
    if not os.path.exists("IMGs"):
        os.mkdir("IMGs")
    if not os.path.exists("Layouts"):
        os.mkdir("Layouts")
    if not os.path.exists("ISBN_MAPS"):
        os.mkdir("ISBN_MAPS")
    if not os.path.exists("docai"):
        with open("docai", "w", encoding="utf-8") as writer:
            writer.write("project_id;region;processor_id")

if __name__ == "__main__":
    main()
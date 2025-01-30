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
    if not os.path.exists("VRT"):
        os.mkdir("VRT")
    if not os.path.exists("PDFs"):
        os.mkdir("PDFs")
    if not os.path.exists("IMGs"):
        os.mkdir("IMGs")

if __name__ == "__main__":
    main()
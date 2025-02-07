#Imports
import os
from natsort import natsorted
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import pymupdf
import fitz
from tqdm import tqdm

#Constants
OUTPUT_FOLDER = "PDFs"
INPUT_FOLDER = "IMGs"
IMG_FORMAT = ".png" #switch to jpg if needed

def main():

    #Load and convert images of pages into PDFs
    with tqdm(range(len(os.listdir(INPUT_FOLDER))), desc="Converting books...") as pbar:
        for book in os.listdir(INPUT_FOLDER):
            #Don't do any work if text already exists!
            if os.path.exists(OUTPUT_FOLDER+"/"+book):
                pbar.update(1)
                continue
            else:
                os.mkdir(OUTPUT_FOLDER+"/"+book)
            files = natsorted(os.listdir(INPUT_FOLDER+"/"+book))
            #Show progress bar for books
            with tqdm(range(len(files)), desc="Pages...") as pbar2:
                #For each book in PDFs
                for page in files:
                    imgdoc = fitz.open(INPUT_FOLDER+"/"+book+"/"+page)
                    #Open PDF as pymupdf doc
                    pdfbytes = imgdoc.convert_to_pdf()
                    imgpdf = fitz.open("pdf", pdfbytes)
                    imgpdf.save(OUTPUT_FOLDER+"/"+book+"/"+page.replace(IMG_FORMAT, ".pdf"))
                    pbar2.update(1)

            pbar.update(1)

if __name__ == "__main__":
    main()
   
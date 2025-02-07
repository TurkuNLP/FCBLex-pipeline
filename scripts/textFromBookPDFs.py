#Imports
import pytesseract
import pymupdf
import os
from tqdm import tqdm


def main():
    folder = "PDFs/"
    files = os.listdir(folder)
    #Show progress bar for books
    with tqdm(range(len(files)), desc="Extracting books...") as pbar:
        #For each book in PDFs
        for book in files:
            #Don't do any work if text already exists!
            if os.path.exists("Texts/"+book.replace(".pdf", "")):
                continue
            #Open PDF as pymupdf doc
            doc = pymupdf.open(folder+book)
            text = ""
            with tqdm(range(len(doc)), desc="Extracting text from "+book) as pbar2:
                for index in range(len(doc)):
                    #Grab page from doc
                    page = doc[index]
                    #Transform page to a pillow-image
                    pix = page.get_pixmap()
                    img = pix.pil_image()
                    #Tesseract the image
                    text += pytesseract.image_to_string(img, lang='fin')
                    pbar2.update(1)
            #print(text)
            #Write text to file
            with open("Texts/"+book.replace(".pdf", ""), 'w', encoding='utf-8') as writer:
                writer.write(text)

            pbar.update(1)


if __name__ == "__main__":
    main()
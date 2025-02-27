#Imports
import os
from natsort import natsorted
from google.api_core.client_options import ClientOptions
import pymupdf
from PIL import Image
from tqdm import tqdm
import subprocess

#Constants
OUTPUT_FOLDER = "PDFs"
INPUT_FOLDER = "IMGs"
IMG_FORMAT = ".JPG" #switch to png if needed

def isPageColor(im, red_r: tuple, green_r: tuple, blue_r: tuple) -> bool:
    pix = im.load()

    color_counter = 0
    for y in range(1000, 2000):
        for x in range(1000, 2000):
            if pix[x,y][2]>blue_r[0] and pix[x,y][2]<blue_r[1] and pix[x,y][1]>green_r[0] and pix[x,y][1]<green_r[1] and pix[x,y][0]>red_r[0] and pix[x,y][0]<red_r[1]:
                color_counter += 1
    #If more than 85% of pixels match the specified color ranges, then it is deemed that color
    return color_counter > 8500

def main():

    #Load and convert images of pages into PDFs
    with tqdm(range(len(os.listdir(INPUT_FOLDER))), desc="Converting books...") as pbar:
        for book in os.listdir(INPUT_FOLDER):
            #Don't do any work if pdf already exists!
            if os.path.exists(OUTPUT_FOLDER+"/"+book):
                pbar.update(1)
                continue
            else:
                os.mkdir(OUTPUT_FOLDER+"/"+book)
            #Natsorted files so that the images are correctly placed in the list
            files = natsorted(os.listdir(INPUT_FOLDER+"/"+book))
            files = natsorted([x for x in files.copy() if str(x)[0] != '.'])
            ordered_files = []
            #If dealing with eBook scans
            if files[0].find('.png') != -1:
                ordered_files = files
            #If dealing with physical scans
            else:
                turn_index = 0
                helper = 0
                #Find the 'TURN' picture in the folder
                for pic in files:
                    im = Image.open(INPUT_FOLDER+"/"+book+"/"+pic)
                    #Check for the blue 'TURN' page and break when it's found
                    if isPageColor(im, (60,95), (135,170), (135, 185)):
                        turn_index = helper
                        break
                    helper += 1
                #Starting odd or even?
                im = Image.open(INPUT_FOLDER+"/"+book+"/"+files[-1])
                #Is the last page of the folder the green 'ODD/EVEN' page?
                right_start = isPageColor(im, (115,155), (145,185), (80,125))
                #Special cases when I type in manually
                #right_start = True
                #Reorder the images and turn them appropriately
                for i in range(0, turn_index):
                    if right_start:
                        if len(files)-1*i-2 != turn_index:
                            #Rotate turned pages
                            p = subprocess.Popen(['sh','scripts/rotateCameraImages.sh',files[-1*i-2],'270'])
                            p.communicate()
                            ordered_files.append(files[-1*i-2])
                        p = subprocess.Popen(['sh','scripts/rotateCameraImages.sh',files[i],'90'])
                        p.communicate()
                        ordered_files.append(files[i])
                    else:
                        p = subprocess.Popen(['sh','scripts/rotateCameraImages.sh',files[i],'90'])
                        p.communicate()
                        ordered_files.append(files[i])
                        if len(files)-1*i-1 != turn_index:
                            #Rotate turned pages
                            p = subprocess.Popen(['sh','scripts/rotateCameraImages.sh',files[-1*i-1],'270'])
                            p.communicate()
                            ordered_files.append(files[-1*i-1])

            #Show progress bar for books
            with tqdm(range(len(ordered_files)), desc="Pages...") as pbar2:
                #For each book in PDFs
                for i in range(len(ordered_files)):
                    imgdoc = pymupdf.open(INPUT_FOLDER+"/"+book+"/"+ordered_files[i])
                    #Open PDF as pymupdf doc
                    pdfbytes = imgdoc.convert_to_pdf()
                    imgpdf = pymupdf.open("pdf", pdfbytes)
                    imgpdf.save(OUTPUT_FOLDER+"/"+book+"/"+str(i)+".pdf")
                    pbar2.update(1)

            pbar.update(1)

if __name__ == "__main__":
    main()
   
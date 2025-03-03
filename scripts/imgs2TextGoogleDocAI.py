# [START documentai_process_document]
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
import os
from natsort import natsorted
from tqdm import tqdm

#Constants, switch up if need be
MANUAL_SCAN = True
img_folder = "IMGs"
output_folder = "Texts"
#Chiefly using 'image/jpeg', but switch to 'image/png' if need be (see https://cloud.google.com/document-ai/docs/file-types)
mime_type = "image/png"
#We use the program only to get text - see https://github.com/googleapis/google-cloud-python for inspiration if changing
field_mask = "text"


#Edited version of the code sample from https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/documentai/snippets/process_document_sample.py
def main(
    project_id: str,
    location: str,
    processor_id: str,
    mime_type: str,
    field_mask: Optional[str] = None,
    processor_version_id: Optional[str] = None,
) -> None:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        name = client.processor_path(project_id, location, processor_id)
    with tqdm(range(len(os.listdir(img_folder))), desc="OCRing books...") as pbar:
        #Fetch the images to be scanned
        for book in os.listdir(img_folder):
            output_path = output_folder+"/"+book+".txt"
            #Don't do unnecessary work if book has already been processed
            if os.path.exists(output_path):
                pbar.update(1)
                continue
            
            text = ""
            with tqdm(range(len(os.listdir(img_folder+"/"+book))), desc="Processing pages...") as pbar2:
                #Natsort the images so that we get the book in the correct order
                for page in natsorted(os.listdir(img_folder+"/"+book)):
                    page_path = img_folder+"/"+book+"/"+page
                    #Load image to memory
                    # Read the file into memory
                    with open(page_path, "rb") as image:
                        image_content = image.read()

                    # Load binary data
                    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

                    process_options = documentai.ProcessOptions(

                    )

                    # Configure the process request
                    request = documentai.ProcessRequest(
                        name=name,
                        raw_document=raw_document,
                        field_mask=field_mask,
                        process_options=process_options,
                    )

                    result = client.process_document(request=request)

                    document = result.document

                    text += document.text

                    pbar2.update(1)
                    #The count is maxxed out at 120 requests/min, so need to wait a bit in-between requests
                    #time.sleep(0.1)

            #Write gotten text into a file!
            with open(output_path, 'w', encoding='utf-8') as writer:
                writer.write(text)
                pbar.update(1)

def getKeys(file_path: str = "docai") -> list:
    """
    Helper function to get Google Cloud keys, processor IDs etc. from the 'docai' file
    This is done so that you all won't get my keys :)
    :file_path: str that is by default 'docai
    :return: list of the following structure (project_id, location, processor_id)
    """
    with open(file_path, 'r') as reader:
        return reader.read().split(';')

if __name__ == "__main__":
    project_id, location, processor_id = getKeys()
    main(project_id, location, processor_id, mime_type)
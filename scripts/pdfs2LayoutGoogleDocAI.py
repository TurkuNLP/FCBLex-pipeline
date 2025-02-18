# [START documentai_process_document]
from typing import Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore
import os
from natsort import natsorted
import time
from tqdm import tqdm
import json
from multiprocessing import Process

#Constants, switch up if need be
MANUAL_SCAN = True
doc_folder = "PDFs"
output_folder = "Layouts"
#We use PDFs (see https://cloud.google.com/document-ai/docs/file-types)
mime_type = "application/pdf"
#We use the program only to get text - see https://github.com/googleapis/google-cloud-python for inspiration if changing
field_mask = "text"
#Processor type is 'rc' since we want to use the Layout Analyzer
processor_version = "rc"



#Edited version of the code sample from https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/documentai/snippets/process_document_sample.py
def main(
    project_id: str,
    location: str,
    processor_id: str,
    mime_type: str,
    processor_version: str,
    field_mask: Optional[str] = None,
    processor_version_id: Optional[str] = None,
) -> None:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )

    def processBooks(order: int, pbar: tqdm) -> None:
        indices = list(range(order, len(os.listdir(doc_folder)), 3))
        with pbar:
            #Fetch the images to be scanned
            for index in indices:
                book = os.listdir(doc_folder)[index]
                output_subdir = output_folder+"/"+book
                #Don't do unnecessary work if book has already been processed
                if os.path.exists(output_subdir):
                    pbar.update(1)
                    continue
                else:
                    os.mkdir(output_subdir)
                
                    #Natsort the images so that we get the book in the correct order
                    for page in natsorted(os.listdir(doc_folder+"/"+book)):
                        page_path = doc_folder+"/"+book+"/"+page
                        output_path = output_subdir+"/"+page.replace(".pdf", ".json")
                        #Load doc to memory
                        with open(page_path, "rb") as image:
                            image_content = image.read()

                        # Load binary data
                        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

                        process_options = documentai.ProcessOptions(
                            layout_config=documentai.ProcessOptions.LayoutConfig(
                                chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                                    chunk_size=1000,
                                    include_ancestor_headings=True,
                                )
                            )
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

                        jsonObj = documentai.Document.to_json(document)

                        with open(output_path, "w", encoding="utf-8") as fp:
                            #Parse text if text is not empty
                            json.dump(jsonObj, fp, ensure_ascii=False)
                        #The count is maxxed out at 120 requests/min, so need to wait a bit in-between requests
                        #time.sleep(0.1)
                pbar.update(1)
    tqdms = []
    for j in range(1,4):
        tqdms.append(tqdm(list(range(j, len(os.listdir(doc_folder)), 3)), desc="Processing books in subprocess "+str(j)))
    p1 = Process(name='p1', target=processBooks, args=[0, tqdms[0]])
    p2 = Process(name='p2', target=processBooks, args=[1, tqdms[1]])
    p3 = Process(name='p3', target=processBooks, args=[2, tqdms[2]])
    p1.start()
    p2.start()
    p3.start()

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
    main(project_id, location, processor_id, mime_type, processor_version)
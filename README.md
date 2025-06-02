# Turku Children's Book Corpus pipeline

## Usage

Built to be used on Debian based Linux distributions, so unfortunately will not work out of the gate on Windows machines.

To get started, run:
```
sh imgs2text_pipeline.sh
```
If the folder has not yet been initialized, the script will do it and inform the user.

After initialization (you should see several folders created), place folders containing images (.jpg) to the "IMGs" folder and re-run 
```
sh imgs2text_pipeline.sh
```
After this, the output txt-files will be placed in the "UncleanTexts" folder. If you don't want to perform manual cleaning of these text files, then you can proceed to syntactic parsing by placing the wanted text files into the "Texts" folder, and running:
```
sh txt2conllu_pipeline.sh
```
The output CoNLLU files will be in the "Conllu" folder.

## Step-by-step on how the pipeline works

The pipeline consists of the following steps:
  - Process a folder of jpg-files of book pages and transform them into PDF-files
  - Process PDF-files into a folder os JSON-files with the Layout Processor of Google Document AI
  - Parse the layouts into a single text file
  - Transform the text file into unix-readable format
  - Syntactically parse the text files using Trankit
  - Transform outputted JSON-files into CoNLLU-files

## Important notices
The piepline has been made for our local setup and as such is not necessarily suitable to be run out of the box for other projects. For instance, this pipeline uses Google Document AI to OCR pictures, which is a commercial service and requires a billing account. Another one is that for our manual scanning, we use special colored papers to mark for example, if the first page of a book is a left page or a right page.

For the public, this pipeline serves more-so as an example of what can be done and how a pipeline can be structured.

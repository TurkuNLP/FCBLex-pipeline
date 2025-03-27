#Imports
import pandas as pd
import os
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

def initBooksFromJsons(json_path: str) -> dict:
    """
    Function which takes in a path to a folder containing .json files produced by Trankit and creates python dicts from them
    :JSON_PATH: path to folder as str
    :return: python dict with form [book_name, pandas.DataFrame]
    """

    books = {}
    #Loading the conllus (jsons) as Dataframes

    for file in os.listdir(json_path):
        #Opening json contents
        with open(json_path+"/"+file) as json_file:
            #Transform into dataframe
            df = pd.read_json(json_file)
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

def initBooksFromConllus(conllu_path: str) -> dict:
    """
    Function which takes in a path to a folder containing conllu files and returns a dict with pd.DataFrames of sentence data
    :conllu_path: path to folder of conllus as str
    :return: dict of form [book_name, pd.DataFrame], df is sentence data of a book
    """

    books = {}
    #Loading the conllus as Dataframes
    for file in os.listdir(conllu_path):
        #Opening conllus contents
        with open(conllu_path+"/"+file) as conllu_file:
            #Transform into dataframe
            df = pd.read_csv(conllu_file, sep="[\t]", header=None, on_bad_lines='skip', engine='python')
            #Set names for columns
            df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

#Functions which extract data from a dictionary to get data on sentences

#Function that removes PUNCT
def getNoPunct(sentences: dict) -> dict:
    """
    Function which takes in a dict created by getTokenData and spits out a version with the upos tag PUNCT removed from the DataFrames
    Recommend only editing sentence data
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with upos PUNCT removed from df
    """
    no_punct = {}
    #Remove the rows from each dataframe what are classified as PUNCT
    for key in sentences:
        df = sentences[key]
        no_punct[key] = df[df.upos != "PUNCT"]
    return no_punct

#Function that takes in a dictionary of [book_name, conllu_dataframe] and returns a dictionary with [book_name, sentences_dataframe]
def getSentenceData(books: dict) -> dict:
    """
    Function which takes in a dict created by initBooks and spits out a version where DataFrames only contain sentences data (see CoNLLU file format for more info)
    :books: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with only sentence data
    """
    return_dict = {}
    with tqdm(range(len(books.keys())), desc="Extracting sentences...") as pbar:
        #For key-value pair in dict
        for key in books:
            #Init a new array for sentences
            sentence_dfs = []
            df = books[key]
            
            #Only care about the sentences
            for sentence in df['sentences']:
                #Add dfs created from sentences to list
                sentence_dfs.append(pd.DataFrame.from_dict(sentence['tokens']))
            #Map book_name to a dataframe from all its sentences
            sentece_df = pd.concat(sentence_dfs, ignore_index=True)
            return_dict[key]=sentece_df
            #Update pbar
            pbar.update(1)
    return return_dict

def maskPropn(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function which masks all the proper nouns from the dict for classifier training purposes
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        df.loc[df['upos'] == 'PROPN', 'lemma'] = ""
        df.loc[df['upos'] == 'PROPN', 'text'] = ""
        returnable[key] = df
    return returnable


#Function which returns a dictionary [book_name, lemma_freq_series]
def getLemmaFrequencies(sentences: dict) -> dict:
    """
    Function which takes in sentence data and creates a dict with the lemma frequencies of each book
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.Series] series contains lemmas and frequencies
    """
    lemma_freqs = {}
    for key in sentences:
        lemma_freqs[key] = sentences[key]['lemma'].value_counts()
    return lemma_freqs

def getOnlyAlnums(sentences: dict, column: str) -> dict:
    """
    Function which takes in sentence data and cleans punctuation and other non-alnum characters
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :param column: name of the column which to clean (recommend 'text' or 'lemma')
    :return: dict of the same form
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        #Get rid of PUNCT
        no_punct = df[df.upos != "PUNCT"].copy()
        #Make words lowercase
        no_punct[column] = no_punct[column].apply(lambda x: x.lower())
        #Remove non-alnums
        no_punct[column] = no_punct[column].apply(lambda x: ''.join(filter(str.isalnum, x)))
        #Filter rows with nothing in them
        no_punct = no_punct[no_punct.text != '']
        clean[key] = no_punct
    return clean

#Get frequencies of words)
def getWordFrequencies(sentences: dict) -> dict:
    """
    Function which takes in a dict of sentence data and spits out a dict with pd.Series containing word frequencies of the books
    """
    word_freqs = {}
    for key in sentences:
        df = sentences[key]
        #Map book_name to pivot table
        word_freqs[key] = sentences[key]['text'].value_counts()
    return word_freqs


def getTokenAmounts(sentences: dict) -> dict:
    """
    Get amount of tokens in sentences
    """
    word_amounts = {}
    for key in sentences:
        df = sentences[key]
        word_amounts[key] = len(df)
    return word_amounts

#Get PoS frequencies
def getPOSFrequencies(sentences: dict, scaler_sentences: bool=None) -> dict:
    """
    Function which gets the POS frequencies of sentences of books
    """
    pos_freqs = {}

    if scaler_sentences:
        sentences_sizes = getNumOfSentences(sentences)

    for key in sentences:
        #Map book_name to pivot table
        if scaler_sentences:
            freqs = sentences[key]['upos'].value_counts()
            pos_freqs[key]=freqs/sentences_sizes[key]
        else:
            pos_freqs[key] = sentences[key]['upos'].value_counts()
        #pd.DataFrame.pivot_table(sentences[key], columns='upos', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})

    return pos_freqs


#Functions to get metrics from sentences

#Function the get the average length of the unique lemmas in the sentenes
def getAvgLen(data: dict, column: str=None) -> dict:
    """
    Get the average length of either words or lemmas from sentence data. Works for both original sentence data (pd.DataFrame) and processed ones,
    such as frequency data (pd.Series)
    :data: dict of form [book_name, pd.DataFrame], df should contain sentence data
    :column: name of the CoNLLU column for which to calculate the average. Recommend either 'text' for words and 'lemma' for lemmas
    :return: dict of [book_name, avg_len], where avg_len is float
    """
    avg_lens = {}
    for key in data:
        i = 1
        total_len = 0
        df = data[key]
        if type(df) is pd.DataFrame:
            #For each lemma count the length and add one to counter
            for lemma in df[column]:
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        elif type(df) is pd.Series:
            #For each lemma count the length and add one to counter
            for lemma in list(df.index):
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        #If no lemmas were found (should never happen but just in case), we make the avg_len be 0
        if i==1:
            avg_lens[key] = 0
        else:
            #Map book_name to avg lemma length
            avg_lens[key] = total_len/(i-1.0)
    return avg_lens

def getAvgSentenceLens(books: dict) -> dict:
    """
    Functon for getting the average length of sentences in each book
    :param books: dict of form [id, pd.DataFrame] like in the other methods
    :return: dict of form [id, double], where the double is the average sentence length of the corresponding book
    """
    lens = {}
    for key in books:
        df = books[key]
        help1 = len(df)
        nums = df.id.value_counts()
        num_of_sents = nums.iloc[0]
        lens[key] = (help1/num_of_sents)
    return lens

#Function to calculate DP (deviation of proportions) of all the words in the corpus
def getDP(v: dict, f_series: pd.Series, s: dict) -> tuple:
    """
    Function which calculates the dispersion (DP) based on the formula by Greis
    :v: dict of form [book_name, pd.Series], series has frequencies per book
    :f_df: pd.Series that includes the total frequencies of words/lemmas in the whole corpus
    :s: dict of form [book_name, ratio], where ratio is how much of the whole corpus a book takes
    :return: tuple, where the first member is a pd.Series with DP, the second is a series with DP_norm
    """
    #First get the minimum s
    min_s = 1
    for key in s:
        if s[key] < min_s:
            min_s = s[key]
    #For corpus parts that are length 1
    if min_s == 1:
        min_s = 0
    
    texts = []
    DP = []
    DP_norm = []
    with tqdm(range(len(f_series)), desc="DP calculations") as pbar:

        #Loop through every single word in the corpus
        for word in list(f_series.index):
            #Get the freq of the word in the whole corpus
            f = f_series.loc[word]
            abs_sum = 0
            #For each document in the corpus
            for key in v:
                #Freq of word in document. Set to 0 if not found
                v_i = 0
                try:
                    v_i = v[key].loc[word]*1.0
                except:
                    v_i = 0.0
                #Comparative size of document to whole corpus
                s_i = s[key]
                #Calculate the abs_sum used in calculating DP as written by Gries [2020]
                abs_sum += abs(((v_i)/f)-s_i)
            #Append word to list
            texts.append(word)
            #Calculate and append DP
            dp = 0.5*abs_sum
            DP.append(dp)
            #Append DP_norm to list (alltho with how many documents we have, the normalization doesn't work very well at all)
            DP_norm.append(dp/(1-min_s))
            #Update pbar
            pbar.update(1)
    return pd.Series(DP, texts), pd.Series(DP_norm, texts)

#Function to get contextual diversity
def getCD(v: dict) -> pd.Series:
    """
    Function which gets the contextual diversity of words/lemmas based on frequency data
    """
    #Get number of books
    books_num = len(v.keys())
    word_series = []
    #For each series attached to a book, look for a frequency list and gather all the words in a list
    for key in v:
        v_series = v[key]
        word_series.append(list(v_series.index))
    #Add all words to a new series
    series = pd.Series(word_series)
    #Create series to count in how many books does a word appear in (explode the series comprised of lists)
    CD_raw = series.explode().value_counts()
    #Return Contextual Diversity by dividing the number of appearances by the total number of books
    return CD_raw/books_num

#Functions for getting values for different variables used in metrics


def getL(word_amounts: dict) -> int:
    """
    Function for getting the total length of the corpus in terms of the number of words
    """
    l = 0
    for key in word_amounts:
        l += word_amounts[key]
    return l

def getS(word_amounts: dict, l: int) -> dict:
    """
    Function for getting how big each part is in relation to the total size of the corpus
    """
    s = {}
    for key in word_amounts:
        s[key] = (word_amounts[key]*1.0)/l
    return s


def combineFrequencies(freq_data: dict) -> pd.Series:
    """
    Get the total frequencies of passed freq_data in the corpus
    """
    series = []
    #Add all series to list
    for key in freq_data:
        series.append(freq_data[key])
    #Concat all series together
    ser = pd.concat(series)
    #Return a series containing text as index and total freq in collection in the other
    return ser.groupby(ser.index).sum()


#Functions to do with sub-corpora

def getAvailableAges(corpus: dict) -> list[int]:
    """
    Function which returns the ages that are currently available as sub corpora
    """
    return list(map(int,list(set([findAgeFromID(x) for x in list(corpus.keys())]))))


def getRangeSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age, such that a book will go to +-1 range of its target age
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        age = int(findAgeFromID(key))
        if (num - age < 2 and num - age > -2):
            df = corp[key]
            sub_corp[key] = df
    return sub_corp

def getDistinctSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age exactly, so eahc book will only be included once
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        if key.find('_'+str(num)+'_') != -1:
            sub_corp[key] = corp[key]
    return sub_corp


def combineSubCorpDicts(corps: list) -> dict:
    """
    Combine a list of sub-corp dicts into one dict
    """
    whole = corps[0].copy()
    for i in range(1, len(corps)):
        whole.update(corps[i])
    return whole

def combineSubCorpsData(corps: list):
    """
    Takes in a list of dataframes (or series) and combines them together
    """
    dfs = []
    for df in corps:
        dfs.append(df)
    combined = pd.concat(dfs)
    if type(combined) is pd.DataFrame:
        return combined.groupby(combined.columns[0])[combined.columns[1]].sum().reset_index()
    else:
        return combined.groupby(level=0).sum()
    

def getTypeTokenRatios(v: dict, word_amounts: dict) -> pd.Series:
    """
    Function which gets the type-token ratios of each book that's in the corpus
    :param v:frequency data per book
    :param word_amounts:token amounts per book
    :return: pd.Series with book names being indexes and ttr being values 
    """
    names = []
    ttrs = []
    for key in word_amounts:
        v_df = v[key]
        #Get the number of unique entities in freq data
        types = len(v_df)
        #Get the number of token in book
        tokens = word_amounts[key]
        #Add ttr to lis
        ttrs.append(types/tokens)
        #Add key to list
        names.append(key)
    return pd.Series(ttrs, names)

def getZipfValues(l: int, f: pd.Series) -> pd.Series:
    """
    Function for calculating the Zipf values of words/lemmas in a corpus
    Zipf = ( (raw_freq + 1) / (Tokens per million + Types per million) )+3.0
    :param l: total length of corpus (token amount)
    :param f: series containing frequency data of words/lemmas for the corpus
    :return: pd.Series, where indexes are words/lemmas and values the Zipf values
    """
    indexes = list(f.index)
    types_per_mil = len(indexes)/1000000
    tokens_per_mil = l/1000000
    zipfs = f.values+1
    zipfs = zipfs / (tokens_per_mil + types_per_mil)
    zipfs = np.log10(zipfs)
    zipfs = zipfs + 3.0
    #zipfs_ser = pd.Series(zipfs, indexes)
    return pd.Series(zipfs, indexes)

def cleanLemmas(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on a number of filters on lemma forms to guarantee better quality data
    Does get rid of PUNCT
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['lemma'] = no_punct['lemma'].apply(lambda x: str(x).lower())
        #First mask
        #Remove lemmas which are not alnum or have '-' but no weird chars at start or end, length >1, has no ' ', and has no ','
        m = no_punct.lemma.apply(lambda x: (x.isalnum() 
                                            or (not x.isalnum() and '-' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            or (not x.isalnum() and '#' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            and len(x)>1 
                                            and not ' ' in x
                                            and not ',' in x))
        filtered = no_punct[m]
        #Second mask
        #Remove lemmas that have the same character more than thrice consecutively at the start (Finnish doesn't work like this)
        m_2 = no_punct.lemma.apply(lambda x: conseqChars(x)
                                   and not (x.isnumeric() and len(x)>4)
                                   )
        filtered_2 = filtered[m_2] 
        clean[key] = filtered_2
    return clean

def conseqChars(x: str):
    if len(x)>2:
        return not x[0]==x[1]==x[2]
    else:
        return True
    
def cleanWordBeginnings(books: dict) -> dict:
    """
    Function for cleaning non-alnum characters from the beginning of words in sentence data
    :param books: dict of the sentence data of books
    :return: dictionary, where the dataframes have been cleaned
    """
    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: delNonAlnumStart(str(x)))
        clean[key] = df
    return clean

def delNonAlnumStart(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[0].isalnum() and len(x)>1:
        ind = 0
        for i in range(len(x)):
            if x[i].isalnum():
                ind=i
                break
        return x[ind:]
    return x    

def ignoreOtherAlphabets(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on if words contain characters not in the Finnish alphabe (e.g. cyrillic)
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['text'] = no_punct['text'].apply(lambda x: str(x).lower())
        #First mask
        #Remove words which are not in the Finnish alphabet
        m = no_punct.text.apply(lambda x: (
                len(x.encode("ascii", "ignore")) == len(x)
                or x.find('ä') != -1
                or x.find('ö') != -1 
                or x.find('å') != -1
            )
        )
        filtered = no_punct[m]
        clean[key] = filtered
    return clean

def getSharedWords(wordFrequencies1: dict, wordFrequencies2: dict) -> pd.Series:
    """
    Gives a pd.DataFrame object where there are two columns: first contains those words/lemmas which are shared and the second their combined frequencies
    """
    sub1 = combineFrequencies(wordFrequencies1)
    sub2 = combineFrequencies(wordFrequencies2)

    shared = pd.concat([sub1, sub2])
    mask = shared.index.duplicated(keep=False)
    shared = shared[mask]
    return shared.groupby(shared.index).sum()

def getTaivutusperheSize(corpus: dict) -> pd.Series:
    """
    Returns a series that contains the unique lemmas of the corpus and their 'inflection family size' (taivutusperheen koko)
    """
    #First, combine the data of separate books into one, massive df
    dfs = []
    for book in corpus:
        dfs.append(corpus[book])
    #Then limit to just words and lemmas
    combined_df = pd.concat(dfs, ignore_index=True)[['lemma','feats']]
    #Drop duplicate words
    mask = combined_df.drop_duplicates()
    #Get the counts of lemmas, aka the number of different inflections
    tper = mask.value_counts('lemma')
    return tper

def dfToLowercase(df):
    """
    Simple function which maps all fields into lowercase letters
    """
    return df.copy().applymap(lambda x: str(x).lower())

def getNumOfSentences(corpus: dict) -> dict:
    """
    Function for returning the amount ot sentences for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """

    sentences_sizes = {}
    for id in corpus:
        book = corpus[id]
        #id=='1' means the start of a new sentence (lause)
        sentences_sizes[id] = len(book[book['id'].astype(str)=='1'])
    return sentences_sizes

def getConjPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    conj_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        conj_num = len(book[book['upos'] == ('CCONJ' or 'SCONJ')])
        conj_sentences_ratio[id] = conj_num/sentences_sizes[id]
    return conj_sentences_ratio

def getPosFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted POS features for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['upos'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['upos'] == feature])
        returnable[key] = num
    return returnable

def getDeprelFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted deprel features words for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['deprel'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['deprel'] == feature])
        returnable[key] = num
    return returnable

def getFeatsFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted feats feature for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        #Mask those rows that don't have the wanted feature
        m = book.copy().feats.apply(lambda x: (
            x.find(feature) != -1
                )
            )
        if scaler_sentences:
            num = len(book[m])/sentences_sizes[key]
        else:
            num = len(book[m])
        returnable[key] = num
    return returnable

def cohensdForSubcorps(subcorp1: dict, subcorp2: dict) -> float:
    """
    Function for calculating the effect size using Cohen's d for some feature values of two subcorpora
    :param subcorp1: dictionary of form [id, float], calculated with e.g. getDeprelFeaturePerBook()
    :param subcorp2: dict of the same form as above
    :return: flaot measuring the effect size
    """
    data1 = list(subcorp1.values())
    data2 = list(subcorp2.values())
    #Sample size
    n1, n2 = len(data1), len(data2)
    #Variance
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    #Pooled standard deviation
    s = math.sqrt( ((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2) )
    #Return Cohen's d
    return ((np.mean(data1)-np.mean(data2)) / s)

def getMultiVerbConstrNumPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    multiverb_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        modal_verb_num = len(book[((book['upos'] == 'AUX') & (book['xpos'] == 'V') & (book['deprel'] == 'aux')) | ((book['upos'] == 'VERB') & (book['deprel'] == 'xcomp'))])
        multiverb_sentences_ratio[id] = modal_verb_num/sentences_sizes[id]
    return multiverb_sentences_ratio

def getDictAverage(corp_data: dict) -> float:
    """
    Simple function for calculating the average value of a dict containing book ids and some numerical values
    """
    return sum(list((corp_data.values())))/len(list(corp_data.keys()))

def getBookLemmaCosineSimilarities(corpus: dict, f_lemma: pd.Series) -> pd.DataFrame:
    """
    Calculating cosine similarities of all lemmas between the books in the corpus. Inspired by Korochkina et el. 2024
    """
    tf_idf_scores = {}

    #Sort the books so that we get groupings by age group
    sorted_keys = list(corpus.keys())
    sorted_keys.sort(key=lambda x:int(findAgeFromID(x)))

    #Get all corpus' lemmas from lemma frequency data
    all_lemmas = list(f_lemma.index)
    book_vectorizer = TfidfVectorizer(vocabulary=all_lemmas)
    for book in sorted_keys:
        #Tf-idf scores from lemma data of a book
        book_lemmas = " ".join(corpus[book]['lemma'].values)
        #print(book_lemmas.values)
        tf_idf_scores[book] = book_vectorizer.fit_transform([book_lemmas])
    similarity_scores = {}
    for book in sorted_keys:
        #Compare current book to every other book
        scores = []
        for comp in sorted_keys:
            scores.append(cosine_similarity(tf_idf_scores[book], tf_idf_scores[comp]))
        similarity_scores[book] = scores
    #Create df
    matrix_df = pd.DataFrame.from_dict(similarity_scores, orient='index').transpose()
    #Set indexes correctly
    matrix_df.index = tf_idf_scores.keys()
    #Dig out the values from nd.array
    matrix_df_2 = matrix_df.copy().applymap(lambda x: x[0][0])
    return matrix_df_2
    return addAgeGroupSeparatorsToDF(matrix_df_2)

def addAgeGroupSeparatorsToDF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for adding separator lines to a df that's meant to be shown as a heatmap!
    """
    indices = list(df.index)
    one2two = 0
    while int(findAgeFromID(indices[one2two]))<9:
        one2two += 1
    two2three = one2two
    while int(findAgeFromID(indices[two2three]))<13:
        two2three += 1
    df.insert(one2two, 'one2two', pd.Series([1]*len(indices)))
    df.insert(two2three+1, 'two2three', pd.Series([1]*len(indices)))
    temp_dict = dict(zip(df.columns, ([1]*(len(indices)+2))))
    row1 = pd.DataFrame(temp_dict, index=['one2two'])
    row2 = pd.DataFrame(temp_dict, index=['two2three'])
    df_2 = pd.concat([df.iloc[:one2two], row1, df.iloc[one2two:]])
    df_2 = pd.concat([df_2.iloc[:two2three+1], row2, df_2.iloc[two2three+1:]])
    return df_2

def combineSeriesForExcelWriter(f_lemmas, corpus, lemma_DP, lemma_CD, f_words, word_DP, word_CD):
    """
    Helper function for combining various Series containing lemma/word data into compact dataframes
    """
    lemma_data = pd.concat([f_lemmas, getTaivutusperheSize(corpus), lemma_DP, lemma_CD], axis=1)
    lemma_data.columns = ['frequency','t_perh_size', 'DP', 'CD']

    word_data = pd.concat([f_words, word_DP, word_CD], axis=1)
    word_data.columns = ['frequency', 'DP', 'CD']


    return lemma_data, word_data


def filterRegisters(corpus: dict[str,pd.DataFrame], registers: list[int]) -> dict[str,pd.DataFrame]:
    """
    Function for creating a register sepcific subcorpus. Valid registers are:
    1 = Fiction
    2 = Non-fiction, non-textbook
    3 = Textbook
    You can pass as many registers as you want (any valid subset of [1,2,3])
    """

    returnable = {}
    for key in corpus:
        if int(key[-1]) in registers:
            df = corpus[key]
            returnable[key] = df
    return returnable

#Moving to a regression task instead of hard age groups

def findAgeFromID(key: str) -> str:
    "Function that returns the age information embedded in a book id"
    return key[key.find('_')+1:key.find('_')+1+key[key.find('_')+1:].find('_')]

def mapGroup2Age(corpus: dict[str,pd.DataFrame], sheet_path: str) -> dict[str,pd.DataFrame]:
    """
    Function for changing the file keys to use exact ages instead of age groups [1,3]
    """

    returnable = {}
    isbn2age_series = pd.DataFrame(pd.read_excel(sheet_path, index_col=0))
    for key in corpus:
        df = corpus[key]
        new_key = key
        new_key = key[:14] +  str(isbn2age_series.at[int(key[:13]),isbn2age_series.columns[0]]) + key[15:]
        returnable[new_key] = df
    return returnable

#Writing all data into one big xlsx-file
def writeDataToXlsx(
        name, 
        lemmas,
        words, 
        pos_freqs
        ):
    """
    Write all wanted data to an xlsx file for testing purposes
    """
    with pd.ExcelWriter("Data/"+name+".xlsx") as writer:
        words.to_excel(writer, sheet_name="Word data")
        lemmas.to_excel(writer, sheet_name="Lemma data")
        pos_freqs.to_excel(writer, sheet_name="POS data")
        #avg_lens.to_excel(writer, sheet_name="Average unique lengths data")
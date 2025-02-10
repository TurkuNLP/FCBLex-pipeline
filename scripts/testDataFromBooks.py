import bookdatafunctions as bdf
import pandas as pd

#Constants
JSON_PATH = "Parsed"
CONLLU_PATH = "Conllus"

def main():
    

    books = bdf.initBooksFromJsons(JSON_PATH)

    #Move to working with just sentence data
    #Whole corpus
    sentences = bdf.initBooksFromConllus(CONLLU_PATH)

    #Subcorpora based on the target age groups
    sentences_1 = bdf.cleanLemmas(bdf.getSubCorp(sentences, 1))
    sentences_2 = bdf.cleanLemmas(bdf.getSubCorp(sentences, 2))
    sentences_3 = bdf.cleanLemmas(bdf.getSubCorp(sentences, 3))

    #Versions of sentences for more meaningful data
    sentences_no_punct_1 = bdf.cleanLemmas(sentences_1)
    sentences_no_punct_2 = bdf.cleanLemmas(sentences_2)
    sentences_no_punct_3 = bdf.cleanLemmas(sentences_3)
    sentences_no_punct = bdf.combineSubCorpDicts([sentences_no_punct_1, sentences_no_punct_2, sentences_no_punct_3])

    #Count lemma frequencies

    lemma_freqs_1 = bdf.getLemmaFrequencies(sentences_1)
    lemma_freqs_2 = bdf.getLemmaFrequencies(sentences_2)
    lemma_freqs_3 = bdf.getLemmaFrequencies(sentences_3)

    lemma_freqs = bdf.combineSubCorpDicts([lemma_freqs_1, lemma_freqs_2, lemma_freqs_3])

    #Count word frequencies

    word_freqs_1 = bdf.getWordFrequencies(sentences_1)
    word_freqs_2 = bdf.getWordFrequencies(sentences_2)
    word_freqs_3 = bdf.getWordFrequencies(sentences_3)

    word_freqs = bdf.combineSubCorpDicts([word_freqs_1, word_freqs_2, word_freqs_3])

    #Just for interest's sake, info on how many tokens (non-punct) are in each book

    word_amounts_1 = bdf.getTokenAmounts(sentences_1)
    word_amounts_2 = bdf.getTokenAmounts(sentences_2)
    word_amounts_3 = bdf.getTokenAmounts(sentences_3)

    word_amounts = bdf.combineSubCorpDicts([word_amounts_1, word_amounts_2, word_amounts_3])

    #Count the average uniq lemma lengths
    avg_uniq_lemma_lens_1 = bdf.getAvgLen(lemma_freqs_1, 'lemma')
    avg_uniq_lemma_lens_2 = bdf.getAvgLen(lemma_freqs_2, 'lemma')
    avg_uniq_lemma_lens_3 = bdf.getAvgLen(lemma_freqs_3, 'lemma')
    avg_uniq_lemma_lens = bdf.getAvgLen(lemma_freqs, 'lemma')
    #print(avg_uniq_lemma_lens)

    #Count the average uniq word lengths
    avg_uniq_word_lens_1 = bdf.getAvgLen(word_freqs_1, 'text')
    avg_uniq_word_lens_2 = bdf.getAvgLen(word_freqs_2, 'text')
    avg_uniq_word_lens_3 = bdf.getAvgLen(word_freqs_3, 'text')
    avg_uniq_word_lens = bdf.getAvgLen(word_freqs, 'text')
    #print(avg_uniq_word_lens)

    #Count the average lemma lengths
    avg_lemma_lens_1 = bdf.getAvgLen(sentences_no_punct_1, 'lemma')
    avg_lemma_lens_2 = bdf.getAvgLen(sentences_no_punct_2, 'lemma')
    avg_lemma_lens_3 = bdf.getAvgLen(sentences_no_punct_3, 'lemma')
    avg_lemma_lens = bdf.getAvgLen(sentences_no_punct, 'lemma')
    #print(avg_lemma_lens)

    #Count the average word lengths
    avg_word_lens_1 = bdf.getAvgLen(sentences_no_punct_1, 'text')
    avg_word_lens_2 = bdf.getAvgLen(sentences_no_punct_2, 'text')
    avg_word_lens_3 = bdf.getAvgLen(sentences_no_punct_3, 'text')
    avg_word_lens = bdf.getAvgLen(sentences_no_punct, 'text')
    #print(avg_word_lens)


    #Combining results into dfs

    avg_uniq_lens_df_1 = pd.DataFrame.from_dict([avg_uniq_lemma_lens_1, avg_uniq_word_lens_1]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})
    avg_uniq_lens_df_2 = pd.DataFrame.from_dict([avg_uniq_lemma_lens_2, avg_uniq_word_lens_2]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})
    avg_uniq_lens_df_3 = pd.DataFrame.from_dict([avg_uniq_lemma_lens_3, avg_uniq_word_lens_3]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})
    avg_uniq_lens_df = pd.DataFrame.from_dict([avg_uniq_lemma_lens, avg_uniq_word_lens]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})


    avg_lens_df_1 = pd.DataFrame.from_dict([avg_lemma_lens_1, avg_word_lens_1]).transpose().rename(columns={0: 'All lemmas avg length', 1: 'All words avg length'})
    avg_lens_df_2 = pd.DataFrame.from_dict([avg_lemma_lens_2, avg_word_lens_2]).transpose().rename(columns={0: 'All lemmas avg length', 1: 'All words avg length'})
    avg_lens_df_3 = pd.DataFrame.from_dict([avg_lemma_lens_3, avg_word_lens_3]).transpose().rename(columns={0: 'All lemmas avg length', 1: 'All words avg length'})
    avg_lens_df = pd.DataFrame.from_dict([avg_lemma_lens, avg_word_lens]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})

    #Constants to be used in different measures

    #The length of the corpus in words (no PUNCT)
    l_1 = bdf.getL(word_amounts_1)
    l_2 = bdf.getL(word_amounts_2)
    l_3 = bdf.getL(word_amounts_3)
    l = l_1+l_2+l_3
    #The length of the corpus in parts
    n = len(sentences.keys())
    #The percentages of the n corpus part sizes
    s_1 = bdf.getS(word_amounts_1, l_1)
    s_2 = bdf.getS(word_amounts_2, l_2)
    s_3 = bdf.getS(word_amounts_3, l_3)
    s = bdf.getS(word_amounts, l)
    #The overall frequencies of words in corpus
    f_words_1 = bdf.combineFrequencies(word_freqs_1)
    f_words_2 = bdf.combineFrequencies(word_freqs_2)
    f_words_3 = bdf.combineFrequencies(word_freqs_3)
    f_words = bdf.combineFrequencies(word_freqs)
    #The overall frequencies of lemmas in corpus
    f_lemmas_1 = bdf.combineFrequencies(lemma_freqs_1)
    f_lemmas_2 = bdf.combineFrequencies(lemma_freqs_2)
    f_lemmas_3 = bdf.combineFrequencies(lemma_freqs_3)
    f_lemmas = bdf.combineFrequencies(lemma_freqs)
    #The frequencies of words in each corpus part
    v_words = word_freqs
    #The frequencies of lemmas in each corpus part
    v_lemmas = lemma_freqs

    #Whole corpus
    lemma_DP = bdf.getDP(v_lemmas, f_lemmas, s)
    #Sub-corpora
    lemma_DP_1 = bdf.getDP(lemma_freqs_1, f_lemmas_1, s_1)
    lemma_DP_2 = bdf.getDP(lemma_freqs_2, f_lemmas_2, s_2)
    lemma_DP_3 = bdf.getDP(lemma_freqs_3, f_lemmas_3, s_3)
    #Whole corpus
    word_DP = bdf.getDP(v_words, f_words, s)
    #Sub-corpora
    word_DP_1 = bdf.getDP(word_freqs_1, f_words_1, s_1)
    word_DP_2 = bdf.getDP(word_freqs_2, f_words_2, s_2)
    word_DP_3 = bdf.getDP(word_freqs_3, f_words_3, s_3)

    #Getting CD

    #Whole corpus
    word_CD = bdf.getCD(v_words)
    #Sub-corpora
    word_CD_1 = bdf.getCD(word_freqs_1)
    word_CD_2 = bdf.getCD(word_freqs_2)
    word_CD_3 = bdf.getCD(word_freqs_3)

    #Whole corpus
    lemma_CD = bdf.getCD(v_lemmas)
    #Sub-corpora
    lemma_CD_1 = bdf.getCD(lemma_freqs_1)
    lemma_CD_2 = bdf.getCD(lemma_freqs_2)
    lemma_CD_3 = bdf.getCD(lemma_freqs_3)

    #Get POS frequencies

    #Count POS frequencies

    pos_freqs_per_book = bdf.getPOSFrequencies(sentences)

    pos_freqs_1 = bdf.combineFrequencies(bdf.getSubCorp(pos_freqs_per_book, 1))
    pos_freqs_2 = bdf.combineFrequencies(bdf.getSubCorp(pos_freqs_per_book, 2))
    pos_freqs_3 = bdf.combineFrequencies(bdf.getSubCorp(pos_freqs_per_book, 3))

    pos_freqs_corpus = bdf.combineFrequencies(pos_freqs_per_book)

    #Commencing the writing part
    bdf.writeDataToXlsx("Whole_corpus", f_words, f_lemmas, pos_freqs_corpus, lemma_DP, word_DP, lemma_CD, word_CD, avg_uniq_lens_df, avg_lens_df)
    bdf.writeDataToXlsx("Sub_corp_1", f_words_1, f_lemmas_1, pos_freqs_1, lemma_DP_1, word_DP_1, lemma_CD_1, word_CD_1, avg_uniq_lens_df_1, avg_lens_df_1)
    bdf.writeDataToXlsx("Sub_corp_2", f_words_2, f_lemmas_2, pos_freqs_2, lemma_DP_2, word_DP_2, lemma_CD_2, word_CD_2, avg_uniq_lens_df_2, avg_lens_df_2)
    bdf.writeDataToXlsx("Sub_corp_3", f_words_3, f_lemmas_3, pos_freqs_3, lemma_DP_3, word_DP_3, lemma_CD_3, word_CD_3, avg_uniq_lens_df_3, avg_lens_df_3)


if __name__ == "__main__":
    main()
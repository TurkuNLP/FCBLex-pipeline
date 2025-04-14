import bookdatafunctions as bdf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Constants
JSON_PATH = "Parsed"
CONLLU_PATH = "Conllus"
ISBN2AGE_PATH = "ISBN_MAPS/ISBN2AGE.xlsx"

def main():
    

    #books = bdf.initBooksFromJsons(JSON_PATH)

    #Move to working with just sentence data
    #Whole corpus
    sentences = bdf.mapGroup2Age(bdf.cleanWordBeginnings(bdf.cleanLemmas(bdf.initBooksFromConllus(CONLLU_PATH))), ISBN2AGE_PATH)

    ages = bdf.getAvailableAges(sentences)

    #Subcorpora based on the target age groups
    sub_sentences = []
    for i in ages:
        sub_sentences.append(bdf.cleanWordBeginnings(bdf.cleanLemmas(bdf.getDistinctSubCorp(sentences, i))))

    #Versions of sentences for more meaningful data
    sub_sentences_no_punct = []
    for i in range(len(ages)):
        sub_sentences_no_punct.append(bdf.cleanWordBeginnings(bdf.cleanLemmas(sub_sentences[i])))
    sentences_no_punct = bdf.cleanWordBeginnings(sentences)

    #Count lemma frequencies
    sub_lemma_freqs = []
    for i in range(len(ages)):
        sub_lemma_freqs.append(bdf.getLemmaFrequencies(sub_sentences[i]))

    lemma_freqs = bdf.getLemmaFrequencies(sentences)

    #Count word frequencies
    sub_word_freqs = []
    for i in range(len(ages)):
        sub_word_freqs.append(bdf.getWordFrequencies(sub_sentences[i]))

    word_freqs = bdf.getWordFrequencies(sentences)

    #Just for interest's sake, info on how many tokens (non-punct) are in each book

    sub_word_amounts = []
    for i in range(len(ages)):
        sub_word_amounts.append(bdf.getTokenAmounts(sub_sentences[i]))

    word_amounts = bdf.getTokenAmounts(sentences)

    #Count the average uniq lemma lengths
    sub_avg_uniq_lemma_lens = []
    for i in range(len(ages)):
        sub_avg_uniq_lemma_lens.append(bdf.getAvgLen(sub_lemma_freqs[i], 'lemma'))
    avg_uniq_lemma_lens = bdf.getAvgLen(lemma_freqs, 'lemma')
    #print(avg_uniq_lemma_lens)

    #Count the average uniq word lengths
    sub_avg_uniq_word_lens = []
    for i in range(len(ages)):
        sub_avg_uniq_word_lens.append(bdf.getAvgLen(sub_word_freqs[i], 'text'))
    avg_uniq_word_lens = bdf.getAvgLen(word_freqs, 'text')
    #print(avg_uniq_word_lens)

    #Count the average lemma lengths
    sub_avg_lemma_lens = []
    for i in range(len(ages)):
        sub_avg_lemma_lens.append(bdf.getAvgLen(sub_sentences_no_punct[i], 'lemma'))
    avg_lemma_lens = bdf.getAvgLen(sentences_no_punct, 'lemma')
    #print(avg_lemma_lens)

    #Count the average word lengths
    sub_avg_word_lens = []
    for i in range(len(ages)):
        sub_avg_word_lens.append(bdf.getAvgLen(sub_sentences_no_punct[i], 'text'))
    avg_word_lens = bdf.getAvgLen(sentences_no_punct, 'text')
    #print(avg_word_lens)


    #Combining results into dfs
    avg_uniq_lens_dfs = []
    for i in range(len(ages)):
        avg_uniq_lens_dfs.append(pd.DataFrame.from_dict([sub_avg_uniq_lemma_lens[i], sub_avg_uniq_word_lens[i]]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'}))
    avg_uniq_lens_df = pd.DataFrame.from_dict([avg_uniq_lemma_lens, avg_uniq_word_lens]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})



    avg_lens_dfs = []
    for i in range(len(ages)):
        avg_lens_dfs.append(pd.DataFrame.from_dict([sub_avg_lemma_lens[i], sub_avg_word_lens[i]]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'}))
    avg_lens_df = pd.DataFrame.from_dict([avg_lemma_lens, avg_word_lens]).transpose().rename(columns={0: 'Unique lemmas avg length', 1: 'Unique words avg length'})

    #Constants to be used in different measures

    #The length of the corpus in words (no PUNCT)
    sub_l = []
    for i in range(len(ages)):
        sub_l.append(bdf.getL(sub_word_amounts[i]))
    l = sum(sub_l)
    #The length of the corpus in parts
    n = len(sentences.keys())
    #The percentages of the n corpus part sizes
    sub_s = []
    for i in range(len(ages)):
        sub_s.append(bdf.getS(sub_word_amounts[i], sub_l[i]))
    s = bdf.getS(word_amounts, l)
    #The overall frequencies of words in corpus
    sub_f_words = []
    for i in range(len(ages)):
        sub_f_words.append(bdf.combineFrequencies(sub_word_freqs[i]))
    f_words = bdf.combineFrequencies(word_freqs)
    #The overall frequencies of lemmas in corpus
    sub_f_lemmas = []
    for i in range(len(ages)):
        sub_f_lemmas.append(bdf.combineFrequencies(sub_lemma_freqs[i]))
    f_lemmas = bdf.combineFrequencies(lemma_freqs)
    #The frequencies of words in each corpus part
    v_words = word_freqs
    #The frequencies of lemmas in each corpus part
    v_lemmas = lemma_freqs

    #Whole corpus
    lemma_DP, lemma_DP_norm = bdf.getDP(v_lemmas, f_lemmas, s)
    #Sub-corpora
    sub_lemma_dp = []
    for i in range(len(ages)):
        sub_lemma_dp.append(bdf.getDP(sub_lemma_freqs[i], sub_f_lemmas[i], sub_s[i])[0])
    #Whole corpus
    word_DP, word_DP_norm = bdf.getDP(v_words, f_words, s)
    #Sub-corpora
    sub_word_dp = []
    for i in range(len(ages)):
        sub_word_dp.append(bdf.getDP(sub_word_freqs[i], sub_f_words[i], sub_s[i])[0])

    #Getting CD

    #Whole corpus
    word_CD = bdf.getCD(v_words)
    #Sub-corpora
    sub_word_cd = []
    for i in range(len(ages)):
        sub_word_cd.append(bdf.getCD(sub_word_freqs[i]))

    #Whole corpus
    lemma_CD = bdf.getCD(v_lemmas)
    #Sub-corpora
    sub_lemma_cd = []
    for i in range(len(ages)):
        sub_lemma_cd.append(bdf.getCD(sub_lemma_freqs[i]))
        
    #Get POS frequencies

    #Count POS frequencies

    pos_freqs_per_book = bdf.getPOSFrequencies(sentences)

    sub_pos_freqs = []
    for i in ages:
        sub_pos_freqs.append(bdf.combineFrequencies(bdf.getDistinctSubCorp(pos_freqs_per_book, i)))

    pos_freqs_corpus = bdf.combineFrequencies(pos_freqs_per_book)

    #Combine previously gathered data into neat dataframes

    lemma_data, word_data = bdf.combineSeriesForExcelWriter(f_lemmas, sentences, lemma_DP, lemma_CD, f_words, word_DP, word_CD)
    sub_data = []
    for i in range(len(ages)):
        sub_data.append(bdf.combineSeriesForExcelWriter(sub_f_lemmas[i], sub_sentences[i], sub_lemma_dp[i], sub_lemma_cd[i], sub_f_words[i], sub_word_dp[i], sub_word_cd[i]))


    #Write to excel-files
    bdf.writeDataToXlsx("Whole_corpus", lemmas=lemma_data, words=word_data, pos_freqs=pos_freqs_corpus)
    for i in range(len(ages)):
        bdf.writeDataToXlsx("Sub_corpus_"+str(ages[i]), lemmas=sub_data[i][0], words=sub_data[i][1], pos_freqs=sub_pos_freqs[i])


if __name__ == "__main__":
    main()
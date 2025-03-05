import bookdatafunctions as bdf
import pandas as pd
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings('ignore')

DEPRELS = ['root', 'nsubj', 'advmod', 'obl', 'obj', 'conj', 'aux', 'cc', 'amod', 'nmod:poss', 'mark', 'cop', 'nsubj:cop', 'advcl', 'xcomp', 'case', 'det', 'ccomp', 'nmod', 'parataxis', 'acl:relcl', 'acl', 'xcomp:ds', 'discourse', 'nummod', 'fixed', 'cop:own', 'appos', 'flat:name', 'compound:nn', 'aux:pass', 'vocative', 'nmod:gobj', 'nmod:gsubj', 'compound:prt', 'csubj:cop', 'flat:foreign', 'orphan', 'cc:preconj', 'csubj', 'compound', 'flat', 'goeswith', 'dep']

CASES = ['Case=Nom', 'Case=Gen', 'Case=Par', 'Case=Ill', 'Case=Ine', 'Case=Ela', 'Case=Ade', 'Case=All', 'Case=Ess', 'Case=Abl', 'Case=Tra', 'Case=Acc', 'Case=Ins', 'Case=Abe', 'Case=Com']

VERBFORMS = ['VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part']

VERBVOICES = ['Voice=Act', 'Voice=Pass']

VERBMOODS = ['Mood=Ind', 'Mood=Cnd', 'Mood=Imp']

POS = ['NOUN', 'VERB', 'PRON', 'ADV', 'AUX', 'ADJ', 'PROPN', 'CCONJ', 'SCONJ', 'ADP', 'NUM', 'INTJ']

print("Start...")

books = bdf.cleanWordBeginnings(bdf.cleanLemmas(bdf.initBooksFromConllus("Conllus")))

sub_1 = bdf.getSubCorp(books, 1)
sub_2 = bdf.getSubCorp(books, 2)
sub_3 = bdf.getSubCorp(books, 3)

print("Processing dependency relations...")

deprel_effect_sizes_12 = {}
deprel_effect_sizes_13 = {}
deprel_effect_sizes_23 = {}

for deprel in DEPRELS:
    deprel_effect_sizes_12[deprel] = bdf.cohensdForSubcorps(bdf.getDeprelFeaturePerBook(sub_1, deprel, True), bdf.getDeprelFeaturePerBook(sub_2, deprel, True))
    deprel_effect_sizes_13[deprel] = bdf.cohensdForSubcorps(bdf.getDeprelFeaturePerBook(sub_1, deprel, True), bdf.getDeprelFeaturePerBook(sub_3, deprel, True))
    deprel_effect_sizes_23[deprel] = bdf.cohensdForSubcorps(bdf.getDeprelFeaturePerBook(sub_2, deprel, True), bdf.getDeprelFeaturePerBook(sub_3, deprel, True))

deprel_effect_sizes = pd.DataFrame({"1-2":deprel_effect_sizes_12.values(), "2-3":deprel_effect_sizes_23.values(), "1-3":deprel_effect_sizes_13.values()}, index=deprel_effect_sizes_23.keys())

print("Processing cases and verb-forms...")

feats_effect_sizes_12 = {}
feats_effect_sizes_13 = {}
feats_effect_sizes_23 = {}

for ctg in [CASES, VERBFORMS, VERBMOODS, VERBVOICES]:
    for feat in ctg:
        feats_effect_sizes_12[feat] = bdf.cohensdForSubcorps(bdf.getFeatsFeaturePerBook(sub_1, feat, True), bdf.getFeatsFeaturePerBook(sub_2, feat, True))
        feats_effect_sizes_13[feat] = bdf.cohensdForSubcorps(bdf.getFeatsFeaturePerBook(sub_1, feat, True), bdf.getFeatsFeaturePerBook(sub_3, feat, True))
        feats_effect_sizes_23[feat] = bdf.cohensdForSubcorps(bdf.getFeatsFeaturePerBook(sub_2, feat, True), bdf.getFeatsFeaturePerBook(sub_3, feat, True))

feats_effect_sizes = pd.DataFrame({"1-2":feats_effect_sizes_12.values(), "2-3":feats_effect_sizes_23.values(), "1-3":feats_effect_sizes_13.values()}, index=feats_effect_sizes_23.keys())

print("Processing POS features...")

pos_effect_sizes_12 = {}
pos_effect_sizes_13 = {}
pos_effect_sizes_23 = {}

for pos in POS:
    pos_effect_sizes_12[pos] = bdf.cohensdForSubcorps(bdf.getPosFeaturePerBook(sub_1, pos, True), bdf.getPosFeaturePerBook(sub_2, pos, True))
    pos_effect_sizes_13[pos] = bdf.cohensdForSubcorps(bdf.getPosFeaturePerBook(sub_1, pos, True), bdf.getPosFeaturePerBook(sub_3, pos, True))
    pos_effect_sizes_23[pos] = bdf.cohensdForSubcorps(bdf.getPosFeaturePerBook(sub_2, pos, True), bdf.getPosFeaturePerBook(sub_3, pos, True))

pos_effect_sizes = pd.DataFrame({"1-2":pos_effect_sizes_12.values(), "2-3":pos_effect_sizes_23.values(), "1-3":pos_effect_sizes_13.values()}, index=pos_effect_sizes_23.keys())

lemma_freqs = bdf.getLemmaFrequencies(books)

word_freqs_1 = bdf.getWordFrequencies(sub_1)
word_freqs_2 = bdf.getWordFrequencies(sub_2)
word_freqs_3 = bdf.getWordFrequencies(sub_3)

word_amounts_1 = bdf.getTokenAmounts(sub_1)
word_amounts_2 = bdf.getTokenAmounts(sub_2)
word_amounts_3 = bdf.getTokenAmounts(sub_3)

print("Processing type-token-ratios...")

ttrs_1 = bdf.getTypeTokenRatios(word_freqs_1, word_amounts_1)
ttrs_2 = bdf.getTypeTokenRatios(word_freqs_2, word_amounts_2)
ttrs_3 = bdf.getTypeTokenRatios(word_freqs_3, word_amounts_3)

ttr_effect_size_12 = bdf.cohensdForSubcorps(ttrs_1.to_dict(), ttrs_2.to_dict())
ttr_effect_size_23 = bdf.cohensdForSubcorps(ttrs_2.to_dict(), ttrs_3.to_dict())
ttr_effect_size_13 = bdf.cohensdForSubcorps(ttrs_1.to_dict(), ttrs_3.to_dict())

ttr_effect_size = pd.DataFrame({"1-2":ttr_effect_size_12, "2-3":ttr_effect_size_23, "1-3":ttr_effect_size_13}, index=['TTR'])

effect_sizes = pd.concat([ttr_effect_size, deprel_effect_sizes, feats_effect_sizes, pos_effect_sizes])

print("Writing to Excel...")

with pd.ExcelWriter("Data/KeyFactorAnalysis.xlsx") as writer:
    effect_sizes.to_excel(writer, sheet_name="Effect sizes")

print("Done!")

#word_amounts = bdf.getTokenAmounts(books)
#
#l = bdf.getL(word_amounts)
#
#f_lemma = bdf.combineFrequencies(lemma_freqs)
#
#lemma_zipfs = bdf.getZipfValues(l, f_lemma)
#
#print(lemma_zipfs)
#
#ax = lemma_zipfs.plot.hist()
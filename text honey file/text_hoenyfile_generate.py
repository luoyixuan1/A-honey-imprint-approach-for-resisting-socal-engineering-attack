from gensim.models import word2vec
import pandas as pd
import spacy
import pytextrank
import textstat
import matplotlib.pyplot as plt
import scienceplots


model = word2vec.Word2Vec.load('word2vec.model')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
file = open('stop_word.txt')
stop_word = file.read().split('\n')
file.close()


def abstract_keyword(text):
    doc = nlp(text)
    ls = []
    for phrase in doc._.phrases:
        word = phrase.text

        if (len(word.split()) == 1) and (word not in stop_word):
            ls.append(word)
    return ls


def search_word(word):
    try:
        result = model.wv.most_similar(word, topn=10)
        return result[1][0], result[1][1]
    except:
        return '',  -1


def main():
    file = pd.read_csv('essay.csv', header=0, encoding_errors='ignore')
    essays = file['essay']
    essays = essays.tolist()
    length = len(essays)

    new_essays = []
    FKG_old = []
    FKG_new = []
    FRE_old = []
    FRE_new = []
    DCR_old = []
    DCR_new = []
    ARI_old = []
    ARI_new = []
    CLI_old = []
    CLI_new = []
    GFI_old = []
    GFI_new = []
    LW_old = []
    LW_new = []
    for i in range(length):
        text = essays[i]
        words = abstract_keyword(text)
        similar_words = []
        for item in words:
            new_word, weight = search_word(item)
            similar_words.append((item, new_word, weight))
        similar_words.sort(key=lambda x: -x[-1])
        new_text = text
        for item in similar_words:
            if item[-1] > 0.7:
                new_text = new_text.replace(item[0], item[1])
            else:
                break
        new_essays.append(new_text)

        FKG_old.append(textstat.flesch_kincaid_grade(text))
        FKG_new.append(textstat.flesch_kincaid_grade(new_text))

        FRE_old.append(textstat.flesch_reading_ease(text))
        FRE_new.append(textstat.flesch_reading_ease(new_text))

        DCR_old.append(textstat.dale_chall_readability_score(text))
        DCR_new.append(textstat.dale_chall_readability_score(new_text))

        ARI_old.append(textstat.automated_readability_index(text))
        ARI_new.append(textstat.automated_readability_index(new_text))

        CLI_old.append(textstat.coleman_liau_index(text))
        CLI_new.append(textstat.coleman_liau_index(new_text))

        GFI_old.append(textstat.gunning_fog(text))
        GFI_new.append(textstat.gunning_fog(new_text))

        LW_old.append(textstat.linsear_write_formula(text))
        LW_new.append(textstat.linsear_write_formula(new_text))

    result = {
        'essays': essays,
        'Honey_file': new_essays,
        'essays_FKG': FKG_old,
        'Honey_file_FKG': FKG_new,
        'essays_FRE': FRE_old,
        'Honey_file_FRE': FRE_new,
        'essays_DCR': DCR_old,
        'Honey_file_DCR': DCR_new,
        'essays_ARI': ARI_old,
        'Honey_file_ARI': ARI_new,
        'essays_CLI': CLI_old,
        'Honey_file_CLI': CLI_new,
        'essays_GFI': GFI_old,
        'Honey_file_GFI': GFI_new,
        'essays_LW': LW_old,
        'Honey_file_LW': LW_new
 }

    results = pd.DataFrame(result)
    results.to_csv('results.csv')


if __name__ == '__main__':
    main()

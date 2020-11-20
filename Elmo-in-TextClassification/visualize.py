import pandas as pd
from collections import Counter
from wordcloud_fa import WordCloudFa
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from matplotlib import pyplot as plt
from util import reshape_persiantext_for_display
from TextRank import TextRank4Keyword
import math

from collections import OrderedDict


def create_word_clod(all_cluster_hashtags,path):
    '''

    :param all_cluster_hashtags: something like :list of Counter() ,each Counter have hashtags with
     the number of that hashtag in that cluster

    :return:
    '''
    for i,hashtags in enumerate(all_cluster_hashtags):
        wodcloud = WordCloudFa()
        wc = wodcloud.generate_from_frequencies(dict(hashtags.most_common()[0:5]))
        image = wc.to_image()
        # image.show()
        image.save('{0}/cluster_{1}.png'.format(path,i))


def visual_cluster_by_sentiment(path,sentiment_list):
    all_cluster_hashtags =[]
    for cluster in sentiment_list:

        all_cluster_hashtags.append(Counter(cluster))
    create_bar_plot(all_cluster_hashtags, path)


def visual_cluster_by_hashtags(path,text_hashtags,textlist):
    all_cluster_hashtags =[]
    for cluster in text_hashtags:
        cluster_hashtags=[]
        texts_with_no_hashtag=[]
        for i,text in enumerate(cluster):
            hashtags=text.strip().strip('""').strip('\n')
            if str(hashtags).find('-')!= -1:
                hashs=hashtags.split('-')
                hashs=[h.strip() for h  in hashs ]
                cluster_hashtags.extend(hashs)
            else:
                texts_with_no_hashtag.append(textlist[i])
        if len(texts_with_no_hashtag)>0:
            cluster_hashtags.extend(extract_keyword_for_textlist_bytf(texts_with_no_hashtag))
            print(len(texts_with_no_hashtag))
        all_cluster_hashtags.append(Counter(cluster_hashtags))

    # print(all_cluster_hashtags)
    # create_word_clod(all_cluster_hashtags,path)
    create_bar_plot(all_cluster_hashtags, path)

def visual_cluster_by_keyword(text_list,path):
    all_cluster_keyword = []
    for cluster in text_list:
        cluster_keys = extract_keyword_for_textlist(cluster)
        all_cluster_keyword.append(Counter(cluster_keys))
    print(all_cluster_keyword)
    # create_word_clod(all_cluster_keyword,path)
    create_bar_plot(all_cluster_keyword,path)
def create_bar_plot(all_cluster_keyword,path):
    for index,cluster in enumerate(all_cluster_keyword):
        plot_bar_x(cluster,index,path)
    pass

def plot_bar_x(cluster_key_dist,title,path):
    cluster_key_dist=dict(cluster_key_dist)
    dd=OrderedDict(sorted(cluster_key_dist.items(),key=lambda x:x[1],reverse=True))

    labels=list(dd.keys())[0:len(dd.keys()) if len(dd.keys())<30 else 30]
    dists=list(dd.values())[0:len(dd.keys()) if len(dd.keys())<30 else 30]

    index = np.arange(len(labels))
    plt.figure(figsize=(20, 15))
    plt.bar(index, dists)
    plt.xlabel(reshape_persiantext_for_display('کلمه(یا هشتگ)'), fontsize=10)
    plt.ylabel(reshape_persiantext_for_display('تعداد تکرار'), fontsize=10)
    plt.xticks(index, [reshape_persiantext_for_display(label) for label in labels], fontsize=12, rotation=80)
    plt.title(reshape_persiantext_for_display('پر تکرارترین کلمات در کلاستر{0}').format(title))

    plt.savefig('{0}/clc_{1}'.format(path,title))
    plt.close()
    # plt.show()

def get_stop_list(stop_file_path):
    with open(stop_file_path,'r',encoding='utf-8') as f:
        stopwords=f.readlines()
        stop_set=set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def extract_keyword_for_textlist(text_list):

    tokens=[]
    stops=get_stop_list('stoplist.txt')
    for text in text_list:
        # all_tokens=text.split()
        # informative_tokens=[token for token in all_tokens if
        #                     token.strip().replace('.','').replace(',','').replace(':','') not in stops]
        tr4w = TextRank4Keyword()
        tr4w.analyze(text, candidate_pos=['N', 'Ne', 'Aj', 'Aje'], window_size=3, stemed=False)
        standard_key_num = int(2.21 * math.log(len(text.split())) - 3.43)
        keys=list(tr4w.get_keywords())
        if standard_key_num==0 :
            if len(keys)>0:
                informative_tokens=keys
        else:
            informative_tokens=keys[0:standard_key_num]
        tokens.extend(informative_tokens)
    return tokens



def extract_keyword_for_textlist_bytf(text_list):

    tokens=[]
    stops=get_stop_list('stoplist.txt')
    for text in text_list:
        all_tokens=text.split()
        informative_tokens=[clean(token) for token in all_tokens if
                            clean(token) not in stops
                            and len(clean(token))>1
                            and not str(token).startswith('@')]

        tokens.extend(informative_tokens)
    return tokens
def clean(token):
    return  token.replace('؟', ' ').\
        replace('—',' ').\
        replace('!', ' ').\
        replace('#', ' ').\
        replace('۰',' ').\
        replace('.',' ').\
        replace(',',' ').\
        replace(':',' ').\
        replace('?',' ').\
        replace('"',' ').\
        replace('_',' ').\
        replace('-',' ').\
        replace('=',' ').\
        replace('(',' '). \
        replace(')', ' ').replace('*', ' ').replace('*', ' ').replace('$', ' ').replace('%',' ').\
        strip()


def extract_keyword_for_textlist_bytfidf(text_list):

    tokens=[]
    stops=get_stop_list('stoplist.txt')
    for text in text_list:
        all_tokens=text.split()
        informative_tokens=[token.replace('.','').replace(',','').replace(':','').replace('?','').replace('"','') for token in all_tokens if
                            token.strip().replace('.','').replace(',','').replace(':','').replace('?','') not in stops
                            and len(token.replace('.','').replace(',','').replace(':','').replace('?','').replace('"',''))>2]

        tokens.extend(informative_tokens)
    return tokens

def sort_coo(coo_matrix):
    tuples=zip(coo_matrix.col,coo_matrix.data)
    return sorted(tuples,key=lambda x:(x[1],x[0]),reverse=True)


def extract_topn_from_vector(feature_names,sorted_items,topn=10):
    sorted_items=sorted_items[:topn]
    score_vals=[]
    feature_vals=[]
    for idx,score in sorted_items:
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
    results={}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def visual_cluster_by_hashtags_TFIDF(path,text_hashtags,textlist):
    all_cluster_hashtags =[]
    for i,cluster in enumerate(text_hashtags):
        # ----------------------------tfidf---------------------------
        cluster_text_tokens=[]

        texts_with_no_hashtag=[]
        for j,text in enumerate(cluster):
            hashtags=text.strip().strip('""').strip('\n')
            if str(hashtags).find('-')!= -1:
                hashs=hashtags.split('-')

                cluster_text_tokens.append(hashs)
            else:

                if len(textlist[i][j])>0:
                    cluster_text_tokens.append(extract_keyword_for_textlist_bytf([textlist[i][j]]))

        # -----------------------------calculate fidf for hashtags---------------------------
        # cluster_text_tokens is list of list of tokens or hashtags for each text in current cluster

        tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        tfidf_matrix = tfidf.fit_transform(cluster_text_tokens)
        df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())
        tokens=df.columns
        keys = {}
        for token in tokens:
            keys[token]=np.mean(df[token])


        # --------------------------------------------------------

        all_cluster_hashtags.append(keys)


    create_bar_plot(all_cluster_hashtags, path)






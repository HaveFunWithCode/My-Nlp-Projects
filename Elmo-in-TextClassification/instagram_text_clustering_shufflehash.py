import os

from ELMoForManyLangs.elmoformanylangs import Embedder
from hazm import sent_tokenize
from hazm import word_tokenize
import numpy as np
import pandas as pd
from visualize import  visual_cluster_by_hashtags_TFIDF
import util

from sklearn.cluster import KMeans, Birch
import matplotlib.pyplot as plt




X=[]
X_Word_embeding=[]
X_LSTM_1=[]
X_LSTM_2=[]
X_avg=[]
def _save_embeddings_parts(X_Word_embeding,
                           X_LSTM_1,
                           X_LSTM_2,
                           X_LSTM_3,
                           X_LSTM_4,
                           X_avg,
                           postfix):

    store=pd.HDFStore('instaemb4l_shuffle/insta_embedding_{0}.h5'.format(postfix))
    store['X_Word_embeding']=pd.DataFrame(X_Word_embeding)
    store['X_LSTM_1']=pd.DataFrame(X_LSTM_1)
    store['X_LSTM_2']=pd.DataFrame(X_LSTM_2)
    store['X_LSTM_3']=pd.DataFrame(X_LSTM_3)
    store['X_LSTM_4']=pd.DataFrame(X_LSTM_4)
    store['X_avg']=pd.DataFrame(X_avg)
    # store['X_texts']=pd.DataFrame({'Texts':X_texts})


def _calculate_caption_embeding(captions_embs):
    sents_embeddings = []
    for sen in captions_embs:
        # avg over words in each sentence
        sents_embeddings.append(np.mean(sen, axis=0))
    # average over sents in a caption
    # caption_embedding = np.mean(sents_embeddings, axis=0)
    caption_embedding = np.max(sents_embeddings, axis=0)
    return caption_embedding

def _calculate_caption_embeding_concat(captions_embs):
    sents_embeddings = []
    for sen in captions_embs:
        # avg over words in each sentence
        sents_embeddings.append(np.concatenate([np.mean(sen, axis=0),
                                               np.max(sen, axis=0),
                                               np.min(sen, axis=0)]))
    # average over sents in a caption
    caption_embedding = np.mean(sents_embeddings, axis=0)
    return caption_embedding

# calculate caption embedding for dataset
def calculate_embeding(datatype,e):
    problems=[]
    c=0
    X_Word_embeding=[]
    X_LSTM_1=[]
    X_LSTM_2=[]
    X_LSTM_3=[]
    X_LSTM_4=[]
    X_avg=[]
    X_texts=[]
    hd5_capacity=1000
    with open('testdata/{0}/cleaned_captions.txt'.format(datatype)) as f:
    # with open('testdata/cluster_2.txt') as f:
    # with open('testdata/cleaned_captions.txt') as f:
        while True:
            c=c+1
            print(c)
            # if c==50:
            #     break

            sample = f.readline()
            if sample!='':

                sents = sent_tokenize(sample.replace('"','').replace('\n','').replace('#',' '))
                sents_tokens = [word_tokenize(sent) for sent in sents]
                try:
                    word_encoder = e.sents2elmo(sents_tokens, 0)
                    LSTM_hidden_1 = e.sents2elmo(sents_tokens, 1)
                    LSTM_hidden_2 = e.sents2elmo(sents_tokens, 2)
                    LSTM_hidden_3 = e.sents2elmo(sents_tokens, 3)
                    LSTM_hidden_4 = e.sents2elmo(sents_tokens, 4)
                    average_layers = e.sents2elmo(sents_tokens, -1)

                    # X_Word_embeding.append(_calculate_caption_embeding(word_encoder))
                    # X_LSTM_1.append(_calculate_caption_embeding(LSTM_hidden_1))
                    # X_LSTM_2.append(_calculate_caption_embeding(LSTM_hidden_2))
                    # X_avg.append(_calculate_caption_embeding(average_layers))
                    # X_texts.append(sample)

                    X_Word_embeding.append(_calculate_caption_embeding_concat(word_encoder))
                    X_LSTM_1.append(_calculate_caption_embeding_concat(LSTM_hidden_1))
                    X_LSTM_2.append(_calculate_caption_embeding_concat(LSTM_hidden_2))
                    X_LSTM_3.append(_calculate_caption_embeding_concat(LSTM_hidden_3))
                    X_LSTM_4.append(_calculate_caption_embeding_concat(LSTM_hidden_4))
                    X_avg.append(_calculate_caption_embeding_concat(average_layers))

                    if c%hd5_capacity==0:

                        postfix=str(int(c/hd5_capacity))

                        _save_embeddings_parts(X_Word_embeding,
                               X_LSTM_1,
                               X_LSTM_2,
                               X_LSTM_3,
                               X_LSTM_4,
                               X_avg,

                               postfix)
                        X_Word_embeding = []
                        X_LSTM_1 = []
                        X_LSTM_2 = []
                        X_LSTM_3 = []
                        X_LSTM_4 = []
                        X_avg = []

                except ZeroDivisionError:
                    problems.append(c-1)
                    print(sents_tokens)
                    continue
            else:
                break
    return problems
    # with open('instaemb/X_Word_embeding','a') as f:
    #     f.writable()

def load_data(embedding_type='X_avg'):
    '''
    :param embedding_type: X_Word_embeding|X_LSTM_1|X_LSTM_2|X_avg|X_texts
    :return:
    '''
    dfs=[]
    list_hd5s=os.listdir('instaemb4l_shuffle')

    for file in list_hd5s:
        store = pd.HDFStore('instaemb4l_shuffle/'+file)
        dfs.append(store[embedding_type])
        if len(dfs)>=100:
            break
    return pd.concat(dfs)



if __name__=='__main__':

    # --------------------------------------------4l experiment----------------------------------------------
    datatype = 'hash_shuffle10h'
    # datatype='hashtag3hlast'
    # datatype='hashtag3hmiddle'
    # datatype = 'justtext'
    e = Embedder('vecs/elmo-insta-4l/')
    problems=calculate_embeding(datatype)
    problems=[1558,9317]


    for num_of_cluster in [100,200,300,400]:
        for embtype in ['X_LSTM_1','X_LSTM_2','X_LSTM_3','X_avg','X_LSTM_4']:
            which_hashtag = 'fa_hashtag_list'
            # which_hashtag='main_hashtag_list'

            data = load_data(embedding_type=embtype)
            data.hist("0")
            plt.show()

            data = data.iloc[:, 1024:2048]
            # d_size=data.shape[0]

            output_path = 'clusteroutput/{0}/{1}/{2}/{3}'.format(datatype,
                                                                 num_of_cluster,
                                                                 which_hashtag,
                                                                 embtype + str(10000))

            # text=list(load_data('X_texts')['Texts'])
            with open('testdata/{0}/cleaned_captions.txt'.format(datatype)) as f:
                alltext = f.readlines()
            problems = problems[::-1]
            for problem_index in problems:
                del alltext[problem_index]

            with open('testdata/{0}/{1}.txt'.format(datatype, which_hashtag)) as f:

                allhashtags = f.readlines()
            for problem_index in problems:
                del allhashtags[problem_index]

            # brc = Birch(branching_factor=50, n_clusters=num_of_cluster, threshold=0.1, compute_labels=True)
            # brc.fit(data)
            # clusters = brc.predict(data)
            # labels = brc.labels_

            kmeans = KMeans(n_clusters=num_of_cluster)
            kmeans.fit(data)
            clusters = kmeans.predict(data)
            labels = kmeans.labels_

            text_with_same_cluster_list = []
            text_hashtags = []

            clusters = list(set(labels))

            for index, i in enumerate(clusters):
                cluster_i_indexes = [k for k, j in enumerate(labels) if j == i]
                text_with_same_cluster = [alltext[m] for m in cluster_i_indexes]
                text_with_same_cluster_list.append(text_with_same_cluster)
                text_hashtag = [allhashtags[m] for m in cluster_i_indexes]
                text_hashtags.append(text_hashtag)
                file_path = '{0}/cluster_{1}.txt'.format(output_path, i)
                util.folderpath_handler(file_path)
                with open(file_path, 'w') as f:
                    f.write('\n'.join(text_with_same_cluster))

            # visual_cluster_by_keyword(text_with_same_cluster_list,path=output_path)
            # create_word_clod(path=output_path, all_cluster_hashtags=text_hashtags)
            # visual_cluster_by_keyword(text_list=text_with_same_cluster_list,path=output_path)
            visual_cluster_by_hashtags_TFIDF(path=output_path, text_hashtags=text_hashtags,
                                             textlist=text_with_same_cluster_list)

# ----------------------------------------2l experiment ---------------------------------------------------
#     for num_of_cluster in range(0,100):
#         print(num_of_cluster)
#         for embtype in ['X_Word_embeding',
#                         'X_LSTM_1',
#                         'X_LSTM_2',
#                         'X_avg']:
#             for datatype in ['justHashtag']:
#             # for datatype in ['hashinsidehash','justHashtag']:
#                 # datatype='limit'
#                 # datatype = 'hashinsidehash'
#                 # datatype = 'justHashtag'
#
#                 calculate_embeding(datatype)
#                 # embtype='X_Word_embeding'
#                 # embtype='X_LSTM_1'
#                 # embtype='X_LSTM_2'
#                 # embtype = 'X_avg'
#                 # num_of_cluster = 12
#                 which_hashtag = 'fa_hashtag_list'
#                 # which_hashtag='main_hashtag_list'
#
#                 output_path = 'clusteroutput/{0}/{1}/{2}/{3}'.format(datatype, num_of_cluster, which_hashtag, embtype)
#                 data = load_data(embedding_type=embtype)
#                 # text=list(load_data('X_texts')['Texts'])
#                 with open('testdata/{0}/cleaned_captions.txt'.format(datatype)) as f:
#                     alltext = f.readlines()
#                 with open('testdata/{0}/{1}.txt'.format(datatype,which_hashtag)) as f:
#
#                     allhashtags = f.readlines()
#
#
#                 kmeans = KMeans(n_clusters=num_of_cluster)
#                 kmeans.fit(data)
#                 clusters = kmeans.predict(data)
#                 labels = kmeans.labels_
#
#                 text_with_same_cluster_list = []
#                 text_hashtags = []
#
#                 clusters = list(set(labels))
#
#                 for index, i in enumerate(clusters):
#                     cluster_i_indexes = [k for k, j in enumerate(labels) if j == i]
#                     text_with_same_cluster = [alltext[m] for m in cluster_i_indexes]
#                     text_with_same_cluster_list.append(text_with_same_cluster)
#                     text_hashtag = [allhashtags[m] for m in cluster_i_indexes]
#                     text_hashtags.append(text_hashtag)
#                     file_path = '{0}/cluster_{1}.txt'.format(output_path, i)
#                     util.folderpath_handler(file_path)
#                     with open(file_path, 'w') as f:
#                         f.write('\n'.join(text_with_same_cluster))
#
#                 # visual_cluster_by_keyword(text_with_same_cluster_list,path=output_path)
#                 # create_word_clod(path=output_path, all_cluster_hashtags=text_hashtags)
#                 visual_cluster_by_hashtags_TFIDF(path=output_path, text_hashtags=text_hashtags)
#
#
#
#
#

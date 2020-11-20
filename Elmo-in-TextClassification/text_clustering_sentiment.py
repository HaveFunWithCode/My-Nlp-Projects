import os

from ELMoForManyLangs.elmoformanylangs import Embedder
from hazm import sent_tokenize
from hazm import word_tokenize
import numpy as np
import pandas as pd
from visualize import  visual_cluster_by_hashtags
from visualize import  visual_cluster_by_keyword
from visualize import visual_cluster_by_sentiment
from visualize import create_word_clod
import util

from sklearn.cluster import Birch
from sklearn.cluster import KMeans

e = Embedder('vecs/elmo-intsa/')


X=[]
X_Word_embeding=[]
X_LSTM_1=[]
X_LSTM_2=[]
X_avg=[]
def _save_embeddings_parts(X_Word_embeding,
                           X_LSTM_1,
                           X_LSTM_2,
                           X_avg,
                           postfix):

    store=pd.HDFStore('sentiment/insta_embedding_{0}.h5'.format(postfix))
    store['X_Word_embeding']=pd.DataFrame(X_Word_embeding)
    store['X_LSTM_1']=pd.DataFrame(X_LSTM_1)
    store['X_LSTM_2']=pd.DataFrame(X_LSTM_2)
    store['X_avg']=pd.DataFrame(X_avg)
    # store['X_texts']=pd.DataFrame({'Texts':X_texts})


def _calculate_caption_embeding(captions_embs):
    sents_embeddings = []
    for sen in captions_embs:
        # avg over words in each sentence
        sents_embeddings.append(np.mean(sen, axis=0))
    # average over sents in a caption
    caption_embedding = np.mean(sents_embeddings, axis=0)
    return caption_embedding

# calculate caption embedding for dataset
def calculate_embeding(datatype):
    c=0
    X_Word_embeding=[]
    X_LSTM_1=[]
    X_LSTM_2=[]
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

            sents = sent_tokenize(sample)
            sents_tokens = [word_tokenize(sent) for sent in sents]
            try:
                word_encoder = e.sents2elmo(sents_tokens, 0)
                LSTM_hidden_1 = e.sents2elmo(sents_tokens, 1)
                LSTM_hidden_2 = e.sents2elmo(sents_tokens, 2)
                average_layers = e.sents2elmo(sents_tokens, -1)

                X_Word_embeding.append(_calculate_caption_embeding(word_encoder))
                X_LSTM_1.append(_calculate_caption_embeding(LSTM_hidden_1))
                X_LSTM_2.append(_calculate_caption_embeding(LSTM_hidden_2))
                X_avg.append(_calculate_caption_embeding(average_layers))
                # X_texts.append(sample)

                if c%hd5_capacity==0:

                    postfix=str(int(c/hd5_capacity))

                    _save_embeddings_parts(X_Word_embeding,
                           X_LSTM_1,
                           X_LSTM_2,
                           X_avg,

                           postfix)
                    X_Word_embeding = []
                    X_LSTM_1 = []
                    X_LSTM_2 = []
                    X_avg = []

            except ZeroDivisionError:
                print(sents_tokens)
                continue

    # with open('instaemb/X_Word_embeding','a') as f:
    #     f.writable()

def load_data(embedding_type='X_avg'):
    '''
    :param embedding_type: X_Word_embeding|X_LSTM_1|X_LSTM_2|X_avg|X_texts
    :return:
    '''
    dfs=[]
    list_hd5s=os.listdir('sentiment')

    for file in list_hd5s:
        store = pd.HDFStore('sentiment/'+file)
        dfs.append(store[embedding_type])
        if len(dfs)>=100:
            break
    return pd.concat(dfs)



if __name__=='__main__':


    datatype='sentiment'

    # calculate_embeding(datatype)

    # embtype='X_Word_embeding'
    # embtype='X_LSTM_1'
    # embtype='X_LSTM_2'
    # embtype = 'X_avg'
    # num_of_cluster =6
    which_hashtag = 'fa_hashtag_list'
    # which_hashtag='main_hashtag_list'
    for num_of_cluster in range(13,14):
        for embtype in ['X_Word_embeding','X_LSTM_1','X_LSTM_2','X_avg']:


            data = load_data(embedding_type=embtype)
            d_size = data.shape[0]

            output_path = 'clusteroutput/{0}/{1}/{2}/{3}'.format(datatype,
                                                                 num_of_cluster,
                                                                 which_hashtag,
                                                                 embtype + str(d_size))

            # text=list(load_data('X_texts')['Texts'])
            with open('testdata/{0}/cleaned_captions.txt'.format(datatype)) as f:
                alltext = f.readlines()[0:d_size]
            with open('testdata/{0}/{1}.txt'.format(datatype, which_hashtag)) as f:

                allhashtags = f.readlines()[0:d_size]
            with open('data/sent_class.txt'.format(datatype, which_hashtag)) as f:

                sentiments = f.readlines()[0:d_size]

            # brc = Birch(branching_factor=50, n_clusters=5, threshold=0.1, compute_labels=True)
            # brc.fit(data)
            #
            #

            # clusters = brc.predict(data)
            #
            # labels = brc.labels_

            kmeans = KMeans(n_clusters=num_of_cluster,init='k-means++', max_iter=1000, n_init=1)
            kmeans.fit(data)
            clusters = kmeans.predict(data)
            labels = kmeans.labels_

            text_with_same_cluster_list = []
            text_hashtags = []
            text_sents_list = []
            clusters = list(set(labels))

            for index, i in enumerate(clusters):
                cluster_i_indexes = [k for k, j in enumerate(labels) if j == i]
                text_with_same_cluster = [alltext[m] for m in cluster_i_indexes]
                text_with_same_cluster_list.append(text_with_same_cluster)
                text_hashtag = [allhashtags[m] for m in cluster_i_indexes]
                text_hashtags.append(text_hashtag)

                text_sents = [sentiments[m] for m in cluster_i_indexes]
                text_sents_list.append(text_sents)
                file_path = '{0}/cluster_{1}.txt'.format(output_path, i)
                util.folderpath_handler(file_path)
                with open(file_path, 'w') as f:
                    f.write('\n'.join(text_with_same_cluster))

            # visual_cluster_by_keyword(text_with_same_cluster_list,path=output_path)
            # create_word_clod(path=output_path, all_cluster_hashtags=text_hashtags)
            # visual_cluster_by_keyword(text_list=text_with_same_cluster_list,path=output_path)
            # visual_cluster_by_sentiment(path=output_path, sentiment_list=text_sents_list)
            visual_cluster_by_hashtags(path=output_path,text_hashtags=text_hashtags,textlist=text_with_same_cluster_list)

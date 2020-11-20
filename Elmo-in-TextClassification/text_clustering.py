import get_data
import numpy as np
from visualize import visual_cluster_by_hashtags
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from tensorflow.python.keras import backend as K
K._get_available_gpus()
seed = 8

seq_embeding_method='avg'
use_pca=True

x,y,embedding_length,cat_number,x_enc=get_data.getdataset(sequence_embeding_method=seq_embeding_method,_max_sent_length=50)
x=[a.tolist() for a in x]
y=[a.tolist() for a in y]
x=np.array(x)
y=np.array(y)
y=[int(np.nonzero(yy)[0]) for yy in y ]

# politics sport
mycats=[4,6]
cat_number=len(mycats)
cat_indexes = [k for k, j in enumerate(y) if j in mycats]
x=x[cat_indexes]
y=np.array(y)[cat_indexes]




brc = Birch(branching_factor=50, n_clusters=len(list(set(y))), threshold=0.1, compute_labels=True)
brc.fit(x)

clusters = brc.predict(x)

labels = brc.labels_

targets=list(set(y))

text_with_same_cluster_list=[]

clusters=list(set(labels))
for index,i in enumerate(clusters):
    cluster_i_indexes = [k for k, j in enumerate(labels) if j == i]
    text_with_same_cluster=[x_enc[m] for m in cluster_i_indexes]
    text_with_same_cluster_list.append(text_with_same_cluster)
    with open('clusteroutput/cluster_{0}.txt'.format(i),'w') as f:
        f.write('\n'.join(text_with_same_cluster))

visual_cluster_by_hashtags(text_with_same_cluster_list)





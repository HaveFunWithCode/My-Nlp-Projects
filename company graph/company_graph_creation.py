from sys import getsizeof
import psutil
from utils import *
import networkx as nx
import gzip
from pickle import Unpickler
from scipy.spatial import distance




ntopkeys =5
def createCompanyGraph(numfile):

    g = nx.Graph()
    # load keywords
    with open("Files/keywords.txt", 'r',encoding='utf-8') as f:
        keywords = f.read().split('\n')
    keywords = keywords[0:ntopkeys]


    with open("Files/shortest_path_to_keys.pkl", "rb") as prev_Scr:
        shortest_path_to_keys = Unpickler(prev_Scr).load()
    # create nodes
    for i in range(0, numfile):
        print(i)

        with gzip.open('Files/data/company' + str(i) + '.zip', 'rb') as f:
            # pickle.load("f")
            curdic1 = deserialize(f)
        # ----------------------
        c=0
        for key in curdic1:
            c=c+1
            # print(c)

            company = curdic1[key]
            keys_dists={}
            for word in keywords:
                dists=[]
                for s in company['subjects']:
                    if (removenoise(s),word) in shortest_path_to_keys:
                        dists.append(shortest_path_to_keys[(removenoise(s),word)])
                    else:
                        continue
                        # dists.append(-1)


                keys_dists[word]=numpy.array(dists).mean()
            keys_dists=[item[1] for item in sorted(keys_dists.items(),key=lambda s:s[0])]

            if len(keys_dists)<ntopkeys:
                print("error")
            # normalalize vector
            keys_dists=[(a - min(keys_dists)) / (max(keys_dists) - min(keys_dists)) for a in keys_dists]
            if 'nan' in [str(a) for a in keys_dists]:
                print(keys_dists)
                keys_dists=[]
            else:
                keys_dists=[a if a<0.5 else 1 for a in keys_dists]


            # g.add_node(company['companyid'],
            #            subject=[removenoise(s) for s in company['subjects']],
            #            nids=[a['nid'] for a in company['persons']],
            #            keys=keys_dists)
            g.add_node(company['companyid'],
                       subject=[removenoise(s) for s in company['subjects']],
                       keys=keys_dists)
            # print(len(g.nodes.keys()))

            print('mem usage:'+str(psutil.virtual_memory().percent))


    print('creatig edges')
    c=0
    print(getsizeof(g))
    try:
        for node1 in g.nodes:
            for node2 in g.nodes:
                print(getsizeof(g))
                print('mem usage:' + str(psutil.virtual_memory().percent))
                c=c+1
                print(str(c+1))
                if node1!= node2:
                    if len(g.nodes.get(node1)['keys'])==len(g.nodes.get(node2)['keys'])==ntopkeys:
                        try:
                            dist=distance.euclidean(g.nodes.get(node1)['keys'], g.nodes.get(node2)['keys'])
                            common_keys = [a for a in list(set(g.nodes.get(node1)['subject']) & set(g.nodes.get(node2)['subject'])) if a in keywords]

                            g.add_weighted_edges_from([(node1,node2,round(dist,5))],common_keys='-'.join(common_keys))

                        except ValueError:
                            print(dist)

    except TypeError as  e:
        print(e)
    # normalize edges weight and remove edge with weight upper 0.5


    nx.write_gpickle(g,"Files/company_graph.gpickle")
    # nx.draw(g)
    return g


def normalize_edge():
    threshold=0.5
    g=nx.read_gpickle("Files/company_graph.gpickle")
    all_weights=[edg[2]['weight'] for edg in g.edges(data=True)]
    max_weight=max(all_weights)
    should_remove=[]
    for edg in g.edges(data=True):
        newweight=edg[2]['weight']/max_weight
        if newweight>threshold:
            should_remove.append((edg[0], edg[1]))
    for ed in should_remove:
        g.remove_edge(ed[0],ed[1])
    nx.write_gpickle(g,"Files/company_normaliazed.gpickle")



if __name__=="__main__":
    createCompanyGraph(numfile=2)
    normalize_edge()

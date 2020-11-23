import gzip

import pickle

import networkx as nx

from itertools import combinations

from utils import *


def create_word_graph(numfile):
    g = nx.Graph()
    c = 0
    for filendex in range(0,numfile):
        print ("file number :" + str(filendex))
        with gzip.open('Files/data/company' + str(filendex) + '.zip', 'rb') as f:

            curdic1 = deserialize(f)

        for key in curdic1:

            c=c+1
            print (c)
            # print("file number :"+str(findex)+" company "+str(c))
            # if c==100:
            #     break

            company = curdic1[key]
            subjects = company['subjects']
            subjects = [removenoise(s) for s in subjects]

            g.add_nodes_from(subjects)
            if len(subjects) > 3:
                for i in range(0, len(subjects) - 3):
                    new_edgs = list(combinations([subjects[i], subjects[i + 1], subjects[i + 2]], 2))
                    for new_edg in new_edgs:
                        if g.has_edge(new_edg[0], new_edg[1]):
                            g[new_edg[0]][new_edg[1]]['weight'] = g[new_edg[0]][new_edg[1]]['weight'] + 1
                        else:
                            g.add_weighted_edges_from([(new_edg[0], new_edg[1], 1)])

            elif len(subjects) > 1:
                if g.has_edge(subjects[0], subjects[1]):
                    g[subjects[0]][subjects[1]]['weight'] = g[subjects[0]][subjects[1]]['weight'] + 1
                else:

                    g.add_weighted_edges_from([(subjects[0], subjects[1], 1)])
            else:
                pass
                # print("single node")

    nx.write_gpickle(g,"Files/word_graph.gpickle")
    return g


def buil_graph_and_create_keyworkFile(nfile):
    # the number of your desired keys
    num_ok_keywords = 1000

    create_word_graph(numfile=nfile)
    # load saved word graph
    word_graph = nx.read_gpickle("Files/word_graph.gpickle")

    # return list of nodes degree in descending order
    keywords = sorted([a for a in list(word_graph.degree)], key=lambda x: x[1], reverse=True)[0:num_ok_keywords]
    # save keywords
    with open("Files/keywords-raw.txt", 'w',encoding='utf-8') as f:
        f.write('\n'.join([key[0] for key in keywords]))


def build_shortest_path():

    word_graph = nx.read_gpickle("Files/word_graph.gpickle")
    # load keywords
    with open("Files/keywords.txt", 'r',encoding='utf-8') as f:
        keywords = f.read().split('\n')
    # change edge weight from w to 1/w to give more attention to edge with more weight in shortest path
    for u, v, d in word_graph.edges(data=True):
        if 'weight' in d:
            d['weight'] = 1 / d['weight']



    # calculate shortest path from every node to every key
    shortest_path_to_keys = {}
    all_nodes = list(word_graph.nodes)
    c=0
    for node in all_nodes:
        c=c+1
        print(c)
        for key in keywords:
            try:
                shortest_path_to_keys[(node, key)] = nx.shortest_path_length(word_graph, node, key)
            except nx.exception.NetworkXNoPath:
                continue

            except nx.exception.NodeNotFound:
                continue
    # save shortest path from every word to keywords
    with open("Files/shortest_path_to_keys.pkl","wb")as f:
        pickle.dump(shortest_path_to_keys,f)


if __name__=="__main__":
    # Step1:
    buil_graph_and_create_keyworkFile(nfile=2)
    # Step2 filter the keyword then uncomment the next block -run filter.py file  and change


    # Step 3 build shortest path based on filterd keywords
    build_shortest_path()


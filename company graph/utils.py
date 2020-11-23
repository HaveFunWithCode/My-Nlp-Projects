import itertools
import re
import numpy
import io
import struct
import hazm

stems='''
صنعت	صنعتی	صنایع
کشاورزی
مدیریت	مدیریتی
بازرگانی
آموزش	آموزشی
تجارت	تجاری
پزشکی
بهداشتی
غذا	غذائی	موادغذایی	غذایی
مالی
خدمات	خدماتی
فرهنگی
عمران	عمرانی
بازاریابی
اطلاعات
شیمیایی
دانش	دانشگاه
انرژی
اقتصادی
ماشین
مخابرات	مخابراتی
پتروشیمی
اجرایی	اجرائی
تکنولوژی
فناوری
بانک	بانکی	بانکها
معادن	معدنی
دامپروری
نساجی
ارتباطات
دارو	دارویی	داروئی
فلزی	فلزات
نیروگاهی
پالایشگاهی
دامداری
کود
زراعی	مزارع
گمرکات
املاک
فروش
صادرات
واردات
توزیع
اجتماعی
نفت	نفتی
اینترنت
تولید	تولیدی
'''

space_codepoints ="[\u0020\u2000-\u200F\u2028-\u202F]"
def Mystemmer(token):
    words = stems.split("\n")
    word_stems = {}
    for w in words:
        all_word_shapes = w.split('\t')
        for shape in all_word_shapes:
            word_stems[shape] = all_word_shapes[0]

    if token in word_stems:
        return word_stems[token]
    else:
        return token
def removenoise(w):

    w = re.sub(space_codepoints,"",w)

    normalizer = hazm.Normalizer()
    wn=normalizer.normalize(w)
    w=wn


    w=w.replace('\u200c','').replace('\u200d','')
    return Mystemmer(w)


def deserialize(f):
    retval = {}
    while True:
        content = f.read(struct.calcsize('<L'))
        if not content: break
        k_len = struct.unpack('<L', content)[0]
        k_bstr = f.read(k_len)
        k = k_bstr.decode('utf-8')
        v_len = struct.unpack('<L', f.read(struct.calcsize('<L')))[0]
        v_bytes = io.BytesIO(f.read(v_len))
        v = numpy.load(v_bytes, allow_pickle=True)
        retval[k] = v.item()
    return retval


import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import community_detection as cds
import collections

fig_size=(20,20)
def get_centrality(G,i):
    reports=''

    degree_cent = nx.degree_centrality(G)
    reports+='-----------degree centrality--------------\n'
    print('-----------degree centrality--------------')
    for k, v in degree_cent.items():
        print(str.format('{0}:{1}', k, str(v)[0:6]))
        reports +=str.format('{0}:{1}', k, str(v)[0:6])+'\n'

    labels1 = {}
    best_nodes = [a[0] for a in sorted(degree_cent.items(), key=lambda kv: kv[1], reverse=True)[0:1]]
    for node in best_nodes:
        labels1[node] = node


    plt.figure(1,figsize=fig_size)
    plt.title("Degree Cetrality for component {0} \n The central node is:{1} ".format(i,best_nodes[0]))


    nx.draw_spring(G,
                   node_size=[int(v * 100) * 50 + 150 for v in degree_cent.values()],
                   node_color='Blue'
                   , cmap=plt.get_cmap('tab20b'),
                   with_labels=False,
                   edgecolors='yellow')
    pos=nx.spring_layout(G)
    nx.draw_networkx_labels(G, pos, labels1, font_size=10, font_color='#000000')
    plt.savefig("outputs/Degree Cetrality_for_component_{0}.png".format(i))

    between_cent = nx.betweenness_centrality(G)
    reports +='-----------between centrality--------------\n'
    print('-----------between centrality--------------\n')
    for k, v in between_cent.items():
        print(str.format('{0}:{1}', k, str(v)[0:6]))
        reports +=str.format('{0}:{1}', k, str(v)[0:6])+'\n'

    labels2 = {}
    best_nodes = [a[0] for a in sorted(between_cent.items(), key=lambda kv: kv[1], reverse=True)[0:1]]
    for node in best_nodes:
        labels2[node] = node

    plt.figure(2,figsize=fig_size)
    plt.title("Between Cetrality for component {0}\n The central node is: {1}".format(i,best_nodes[0]))



    nx.draw_spring(G,

                   node_size=[int(v * 100) * 50 + 400 for v in between_cent.values()],
                   cmap=plt.get_cmap('tab20b'),
                   with_labels=False,
                   edgecolors='yellow',
                   node_color='Red')
    pos = nx.spring_layout(G)
    nx.draw_networkx_labels(G, pos, labels2, font_size=10, font_color='#000000')
    plt.savefig("outputs/between_centrality_for_component_{0}.png".format(i))
    plt.show()
    return reports


def get_common_keyword_between_companis(g):
    common_keys=nx.get_edge_attributes(g, 'common_keys')
    allcompanies_common_keyword = [v.split('-') for k, v in common_keys.items() if len(v) > 0]
    allcompanies_common_keyword = list(itertools.chain(*allcompanies_common_keyword))
    most_common_keywords = collections.Counter(allcompanies_common_keyword).most_common(10)
    print("common keys=", '-'.join([a[0] for a in most_common_keywords]))

    return '-'.join([a[0] for a in most_common_keywords])

def get_common_subects(g):
    subjects = nx.get_node_attributes(g, 'subject')
    most_common_subjects = collections.Counter(list(itertools.chain(*subjects.values()))).most_common(10)
    most_common_subjects = [a[0] for a in most_common_subjects]
    return '-'.join(most_common_subjects)



def get_communities(g,i):

    reports=''
    partition = community_louvain.best_partition(g)
    pos = cds.community_layout(g, partition)

    values = [partition.get(node) for node in g.nodes()]
    plt.figure(figsize=(15, 15))
    plt.title('Community detection for component {0}'.format(i))

    reports+='\n'+40*'*'+'\n'+'Community detection for component {0}'+'\n'+40*'*'+'\n'
    nx.draw(g, node_color=values, with_labels=False, cmap=plt.get_cmap('tab20b'), pos=pos,edgecolors='yellow')

    plt.savefig("outputs/communities_for_component_{0}.png".format(i))
    plt.show()
    print('partitions based on louvain community detection are:\n')
    reports+='partitions based on louvain community detection are:\n'
    partition_nodes = dict()
    for k, v in partition.items():
        partition_nodes[v] = partition_nodes.get(v, [])
        partition_nodes[v].append(k)
    for key in partition_nodes.keys():
        print(str.format('partition ({0}):', key))
        reports += str.format('partition ({0}):', key) + '\n'
        print(partition_nodes.get(key))
        reports += '\n'.join(partition_nodes.get(key)) + '\n'



        this_comunity_companies=[str(c) for c in partition_nodes[key]]
        print(str.format('list of companies in partition ({0}):\n{1}\n',key,'\n'.join(this_comunity_companies)))
        reports+=str.format('list of companies in partition ({0}):\n{1}\n',key,'\n'.join(this_comunity_companies))

        # -----------most common keywords between companies in this component

        most_common_keywords = get_common_keyword_between_companis(g.subgraph(partition_nodes[key]))
        print("common keys=", most_common_keywords)
        reports += 40 * '*' + '\n'"most common keyword in  partition {0} are:\n".format(most_common_keywords)

        # -----------most common subjects between companies in this component
        most_common_subjects = get_common_subects(g.subgraph(partition_nodes[key]))
        print("common subjects=", most_common_subjects)
        reports += 40 * '*' + '\n'"most common subjects in  partition {0} are:\n".format(most_common_subjects)




    return reports


def plot_degree_hstogram(G,i):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("DegreeHistogram for component {0}".format(i))
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.xticks(rotation=90)
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.savefig("outputs/DegreeHistogram_for_component_{0}.png".format(i))
    plt.show()


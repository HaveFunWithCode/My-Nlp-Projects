import collections
import itertools

import networkx as nx
import matplotlib.pyplot as plt
import utils

# component with n node is giant
n = 5
g=nx.read_gpickle("Files/company_normaliazed.gpickle")
show_edge_label=False
show_node_label=False
reports=''

# -------------------------------------------extract giant components----------------------------

all_components = sorted([g.subgraph(c) for c in nx.connected_components(g)], key=len, reverse=True)
giants=[a for a in all_components if len(a)>n]
print('number of giant components is : {0}\n'.format(str(len(giants))))
reports+='number of giant components is : {0}\n'.format(str(len(giants)))
for i,g in enumerate(giants):

    pos = nx.spring_layout(g)

    companies='\n'.join([node for node in g.nodes])
    edge_labels = nx.get_edge_attributes(g, 'common_keys')

    # companies='\n'.join([str(a) for a in list(set(list(edge_labels.values())))])
    print(40*'*'+'\n'+"companies in  component {0} are:\n{1}".format(i,companies))
    reports+=40*'*'+'\n'"companies in  component {0} are:\n{1}".format(i,companies)
    # TODO: test
    # -----------most common keywords between companies in this component

    most_common_keywords=utils.get_common_keyword_between_companis(g)
    print("common keys=", most_common_keywords)
    reports += 40 * '*' + '\n'"most common keyword in  component {0} are:\n".format(most_common_keywords)

    # -----------most common subjects between companies in this component
    most_common_subjects=utils.get_common_subects(g)
    print("common subjects=", most_common_subjects)
    reports += 40 * '*' + '\n'"most common subjects in  component {0} are:\n".format(most_common_subjects)

    reports+=40*'*'+'\n'"node's degree in  component {0} are:\n".format(i)
    for gg in g.degree():

        print('({0},{1})'.format(gg[0],gg[1]))
        reports+='({0},{1})'.format(gg[0],gg[1])+'\n'
    plt.figure(figsize=(20, 20))
    plt.title("components{0}".format(i))


    if show_edge_label:
        nx.draw(g, pos, edgecolors='yellow', node_color='Green')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    elif show_node_label:
        nx.draw(g, pos, edgecolors='yellow', node_color='Green', with_labels=True)
    else:
        nx.draw(g, pos, edgecolors='yellow', node_color='Green', with_labels=False)




    # plt.text(0.02, 0.5,"companies in this component are:{0}".format(companies))

    plt.savefig("outputs/components{0}.png".format(i),dpi=200)

    # plt.show()
    # -------------------calculate centrality in each component---------------------------
    centrality_report = utils.get_centrality(g, i)
    reports += '\n' + centrality_report
    # --------------------community detection in each component ------------------------
    community_report = utils.get_communities(g, i)
    reports += '\n' + community_report
    # ------------------------plot_degree_histogram--------------------------------------
    utils.plot_degree_hstogram(g, i)
with open('outputs/reports.txt', 'w') as rep:
        rep.write(reports)
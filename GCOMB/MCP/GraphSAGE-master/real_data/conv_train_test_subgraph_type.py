import networkx as nx
import json
from networkx.readwrite import json_graph
import sys

dataset = sys.argv[1]
train_or_test = "test"
DATA_ROOT = f"../../../../data/{dataset}/"
input_file_path = f"{DATA_ROOT}/edges.txt"

id_map = {}
output_dir = f"{dataset}/{train_or_test}"
json_id_map_name = f"{output_dir}/large_graph-id_map.json"
UG = nx.Graph(name=dataset)
node_dic = {}
count = 0
print("json file name ", json_id_map_name)

print("input file of edges ", input_file_path)
file = open(input_file_path, 'r')

counter = 0
while True:
    counter += 1
    if counter % 1000 == 0:
        print("completed ", counter, " lines")
    line = file.readline()
    if len(line) < 2:
        break

    if line[0] == '#':
        continue

    edge = line.strip().split()
    source = int(edge[0].rstrip('\x00'))
    target = int(edge[1].rstrip('\x00'))
    if source == target:
        continue

    if source not in node_dic:
        node_dic[source] = len(node_dic)

    if target not in node_dic:
        node_dic[target] = len(node_dic)

    id_map[str(node_dic[source])] = node_dic[source]
    id_map[str(node_dic[target])] = node_dic[target]

    UG.add_edge(node_dic[source], node_dic[target])

data1 = json_graph.node_link_data(UG)
s1 = json.dumps(data1)
print("dumping to ", dataset + '/' + train_or_test + "/large_graph-G.json")
file1 = open(dataset + '/' + train_or_test + "/large_graph-G.json", "w")
file1.write(s1)
file1.close()


iddata = json.dumps(id_map)
f2 = open(json_id_map_name, 'w')
f2.write(iddata)
f2.close()
print(json_id_map_name)

attribute_file_name = dataset + '/' + train_or_test + '/' + 'attribute.txt'
print("attribute ", attribute_file_name)
file_attribute = open(attribute_file_name, 'w')
file_attribute.write('n=' + str(len(data1['nodes'])) + '\n' + 'm=' + str(UG.number_of_edges()))
file_attribute.close()

import os
import sys

dataset = sys.argv[1]

for size_var in [50, 80, 90, 99]:
    train_or_test = "train" + str(size_var)
    input_file_path = dataset + "/" + train_or_test + "/edges.txt"

    os.system("mkdir -p " + dataset + "/" + train_or_test)
    edges_out_path = dataset + '/' + train_or_test + '/' + 'edges.txt'
    file_write = open(edges_out_path, 'w')
    node_dic = {}

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

        edge = line.strip().split()
        source = int(edge[0])
        target = int(edge[1])

        if source == target:
            continue

        if source not in node_dic:
            node_dic[source] = len(node_dic)

        if target not in node_dic:
            node_dic[target] = len(node_dic)

        file_write.write(f"{node_dic[source]} {node_dic[target]}\n")

    attribute_file_name = dataset + '/' + train_or_test + '/' + 'attribute.txt'
    file_attribute = open(attribute_file_name, 'w')
    file_attribute.write(f'n={len(node_dic)}\nm={counter}')

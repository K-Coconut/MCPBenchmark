import os
import sys

dataset = sys.argv[1]
weight_model = sys.argv[2]

command = "python main.py -k 50 -n 20 -e 1 -w 2 -r 0.0005 -p 1 -b 6 -d %s -m %s" % (dataset, weight_model)
print(command)

os.system(command)

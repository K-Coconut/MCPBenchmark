dataset=$1
weight_model=$2

cd GraphSAGE-master
python3 predict_multiple.py -d $dataset -m $weight_model
cd ..
python easy_testing.py -d $dataset -m $weight_model

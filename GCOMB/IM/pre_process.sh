DATASET=$1
# weight WEIGHT_MODEL
WEIGHT_MODEL=$2

mkdir -p log
echo "$1 pre start: $(date)" >log/pre_time_$1_$2.txt

# convert raw data to graph
cd GraphSAGE-master/real_data
# train
python convto_nx.py -d $DATASET -m $WEIGHT_MODEL -t train
python get_training_subgraph_size_var.py -d $DATASET -m $WEIGHT_MODEL
# test
python convto_nx.py -d $DATASET -m $WEIGHT_MODEL -t test

# get training label, only for training dataset, youtube
cd ../../
if [ $DATASET = "youtube" ]; then
  # validation set
  python convto_nx.py -d $DATASET -m $WEIGHT_MODEL -t validation
  python imm_create_train_datasets_one_only.py -m $WEIGHT_MODEL -d $DATASET -n 5
  python3 imm_conv_sol_marginal.py -m $WEIGHT_MODEL -d $DATASET
fi

# train interpolator
for size in 50 80 90 99; do
  python imm_create_train_datasets_one_only.py -d $DATASET -s $size -m $WEIGHT_MODEL
done

cd GraphSAGE-master
for k in 10 20 50 100 150 200; do
  python size_Var_rank_analysis_getlowest.py -d $DATASET -m $WEIGHT_MODEL -k $k
done

# get mc simulation graphs, only for training dataset, youtube
cd ..
if [ $DATASET = "youtube" ]; then
  echo "generating mc graphs on $DATASET"
  python spread_pre_process.py $DATASET $WEIGHT_MODEL
fi

echo "$1 pre end: $(date)" >>log/pre_time_$1_$2.txt

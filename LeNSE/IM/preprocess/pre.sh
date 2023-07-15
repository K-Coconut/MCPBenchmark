DATASET=$1
WEIGHT_MODEL=$2
TRAIN_SIZE=$3

if [[ -z "$TRAIN_SIZE" ]]; then
TRAIN_SIZE=20;
fi

python get_subgraph.py -d $DATASET -m $WEIGHT_MODEL -t train --point_proportion $TRAIN_SIZE
python get_subgraph.py -d $DATASET -m $WEIGHT_MODEL -t test --point_proportion $TRAIN_SIZE

# generate subgraphs for calculating influence spread
# python get_graph_sample_for_IC.py  -d $DATASET -m $WEIGHT_MODEL -t train

python get_scores_seeds_counts.py -d $DATASET -m $WEIGHT_MODEL -t train --point_proportion $TRAIN_SIZE

python fixed_size_dataset.py -d $DATASET -m $WEIGHT_MODEL -t train --point_proportion $TRAIN_SIZE

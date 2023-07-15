DATASET=$1
WEIGHT_MODEL=$2
TRAIN_SIZE=$3
gpu=2

if [[ -z "$TRAIN_SIZE" ]]; then
TRAIN_SIZE=20;
fi

python embedding_training.py --gpu $gpu --batch_size 32 -d $DATASET -m $WEIGHT_MODEL --embedding_size 128 --output_size 128 --point_proportion $TRAIN_SIZE
python prepare_data.py -d $DATASET -m $WEIGHT_MODEL --gpu $gpu --point_proportion $TRAIN_SIZE

# visualization
# python autoencoder.py -d $DATASET --input_size 128

python guided_exploration_training.py -d $DATASET -m $WEIGHT_MODEL --gpu $gpu --point_proportion $TRAIN_SIZE
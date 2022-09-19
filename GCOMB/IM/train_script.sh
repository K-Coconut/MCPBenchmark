dataset=$1
model=$2
echo "train start: $(date)" >train_time.txt
cd GraphSAGE-master
python train_multiple.py $dataset $model
python3 predict_multiple_for_train.py $dataset $model
cd ..
python train_RL.py $dataset $model
echo "train end: $(date)" >>train_time.txt

cd GraphSAGE-master

python3 predict_multiple_budgeted.py -d $1

cd ..
python easy_pred_mul.py -d $1


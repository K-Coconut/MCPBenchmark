mkdir -p log
echo "$1 pre start: $(date)" > log/pre_time_$1.txt

cd GraphSAGE-master/real_data

python get_subgraph.py $1

python var_training_subgraph.py $1

python create_bp.py $1

python create_bp_size_var.py $1

cd ../greedy_baseline/

python easy_sub_graphs_size_var.py $1
cd ..

python new_size_Var_rank_analysis_getlowest.py $1

cd ..
echo "$1 pre end: $(date)" >> log/pre_time_$1.txt
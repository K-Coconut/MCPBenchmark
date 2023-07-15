DATASET=$1
# train on BrightKite
python get_subgraph.py -d $DATASET
python get_scores_seeds.py -d $DATASET
python fixed_size_dataset.py -d $DATASET
python embedding_training.py -d $DATASET

# link other test dataset
# for d in amazon gowalla dblp youtube wiki_talk live_journal;
#     do bash data_link.sh $d; done;

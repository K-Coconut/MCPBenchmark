mkdir -p data

wget --no-check-certificate http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
mkdir data/youtube
gunzip com-youtube.ungraph.txt.gz
mv com-youtube.ungraph.txt data/youtube/edges.txt

wget --no-check-certificate https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz
mkdir data/gowalla
gunzip loc-gowalla_edges.txt.gz
mv loc-gowalla_edges.txt data/gowalla/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/loc-brightkite_edges.txt.gz
mkdir data/BrightKite
gunzip loc-brightkite_edges.txt.gz
mv loc-brightkite_edges.txt data/BrightKite/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
mkdir data/dblp
gunzip com-dblp.ungraph.txt.gz
mv com-dblp.ungraph.txt data/dblp/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz
mkdir data/amazon
gunzip com-amazon.ungraph.txt.gz
mv com-amazon.ungraph.txt data/amazon/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/wiki-Talk.txt.gz
mkdir data/wiki_talk
gunzip wiki-Talk.txt.gz
mv wiki-Talk.txt data/wiki_talk/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/wiki-topcats.txt.gz
mkdir data/wiki_topcats
gunzip wiki-topcats.txt.gz
mv wiki-topcats.txt data/wiki_topcats/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/higgs-social_network.edgelist.gz
mkdir data/higgs
gunzip higgs-social_network.edgelist.gz
mv higgs-social_network.edgelist data/higgs/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
mkdir data/pokec
gunzip soc-pokec-relationships.txt.gz
mv soc-pokec-relationships.txt data/pokec/edges.txt

wget --no-check-certificate https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz
mkdir data/orkut
gunzip com-orkut.ungraph.txt.gz
mv com-orkut.ungraph.txt data/orkut/edges.txt

wget --no-check-certificate http://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
mkdir data/live_journal
gunzip com-lj.ungraph.txt.gz
mv com-lj.ungraph.txt data/live_journal/edges.txt

wget --no-check-certificate  https://snap.stanford.edu/data/ca-CondMat.txt.gz
mkdir data/condmat
gunzip ca-CondMat.txt.gz
mv ca-CondMat.txt data/condmat/edges.txt

wget --no-check-certificate https://snap.stanford.edu/data/as-skitter.txt.gz
mkdir data/skitter
gunzip as-skitter.txt.gz
mv as-skitter.txt data/skitter/edges.txt

# LND dataset can be downloaded from https://drive.google.com/drive/folders/1jORG__SLxQIVk_DzZOgiUobzJDmKNJVp?usp=sharing

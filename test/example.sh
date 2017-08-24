
# Clean up (in case run previously)
rm -rf test.enrich_features.pickle test.enrich_features.pickle.log test.enrich_features.pickle_model.pickle proteins-to-predict.fasta_* model-stats.txt final-predictions.txt

# Calculate features from the enrichment file and train the model
pep_features test.enrich
pep_train    test.enrich_features.pickle -b 0

# Convert fasta file to kmers, calculate features, then predict binding
pep_kmerize proteins-to-predict.fasta
pep_features proteins-to-predict.fasta_kmers_3023.txt
pep_predict proteins-to-predict.fasta_kmers_3023.txt_features.pickle test.enrich_features.pickle_model.pickle

cp proteins-to-predict.fasta_kmers_3023.txt_features.pickle_test.enrich_features.pickle_model.pickle_predictions.txt final-predictions.txt
cp test.enrich_features.pickle.log model-stats.txt

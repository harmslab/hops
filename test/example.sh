# Clean up (in case run previously)
rm -rf test.enrich_features.pickle test.enrich_features.pickle.log test.enrich_features.pickle_model.pickle proteins-to-predict.fasta_* *.stats final-predictions.txt

# Calculate features from the enrichment file
pep_features test.enrich

# Train the model using k-fold cross validation
pep_train    test.enrich_features.pickle -b 0
pep_stats    test.enrich_features.pickle_model.pickle > working.stats

# At this point, you might tweak the input to pep_train to try to improve your
# model (e.g. change the number of estimators, breaks, etc.).  Once you are 
# happy with the tweaks, you can compare against the actual test set with:
# with the model, you can compare against the actual test-set with:
pep_stats   test.enrich_features.pickle_model.pickle -f > final.stats

# Convert fasta file to kmers, calculate features, then predict binding
pep_kmerize proteins-to-predict.fasta
pep_features proteins-to-predict.fasta_kmers_3023.txt
pep_predict proteins-to-predict.fasta_kmers_3023.txt_features.pickle test.enrich_features.pickle_model.pickle

cp proteins-to-predict.fasta_kmers_3023.txt_features.pickle_test.enrich_features.pickle_model.pickle_predictions.txt final-predictions.txt

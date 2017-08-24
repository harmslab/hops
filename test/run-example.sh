# Clean up (in case run previously)
rm -f *.features *.stats *.predictions *.model

# Calculate features from the enrichment file
pep_features test.enrich -o test.features

# Train the model using k-fold cross validation
pep_train test.features -o test.model 
pep_stats test.model > working.stats

# At this point, you might tweak the input to pep_train to try to improve your
# model (e.g. change the number of estimators, breaks, etc.).  Once you are 
# happy with the tweaks, you can compare against the actual test set with:
# with the model, you can compare against the actual test-set with:
pep_stats test.model -f > final.stats

# Convert fasta file to kmers, calculate features, then predict binding
pep_kmerize proteins-to-predict.fasta -o to_predict

pep_features to_predict_3023.kmers -o to_predict.features
pep_predict to_predict.features test.model -o final.predictions


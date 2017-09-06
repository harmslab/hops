# Clean up (in case run previously)
rm -f *.features *.stats *.predictions *.model

# Calculate features from the enrichment file
hops_features test.enrich -o test.features

# Train the model using k-fold cross validation
hops_train test.features -o test.model -b 0    # train to classify as above or below 0
#hops_train test.features -o test.model   # uncomment to train for continuous variable
hops_stats test.model > working.stats

# At this point, you might tweak the input to hops_train to try to improve your
# model (e.g. change the number of estimators, breaks, etc.).  Once you are 
# happy with the tweaks, you can compare against the actual test set with:
# with the model, you can compare against the actual test-set with:
hops_stats test.model -f > final.stats

# Convert fasta file to kmers, calculate features, then predict binding
hops_kmerize proteins-to-predict.fasta -o to_predict

hops_features to_predict_3023.kmers -o to_predict.features
hops_predict to_predict.features test.model -o final.predictions
hops_pred_to_fasta proteins-to-predict.fasta final.predictions -o predicted-proteins.txt

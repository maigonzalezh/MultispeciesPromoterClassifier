#!/bin/bash

# Generation
python3 ./scripts/preprocess/preprocess.py

# Filtering

cd-hit-est -i ./data/processed/pmd_promoters.fasta -o ./temp/pmd_promoters_filtered.fasta -c 0.8 -n 5 -T 0 -M 0

# Cross Filtering
# CDS

cd-hit-est -i ./data/processed/cds/sample.fasta -o ./temp/cds_samples_filtered.fasta -c 0.8 -n 5 -T 0 -M 0 && \
cd-hit-est-2d -i ./temp/pmd_promoters_filtered.fasta -i2 ./temp/cds_samples_filtered.fasta -o ./temp/cds_cross_filtered.fasta -c 0.8 -n 5 -T 0 -M 0

# RANDOM (SRS)
cd-hit-est -i ./data/processed/random/sample.fasta -o ./temp/random_samples_filtered.fasta -c 0.8 -n 5 -T 0 -M 0 && \
cd-hit-est-2d -i ./temp/pmd_promoters_filtered.fasta -i2 ./temp/random_samples_filtered.fasta -o ./temp/random_cross_filtered.fasta -c 0.8 -n 5 -T 0 -M 0

# Save samples
cp ./temp/pmd_promoters_filtered.fasta ./data/processed/pmd_promoters_filtered.fasta
cp ./temp/cds_cross_filtered.fasta ./data/processed/cds/sample_cross_filtered.fasta
cp ./temp/random_cross_filtered.fasta ./data/processed/random/sample_cross_filtered.fasta

# Merge
python3 ./scripts/preprocess/ensemble.py
#!/bin/bash
# python3 ./scripts/preprocess/preprocess.py

cd-hit-est -i ./data/processed/pmd_promoters.fasta -o ./temp/pmd_promoters_filtered.fasta -c 0.8 -n 5 -T 0 -M 0

# filtering and cross filtering
# CDS

cd-hit-est -i ./data/processed/cds/sample.fasta -o ./temp/cds_samples_filtered.fasta -c 0.8 -n 5 -T 0 -M 0 && \
cd-hit-est-2d -i ./temp/pmd_promoters_filtered.fasta -i2 ./temp/cds_samples_filtered.fasta -o ./temp/cds_cross_filtered.fasta -c 0.8 -n 5 -T 0 -M 0


# # RANDOM FIXED
cd-hit-est -i ./data/processed/random/sample.fasta -o ./temp/random_samples_filtered.fasta -c 0.8 -n 5 -T 0 -M 0 && \
cd-hit-est-2d -i ./temp/pmd_promoters_filtered.fasta -i2 ./temp/random_samples_filtered.fasta -o ./temp/random_cross_filtered.fasta -c 0.8 -n 5 -T 0 -M 0

cp ./temp/pmd_promoters_filtered.fasta ./data/processed/pmd_promoters_filtered.fasta
cp ./temp/cds_cross_filtered.fasta ./data/processed/cds/sample_cross_filtered.fasta
cp ./temp/random_cross_filtered.fasta ./data/processed/random/sample_cross_filtered.fasta

python3 ./scripts/preprocess/ensemble.py
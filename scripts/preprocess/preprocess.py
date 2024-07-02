import pandas as pd
from collections import Counter
from utils import (gc_percentage, create_fasta_records, save_fasta_records,
                   create_cds_sequences)
import os

RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"

# PPD Promoters data from http://lin-group.cn/database/ppd/download.php
PPD_RAW_DATA_PATH = f'{RAW_DATA_PATH}/ppd'


def join_promoters():
    species = os.listdir(PPD_RAW_DATA_PATH)
    df = pd.DataFrame()

    for specie in species:
        specie_df = pd.read_csv(
            f'{PPD_RAW_DATA_PATH}/{specie}').dropna(subset=['PromoterSeq'])
        specie_df['SpeciesName'] = specie_df['SpeciesName'][0]
        df = pd.concat([df, specie_df])

    df = df.iloc[df.groupby('SpeciesName').PromoterName.transform(
        'size').argsort(kind='mergesort')][::-1]

    return df


def process_promoters(joined_promoters):
    df = joined_promoters.copy()
    df = df.rename(columns={"PromoterSeq": "Sequence"})

    df['Sequence'] = df.apply(lambda x: x['Sequence'].lower(), axis=1)
    df['GCPercentage'] = df.apply(
        lambda x: gc_percentage(x['Sequence']), axis=1)
    df['SequenceLength'] = df.apply(lambda x: len(x['Sequence']), axis=1)

    df = df[df['SequenceLength'] == 81]
    df = df[df['Sequence'].str.contains('^[acgt]*$')]
    df = df.drop(columns=['SequenceLength'])
    df = df.reset_index(drop=True)

    return df


# PPD Promoters preprocessing
all_promoters = join_promoters()
all_promoters = process_promoters(all_promoters)

all_promoters.to_csv(f'{PROCESSED_DATA_PATH}/all_promoters.csv', index=False)

pseudomonadota_species = ['Acinetobacter baumannii ATCC 17978',
                          'Agrobacterium tumefaciens str C58',
                          'Bradyrhizobium japonicum USDA 110',
                          'Burkholderia cenocepacia J2315',
                          'Escherichia coli str K-12 substr. MG1655',
                          'Klebsiella aerogenes KCTC 2190',
                          'Pseudomonas putida strain KT2440',
                          'Shigella flexneri 5a str. M90T',
                          'Sinorhizobium meliloti 1021',
                          'Xanthomonas campestris pv. campestrie B100']

pmd_promoters = all_promoters[all_promoters['SpeciesName'].isin(
    pseudomonadota_species)]

pmd_promoters.to_csv(f'{PROCESSED_DATA_PATH}/pmd_promoters.csv', index=False)

all_promoters_fasta_filename = f'{PROCESSED_DATA_PATH}/all_promoters.fasta'
pmd_promoters_fasta_filename = f'{PROCESSED_DATA_PATH}/pmd_promoters.fasta'
pmd_promoters_fasta_records = create_fasta_records(pmd_promoters)
all_promoters_fasta_records = create_fasta_records(all_promoters)

save_fasta_records(all_promoters_fasta_records, all_promoters_fasta_filename)
save_fasta_records(pmd_promoters_fasta_records, pmd_promoters_fasta_filename)

n_sequences = round(1.5 * len(pmd_promoters))
seq_length = 81
min_gc = 25
max_gc = 70

# Random negative preprocessing

# create_random_sequences(
#     n_sequences, seq_length, min_gc, max_gc)

# Random (fixed) negative preprocessing
# create_random_sequences_fixed(pmd_promoters, seq_length)

# CDS negative preprocessing
cds_sequences_df = create_cds_sequences(pmd_promoters)

print('Preprocessing done!')

from Bio import SeqIO
import pandas as pd
import numpy as np

np.random.seed(42)
PROCESSED_DATA_PATH = "./data/processed"


def fasta_to_csv(fasta_filename, label=0):
    records = list(SeqIO.parse(fasta_filename, "fasta"))

    headers = [record.description.split(' ', 1)[1] for record in records]
    specie_names = [header.split('|', 1)[0].split('=', 1)[1]
                    for header in headers]
    cds_ids = [header.split('|', 1)[1] for header in headers]

    sequences = [str(record.seq) for record in records]

    df = pd.DataFrame({
        'id': cds_ids,
        'SpeciesName': specie_names,
        'Sequence': sequences,
        'label': label
    })

    return df


#random_fasta_filename = f'{PROCESSED_DATA_PATH}/random/sample_cross_filtered.fasta'
random_fasta_filename = f'{PROCESSED_DATA_PATH}/random/sample_cross_filtered.fasta'
cds_fasta_filename = f'{PROCESSED_DATA_PATH}/cds/sample_cross_filtered.fasta'
promoters_filename = f'{PROCESSED_DATA_PATH}/pmd_promoters_filtered.fasta'


#random_filtered_df = fasta_to_csv(random_fasta_filename, label=0)
random_fixed_filtered_df = fasta_to_csv(random_fasta_filename, label=0)
cds_filtered_df = fasta_to_csv(cds_fasta_filename, label=0)
promoters_df = fasta_to_csv(promoters_filename, label=1)

# TODO: Check name conversion. Replace Escherichia coli str K-12 substr. MG1655 for Escherichia coli str K-12 substr in promoters_df
# promoters_df['SpeciesName'] = promoters_df['SpeciesName'].replace(
#     'Escherichia coli str K-12 substr. MG1655', 'Escherichia coli str K-12 substr')

# random_fixed_filtered_df['SpeciesName'] = random_fixed_filtered_df['SpeciesName'].replace(
#     'Escherichia coli str K-12 substr. MG1655', 'Escherichia coli str K-12 substr')

species = promoters_df['SpeciesName'].unique()
dataset_cds_df = promoters_df.copy()
dataset_random_df = promoters_df.copy()
dataset_random_fixed_df = promoters_df.copy()

for index, specie in enumerate(species):
    #print("specie", specie)
    specie_promoter_count = promoters_df[promoters_df['SpeciesName']
                                         == specie].shape[0]

    cds_subsample_df = cds_filtered_df[cds_filtered_df['SpeciesName'] == specie].sample(
        n=specie_promoter_count, random_state=index + 1)
    # random_subsample = random_filtered_df.sample(
    #     n=specie_promoter_count, random_state=index + 1)
    random_fixed_subsample = random_fixed_filtered_df[random_fixed_filtered_df['SpeciesName'] == specie].sample(
        n=specie_promoter_count, random_state=index + 1)
    
    #print(random_fixed_subsample.head(2))
    # random_subsample['SpeciesName'] = specie
    random_fixed_subsample['SpeciesName'] = specie

    dataset_cds_df = pd.concat([dataset_cds_df, cds_subsample_df])
    # dataset_random_df = pd.concat([dataset_random_df, random_subsample])
    dataset_random_fixed_df = pd.concat(
        [dataset_random_fixed_df, random_fixed_subsample])


dataset_cds_df.to_csv(f'{PROCESSED_DATA_PATH}/cds/dataset.csv', index=False)
# dataset_random_df.to_csv(
#     f'{PROCESSED_DATA_PATH}/random/dataset.csv', index=False)
dataset_random_fixed_df.to_csv(
    f'{PROCESSED_DATA_PATH}/random/dataset.csv', index=False)

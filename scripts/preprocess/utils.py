import os
import csv
import pandas as pd
import numpy as np
import random
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from collections import Counter
from numpy.lib.stride_tricks import as_strided
from scipy.stats import gaussian_kde
import hashlib

RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"
TEMP_DATA_PATH = "./temp"


def generate_uuid(data, seed=42):
    hash_object = hashlib.md5(f'{seed}{data}'.encode())
    hash_hex = hash_object.hexdigest()
    uuid = f'{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}'
    return uuid


def get_substrings(text, window_size=81):
    if window_size <= 0 or window_size > len(text):
        raise ValueError(
            "The window size must be between 1 and the length of the text.")

    char_array = np.array(list(text), dtype='U1')

    new_shape = (len(text) - window_size + 1, window_size)
    new_strides = (char_array.strides[0], char_array.strides[0])

    strided_array = as_strided(
        char_array, shape=new_shape, strides=new_strides)

    substrings = [''.join(substring) for substring in strided_array]

    return substrings


def get_random_substrings(text, window_size, max_substrings=40):
    substrings = get_substrings(text, window_size)
    if len(substrings) > max_substrings:
        substrings = random.sample(substrings, max_substrings)
    return substrings


def gc_percentage(sequence_text):
    counts = Counter(sequence_text.lower())
    gc_count = counts['g'] + counts['c']
    total_count = counts['g'] + counts['c'] + counts['a'] + counts['t']
    return round((gc_count / total_count) * 100, 4)


def create_fasta_records(df):
    seq_records = []

    for index, row in df.iterrows():
        description = f'SpeciesName={row["SpeciesName"]}|id={row["id"]}'
        fasta_id = generate_uuid(f'{index}{description}')

        seq_record = SeqRecord(
            seq=Seq(row['Sequence']),
            id=fasta_id,
            description=description,
        )
        seq_records.append(seq_record)

    return seq_records


def save_fasta_records(fasta_records, filename):
    with open(filename, 'w') as f:
        SeqIO.write(fasta_records, f, 'fasta')


def create_raw_cds():
    species_cds = os.listdir(f'{RAW_DATA_PATH}/cds')
    print(species_cds)

    with open(f'{TEMP_DATA_PATH}/cds_raw.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "SpeciesName", "Sequence"])

    for specie in species_cds:
        specie_name = specie.split(".fna")[0]
        with open(f'{RAW_DATA_PATH}/cds/{specie}', "r") as handle:
            records = SeqIO.parse(handle, "fasta")

            with open(f'{TEMP_DATA_PATH}/cds_raw.csv', "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for record in records:
                    writer.writerow(
                        [record.id, specie_name, str(record.seq).lower()])

    return pd.read_csv(f'{TEMP_DATA_PATH}/cds_raw.csv')


def process_cds_sequences(cds_df):
    cds_df = cds_df[cds_df['Sequence'].str.len() >= 81]
    cds_df = cds_df.assign(Subsequence=cds_df.apply(lambda x: get_random_substrings(
        x['Sequence'], 81), axis=1)).explode('Subsequence')
    cds_df['GCPercentage'] = cds_df.apply(
        lambda x: gc_percentage(x['Subsequence']), axis=1)
    cds_df = cds_df.drop(columns=['Sequence'])
    cds_df = cds_df.rename(columns={'Subsequence': 'Sequence'})
    cds_df = cds_df[cds_df['Sequence'].str.contains('^[acgt]*$')]
    cds_df['Sequence'] = cds_df.apply(lambda x: x['Sequence'].lower(), axis=1)
    cds_df.reset_index(drop=True, inplace=True)

    return cds_df


def create_cds_sequences(promoters_df):
    np.random.seed(42)
    random.seed(42)

    df_species_cds = create_raw_cds()
    df_species_cds = process_cds_sequences(df_species_cds)

    all_species = promoters_df['SpeciesName'].unique()

    sample_cds_df = pd.DataFrame()

    for specie in all_species:
        cds_specie = df_species_cds[df_species_cds['SpeciesName'] == specie]
        promoters_per_specie = promoters_df[promoters_df['SpeciesName'] == specie]
        print(f"Specie: {specie} - Promoters: {len(promoters_per_specie)}")
        print(f"Specie: {specie} - CDS: {len(cds_specie)}")
        n_samples = round(2 * len(promoters_per_specie))
        sample_specie = cds_specie.sample(n_samples, random_state=42)
        sample_cds_df = pd.concat([sample_cds_df, sample_specie])

    # sample_cds_df = df_species_cds.sample(n_samples, random_state=42)

    if not os.path.exists(f'{PROCESSED_DATA_PATH}/cds'):
        os.makedirs(f'{PROCESSED_DATA_PATH}/cds')
    sample_cds_df.to_csv(
        f'{PROCESSED_DATA_PATH}/cds/sample.csv', index=False)

    cds_filename = f'{PROCESSED_DATA_PATH}/cds/sample.fasta'
    cds_fasta_records = create_fasta_records(sample_cds_df)
    save_fasta_records(cds_fasta_records, cds_filename)

    return df_species_cds


def create_random_sequence(length, GCPercentage):
    gc_count = int(length * GCPercentage / 100)
    at_count = length - gc_count
    sequence = ['g'] * (gc_count // 2) + ['c'] * (gc_count // 2) + \
        ['a'] * (at_count // 2) + ['t'] * (at_count // 2)

    sequence += ['g'] * (gc_count % 2) + ['a'] * (at_count % 2)
    np.random.shuffle(sequence)

    return ''.join(sequence)


def create_random_sequence_fixed(length, GCPercentage):
    gc_rounded = int(round(GCPercentage))

    at_rounded = 100 - gc_rounded

    g_percentage = 0 if gc_rounded == 0 else np.random.randint(0, gc_rounded)
    c_percentage = gc_rounded - g_percentage

    a_percentage = 0 if at_rounded == 0 else np.random.randint(0, at_rounded)

    g_count = int(length * g_percentage / 100)
    c_count = int(length * c_percentage / 100)
    a_count = int(length * a_percentage / 100)
    t_count = length - g_count - c_count - a_count

    sequence = 'g' * g_count + 'c' * c_count + 'a' * a_count + 't' * t_count
    sequence = ''.join(np.random.choice(list(sequence), length, replace=False))

    return sequence


def create_random_seq_list(n_sequences, length, min_gc=0, max_gc=100, gc_list=None, fixed=False):
    sequences = []
    GCPercentages = [np.random.uniform(min_gc, max_gc)
                     for _ in range(n_sequences)] if gc_list is None else gc_list

    for index, GCPercentage in enumerate(GCPercentages):
        sequence = create_random_sequence(length, GCPercentage) if not fixed \
            else create_random_sequence_fixed(length, GCPercentage)
        sequences.append(sequence)

    return sequences


def create_random_sequences(n_sequences, length, min_gc, max_gc):
    sequences = create_random_seq_list(n_sequences, length, min_gc, max_gc)
    df = pd.DataFrame(sequences, columns=["Sequence"])
    df['id'] = [generate_uuid(f'Random_{i}') for i in range(n_sequences)]
    df['SpeciesName'] = 'Random'
    df['GCPercentage'] = df.apply(
        lambda x: gc_percentage(x['Sequence']), axis=1)
    df = df[['id', 'SpeciesName', 'Sequence', 'GCPercentage']]
    df.reset_index(drop=True, inplace=True)

    df.to_csv(
        f'{PROCESSED_DATA_PATH}/random/sample.csv', index=False)

    random_filename = f'{PROCESSED_DATA_PATH}/random/sample.fasta'
    random_fasta_records = create_fasta_records(df)
    save_fasta_records(random_fasta_records, random_filename)

    return df


def create_random_sequences_fixed(promoters_df, seq_length):
    np.random.seed(42)
    random.seed(42)

    all_species = promoters_df['SpeciesName'].unique()

    df = pd.DataFrame()

    for specie in all_species:
        random_df = pd.DataFrame()
        promoters_per_specie = promoters_df[promoters_df['SpeciesName'] == specie]
        gc_values = promoters_per_specie['GCPercentage'].to_numpy()
        kde = gaussian_kde(gc_values)
        num_samples = len(gc_values)

        n_sequences = round(2 * num_samples)
        generated_gc = kde.resample(n_sequences)
        generated_gc = np.maximum(generated_gc[0], 0)
        generated_gc = np.round(generated_gc, 2).tolist()

        sequences = create_random_seq_list(
            n_sequences, seq_length, gc_list=generated_gc, fixed=True)

        random_df['id'] = [generate_uuid(
            f'Random_{i}_{specie}') for i in range(n_sequences)]
        random_df['SpeciesName'] = specie
        random_df['Sequence'] = sequences
        random_df['GCPercentage'] = random_df.apply(
            lambda x: gc_percentage(x['Sequence']), axis=1)

        df = pd.concat([df, random_df])

    df.reset_index(drop=True, inplace=True)

    if not os.path.exists(f'{PROCESSED_DATA_PATH}/random'):
        os.makedirs(f'{PROCESSED_DATA_PATH}/random')

    df.to_csv(f'{PROCESSED_DATA_PATH}/random/sample.csv', index=False)

    random_filename = f'{PROCESSED_DATA_PATH}/random/sample.fasta'
    random_fasta_records = create_fasta_records(df)
    save_fasta_records(random_fasta_records, random_filename)

    return df

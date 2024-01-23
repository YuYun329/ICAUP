#!scAuto_data-process
import argparse
import logging
import os
import sys

import anndata
import joblib
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def make_bed_seqs_from_df(input_bed, fasta_file, seq_len):
    """Return BED regions as sequences and regions as a list of coordinate
    tuples, extended to a specified length."""
    """Extract and extend BED sequences to seq_len."""

    seqs_dna = []
    fasta_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, 'fasta'))

    for i in tqdm(range(input_bed.shape[0]), desc="DNA sequence extract ..."):
        chrm = input_bed.iloc[i, 0]
        start = int(input_bed.iloc[i, 1])
        end = int(input_bed.iloc[i, 2])

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len

        # initialize sequence
        seq_dna = ""
        # add N's for left over reach
        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_record = fasta_dict[chrm]
        seq_dna += str(seq_record.seq[seq_start:seq_end]).upper()

        # add N's for right over reach
        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        assert len(seq_dna) == seq_len
        # append
        seqs_dna.append(seq_dna)

    return seqs_dna


def DNA_extract(adata, fa_path, seq_len=1344):
    var = adata.var
    peaks_info = var.loc[:, ['chr', 'start', 'end']]
    seqs_dna = make_bed_seqs_from_df(
        input_bed=var,
        fasta_file=fa_path,
        seq_len=seq_len
    )
    peaks_info['DNA sequence'] = seqs_dna
    return peaks_info


def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k) if 'N' not in seq[x:x + k]]
    kmers = " ".join(kmer)
    return kmers


def split_train_test_val(ids, seed=10, train_ratio=0.8):
    np.random.seed(seed)
    test_val_ids = np.random.choice(
        ids,
        int(len(ids) * (1 - train_ratio)),
        replace=False,
    )
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(
        test_val_ids,
        int(len(test_val_ids) / 2),
        replace=False,
    )
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    return train_ids, test_ids, val_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ad", default="./data_ft/ad.h5ad", type=str, help="the input h5ad file")
    parser.add_argument("--save_dir", default="./data_ft", type=str, help="the direction data save")
    parser.add_argument("--input_fa", default="./data_ft/hg19.fa",
                        type=str, help="the inference fasta file")
    parser.add_argument("--k_mer", default=3, type=int, help="")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    save_cwd = os.path.join(args.save_dir, str(args.k_mer) + "_mer")
    if not os.path.exists(save_cwd):
        os.makedirs(save_cwd, exist_ok=True)

    adata = anndata.read_h5ad(args.input_ad)
    seqs = DNA_extract(adata, fa_path=args.input_fa, seq_len=1344)

    train_ids, test_ids, val_ids = split_train_test_val(range(len(seqs)))
    train_seqs = seqs.iloc[train_ids, :]['DNA sequence'].tolist()
    test_seqs = seqs.iloc[test_ids, :]["DNA sequence"].tolist()
    val_seqs = seqs.iloc[val_ids, :]["DNA sequence"].tolist()

    logging.info("data processing, about one min for 31783 peaks and 2034 cells ...")
    with open(os.path.join(save_cwd, "learn_seqs.txt"), 'w') as f:
        for seq in train_seqs:
            seq = seq2kmer(seq, args.k_mer)
            f.write(seq + "\n")
    with open(os.path.join(save_cwd, "infer_seqs.txt"), 'w') as f:
        for seq in test_seqs:
            seq = seq2kmer(seq, args.k_mer)
            f.write(seq + "\n")
    with open(os.path.join(save_cwd, "val_seqs.txt"), 'w') as f:
        for seq in val_seqs:
            seq = seq2kmer(seq, args.k_mer)
            f.write(seq + "\n")
    seqs = seqs['DNA sequence']
    with open(os.path.join(save_cwd, "all_seqs.txt"), 'w') as f:
        for seq in seqs:
            seq = seq2kmer(seq, args.k_mer)
            f.write(seq + "\n")

    joblib.dump(train_ids, os.path.join(save_cwd, 'learn_ids.ids'))
    joblib.dump(val_ids, os.path.join(save_cwd, 'val_ids.ids'))
    joblib.dump(test_ids, os.path.join(save_cwd, 'infer_ids.ids'))


if __name__ == "__main__":
    main()

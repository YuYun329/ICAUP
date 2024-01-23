#!scAuto pretrain- data processing
import argparse
import math
import os.path

import numpy as np
from Bio import SeqIO
import random

from tqdm import tqdm

random.seed(42)


def sliding_window(sequence):
    windows = []
    length = len(sequence)
    i = 0

    while i < length:
        window_size = np.random.randint(6, 513)
        step_size = window_size // 2
        end_index = min(i + window_size, length)

        window = sequence[i:end_index]

        if "N" not in window:
            windows.append(window)

        i += step_size

    return windows


def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers.upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fa", default="D:/yuyun/Python/ICAUP/pre_train_dir/hg19.fa",
                        type=str, help="the inference fa file")
    parser.add_argument("--k_mer", default=3, type=int, help="")
    parser.add_argument("--file_count", default=8, type=int,
                        help="To save the memory consumption, we split the pretrain data into a few files")
    parser.add_argument("--save_path", default="./data_save", type=str, help="")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    records = list(SeqIO.parse(args.input_fa, 'fasta'))
    random.shuffle(records)
    total_records = sum(1 for _ in records)
    records_per_file = math.ceil(total_records / args.file_count)
    file_counter = 1
    record_counter = 0

    with open(os.path.join(args.save_path, f"seq_{args.k_mer}mer_{file_counter}.txt"), 'w') as f:
        for record in tqdm(records):
            seq = str(record.seq)
            kmer_seq = seq2kmer(seq, args.k_mer)
            f.write(kmer_seq + "\n")
            record_counter += 1

            if record_counter == records_per_file:
                f.close()
                file_counter += 1
                record_counter = 0
                if file_counter <= args.file_count:
                    f = open(os.path.join(args.save_path, f"seq_{args.k_mer}mer_{file_counter}.txt"), 'w')


if __name__ == "__main__":
    main()

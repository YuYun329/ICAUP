import logging
import os
import pickle
from multiprocessing import Pool

import torch
# Setup logging
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=True):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, max_length=max_length)[
        "input_ids"]
    return examples


def process_line(line, tokenizer, block_size):
    line = line.strip()
    if len(line) > 0 and not line.isspace():
        example = tokenizer.encode(line, add_special_tokens=True, max_length=block_size)["input_ids"]
        return example
    return None


def line_generator(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip()


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )
        print(cached_features_file)
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logging.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in tqdm(f.read().splitlines()) if (len(line) > 0 and not line.isspace())]

            if args.n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)[
                    "input_ids"]

            else:
                n_proc = args.n_process
                p = Pool(n_proc)
                indexes = [0]
                len_slice = int(len(lines) / n_proc)
                for i in range(1, n_proc + 1):
                    if i != n_proc:
                        indexes.append(len_slice * (i))
                    else:
                        indexes.append(len(lines))
                results = []
                for i in range(n_proc):
                    results.append(p.apply_async(convert_line_to_example,
                                                 [tokenizer, lines[indexes[i]:indexes[i + 1]], block_size, ]))
                    print(str(i) + " start")
                p.close()
                p.join()

                self.examples = []
                for result in results:
                    ids = result.get()
                    self.examples.extend(ids)

            logging.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

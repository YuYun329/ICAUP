#!scAuto_main
import argparse
import gc
import random
from sklearn.metrics import roc_auc_score
from torch import optim, nn
from tqdm import tqdm
from pre_train_dir.auto_dir.tokenization_dna import DNATokenizer
from scAuto.scAuto_model import AutoFormerFT
import os.path
import pickle
from multiprocessing import Pool
import anndata
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
import joblib
import lda
import os
from utils.earlyStopping import EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def fine_tune_func(net, pretrain_path, device):
    data_pretrain = torch.load(pretrain_path, map_location=device)
    data_auto = net.state_dict()
    for param in data_pretrain:
        if param.startswith("auto.encoder"):
            data_auto[param] = data_pretrain[param]
    return data_auto


def convert_line_to_example(tokenizer, lines, max_length, add_special_tokens=False):
    examples = tokenizer.batch_encode_plus(lines, add_special_tokens=add_special_tokens, padding="max_length",
                                           max_length=max_length, return_attention_mask=False)["input_ids"]
    return examples


class MyDataset(Dataset):
    def __init__(self, config, mode, n_process=12, block_size=512):
        super(MyDataset, self).__init__()
        source_root = os.path.dirname(config.input_ad)
        cwd = os.path.join(source_root, '{}_mer'.format(str(config.k_mer)))
        config_cwd = "./config"
        ids = joblib.load(os.path.join(cwd, "{}_ids.ids".format(mode)))
        file_name = "{}_seqs.txt".format(mode)
        tokenizer = DNATokenizer.from_pretrained(
            os.path.join(config_cwd, "bert-config-{}/vocab.txt".format(str(config.k_mer))),
            cache_dir=None)
        file_path = os.path.join(cwd, file_name)
        directory, filename = os.path.split(file_path)

        # define the labels
        self.labels = anndata.read_h5ad(os.path.join(source_root, "ad.h5ad")).X.toarray().T[ids, :]
        self.labels[self.labels > 1] = 1
        self.labels = torch.tensor(self.labels, dtype=torch.float16)
        # define the input_ids
        cached_features_file = os.path.join(
            directory, mode + "_cached_lm_" + str(block_size) + "_" + filename
        )
        if os.path.exists(cached_features_file):
            logging.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info("Creating features from dataset file at %s", file_path)
            lines = np.genfromtxt(file_path, dtype=str, delimiter='\n')
            # lines = lines[ids]
            if n_process == 1:
                self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=False, padding="max_length",
                                                            max_length=block_size)["input_ids"]
            else:
                n_proc = n_process
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
            self.examples = torch.tensor(self.examples, requires_grad=True, dtype=torch.float16)
            logging.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, index):
        cur_sequence = self.examples[index]
        cur_label = self.labels[index]
        return cur_sequence, cur_label

    def __len__(self):
        return len(self.examples)


class MyDataLoader:
    def __init__(self, config, mode):

        super(MyDataLoader, self).__init__()

        shuffle = True if mode == "learn" else False
        drop_last = shuffle

        self.loader = DataLoader(dataset=MyDataset(config=config, mode=mode),
                                 batch_size=config.batch_size,
                                 shuffle=shuffle,
                                 num_workers=0,
                                 drop_last=drop_last)

    def __call__(self, *args, **kwargs):
        return self.loader


def main(model_train=True, infer=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default=1, type=float, help="")
    parser.add_argument("--dropout", default=0.05, type=float, help="")
    parser.add_argument("--output_attention", default=True, type=bool, help="")
    parser.add_argument("--d_ff", default=2048, type=int, help="")
    parser.add_argument("--activation", default="gelu", type=str, help="")
    parser.add_argument("--num_attention_heads", default=12, type=int, help="")
    parser.add_argument("--pre_seq_len", default=336, type=int, help="")
    parser.add_argument("--num_hidden_layers", default=6, type=int, help="")
    parser.add_argument("--hidden_size", default=768, type=int, help="")
    parser.add_argument("--latent_dim", default=32, type=int, help="")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="")
    parser.add_argument("--prefix_hidden_size", default=336, type=int, help="")
    parser.add_argument("--vocab_size", default=69, type=int, help="")
    parser.add_argument("--max_position_embeddings", default=512, type=int, help="")
    parser.add_argument("--type_vocab_size", default=2, type=int, help="")
    parser.add_argument("--layer_norm_eps", default=1e-12, type=float, help="")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="")
    parser.add_argument("--batch_size", default=8, type=int, help="")
    parser.add_argument("--scaling", default=1.0, type=float, help="")
    parser.add_argument("--lr", default=6e-6, type=float, help="")
    parser.add_argument("--device", default="cuda:0", type=str, help="")
    parser.add_argument("--k_mer", default=3, type=int, help="")
    parser.add_argument("--save_path", default="./output/scAuto.bin", type=str, help="")
    parser.add_argument("--epochs", default=20, type=int, help="")
    parser.add_argument("--pretrain_dir", default="./pretrain_outdir", type=str, help="")
    parser.add_argument("--input_ad", default="./data_ft/ad.h5ad", type=str, help="the input h5ad file")
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model = AutoFormerFT(config=args, n_cells=2034).to(device=args.device)

    # lda to init
    adata = anndata.read_h5ad(args.input_ad)
    X = adata.X.toarray().astype(dtype="int64")
    L = lda.LDA(n_topics=32, n_iter=80, random_state=1)
    L.fit(X)
    doc_topic_matrix = L.doc_topic_
    model.init_with_lda(doc_topic_matrix)
    del X
    del doc_topic_matrix
    gc.collect()

    if model_train:
        train(config=args, model=model)
    if infer:
        inference(config=args, model=model)


def train(config, model):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config.lr, betas=(0.95, 0.9995))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     patience=3,
                                                     verbose=True)
    learn_loader = MyDataLoader(config=config, mode="learn").__call__()
    val_loader = MyDataLoader(config=config, mode="val").__call__()
    loss_function = nn.BCELoss()
    early_stopping = EarlyStopping(patience=10,
                                   verbose=True)
    model.load_state_dict(
        fine_tune_func(
            net=model,
            pretrain_path=os.path.join(config.pretrain_dir, "pre_train_auto_{}mer.bin".format(str(config.k_mer))),
            device=config.device,
        ))
    for name, param in model.named_parameters():
        if name.endswith("query_projection.weight") or name.endswith("key_projection.weight"):
            param.requires_grad = False
    for epoch in range(config.epochs):
        model.train()
        ProgressBar = tqdm(learn_loader)
        for sample in ProgressBar:
            optimizer.zero_grad()

            ProgressBar.set_description("Learning_Epoch %d" % epoch)
            sequence, label = sample
            output = model.forward(input_ids=sequence.float().to(config.device))
            _, y = output
            loss = loss_function(y, label.float().to(config.device))
            roc_auc = roc_auc_score(y_score=y.flatten().detach().cpu().numpy(),
                                    y_true=label.flatten().detach().cpu().numpy())
            ProgressBar.set_postfix(roc_auc=roc_auc, loss=loss.item())

            loss.backward()
            optimizer.step()
        val_loss = []
        with torch.no_grad():
            model.eval()
            for val_sequence, val_label in val_loader:
                _, val_y = model(input_ids=val_sequence.float().to(config.device))
                val_label = val_label.float().to(config.device)

                val_loss.append(loss_function(val_y, val_label).item())

            val_loss_avg = torch.mean(torch.Tensor(val_loss))
        logging.info("val loss : {}".format(val_loss_avg.item()))
        scheduler.step(val_loss_avg)
        early_stopping.__call__(val_loss=val_loss_avg, model=model, path=config.save_path)


def inference(config, model):
    if not os.path.exists(os.path.dirname(config.save_path)):
        raise FileExistsError("your model has not been trained")

    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    infer_loader = MyDataLoader(mode="infer", config=config).__call__()
    infer_loss = []
    auc_list = []
    loss_function = nn.BCELoss()

    ProgressBar = tqdm(infer_loader)
    with torch.no_grad():
        model.eval()
        for infer_sequence, infer_label in ProgressBar:
            _, infer_y = model(input_ids=infer_sequence.float().to(config.device))
            infer_label = infer_label.float().to(config.device)
            loss_item = loss_function(infer_y, infer_label).item()
            roc_auc = roc_auc_score(y_score=infer_y.flatten().detach().cpu().numpy(),
                                    y_true=infer_label.flatten().detach().cpu().numpy())
            ProgressBar.set_postfix(roc_auc=roc_auc, loss=loss_item)
            infer_loss.append(loss_item)
            auc_list.append(roc_auc)
        infer_loss_avg = torch.mean(torch.Tensor(infer_loss))
    auc_mean = np.mean(auc_list)
    logging.info("inference loss = {}, auc = {}".format(infer_loss_avg, auc_mean))


if __name__ == "__main__":
    main(model_train=True, infer=True)

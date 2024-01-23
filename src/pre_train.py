#!pretrain
import argparse
import glob
import logging
import os
import random
import re
import shutil
from copy import deepcopy
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, PreTrainedModel

from pre_train_dir.auto_dir.data_util import LineByLineTextDataset
from pre_train_dir.auto_dir.scAuto_pretrain import ScAutoPretrain
from pre_train_dir.auto_dir.tokenization_dna import DNATokenizer
from pre_train_dir.auto_dir.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    mask_list = MASK_LIST[tokenizer.kmer]

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the "
            "--mlm flag if you want to use this tokenizer. "
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i] != 0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index == 1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write(str(float(perplexity)) + "\n")
            # writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for _ in range(args.num_train_epochs):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(input_ids=inputs, attention_mask=None, token_type_ids=None, position_ids=None,
                            head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                            encoder_attention_mask=None, lm_labels=None,
                            masked_lm_labels=labels)

            loss = outputs[0]
            loss = loss / args.gradient_accumulation_steps if args.gradient_accumulation_steps > 1 else loss
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        checkpoint_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_path = os.path.join(checkpoint_dir, "{}-{}.bin".format("pre_train_auto", str(global_step)))
                        torch.save(model.state_dict(), save_path)
                        tokenizer.save_pretrained(args.output_dir)
                        logger.info("Saving model checkpoint to %s", args.output_dir)

                if 0 < args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < args.max_steps < global_step:
                # train_iterator.close()
                break
    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="./pre_train_dir/data_save", type=str, help="")
    parser.add_argument("--factor", default=1, type=float, help="")
    parser.add_argument("--dropout", default=0.05, type=float, help="")
    parser.add_argument("--output_attention", default=True, type=bool, help="")
    parser.add_argument("--d_ff", default=2048, type=int, help="")
    parser.add_argument("--activation", default="gelu", type=str, help="")
    parser.add_argument("--num_attention_heads", default=6, type=int, help="")
    parser.add_argument("--pre_seq_len", default=336, type=int, help="")
    parser.add_argument("--num_hidden_layers", default=6, type=int, help="")
    parser.add_argument("--hidden_size", default=768, type=int, help="")
    parser.add_argument("--latent_dim", default=32, type=int, help="")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="")
    parser.add_argument("--prefix_hidden_size", default=336, type=int, help="")
    parser.add_argument("--n_batch", default=6, type=int, help="")
    parser.add_argument("--type_vocab_size", default=2, type=int, help="")
    parser.add_argument("--max_position_embeddings", default=512, type=int, help="")
    parser.add_argument("--layer_norm_eps", default=1e-12, type=float, help="")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="")
    parser.add_argument("--output_dir", default="./pretrain_outdir", type=str, help="")
    parser.add_argument("--k_mer", default=3, type=int, help="")
    parser.add_argument("--do_train", default=True, type=bool, help="")
    parser.add_argument("--do_eval", default=False, type=bool, help="")
    parser.add_argument("--mlm", default=True, type=bool, help="")
    parser.add_argument("--gradient_accumulation_steps", default=25, type=int, help="")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="")
    parser.add_argument("--save_steps", default=500, type=int, help="")
    parser.add_argument("--save_total_limit", default=20, type=int, help="")
    parser.add_argument("--max_steps", default=200000, type=int, help="")
    parser.add_argument("--evaluate_during_training", default=True, type=bool, help="")
    parser.add_argument("--logging_steps", default=500, type=int, help="")
    parser.add_argument("--line_by_line", default=True, type=bool, help="")
    parser.add_argument("--learning_rate", default=4e-4, type=float, help="")
    parser.add_argument("--block_size", default=512, type=int, help="")
    parser.add_argument("--overwrite_cache", default=False, type=bool, help="")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="")
    parser.add_argument("--beta1", default=0.9, type=float, help="")
    parser.add_argument("--beta2", default=0.98, type=float, help="")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="")
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="")
    parser.add_argument("--n_process", default=12, type=int, help="")
    parser.add_argument("--mlm_probability", default=0.15, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--n_gpu", default=1, type=int, help="")
    parser.add_argument("--device", default="cuda:0", type=str, help="")
    parser.add_argument("--config_dir", default="./config", type=str, help="")
    parser.add_argument("--model_type", default="dna", type=str, help="")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    args.vocab_size = 4 ** args.k_mer + 5
    args.model_name = "pre_train_auto_{}mer.bin".format(str(args.k_mer))
    file_list = []
    for file in os.listdir(args.input_folder):
        if file.endswith("txt"):
            file_list.append(os.path.join(args.input_folder, file))

    tokenizer_class = DNATokenizer
    tokenizer = tokenizer_class.from_pretrained(
        os.path.join(args.config_dir, "bert-config-{}/vocab.txt".format(str(args.k_mer))),
        cache_dir=None)

    args.block_size = min(args.block_size, tokenizer.max_len)
    model = ScAutoPretrain(config=args)
    model.to(args.device)

    for data_file in file_list:
        args.train_data_file = data_file
        args.eval_data_file = data_file
        logger.info("processed data file:{}".format(args.train_data_file))
        logger.info("Training new model from scratch")

        # Training
        if args.do_train:
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
            # Load a trained model and vocabulary that you have fine-tuned
            if os.path.exists(os.path.join(args.output_dir, args.model_name)):
                model.load_state_dict(
                    torch.load(os.path.join(args.output_dir, args.model_name), map_location=args.device))
                logger.info("the pre-trained model is loaded")
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)
            logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
            logger.info("Saving model checkpoint to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.model_name))
            tokenizer.save_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

        # Evaluation
        results = {}
        if args.do_eval:
            logger.info("Training/evaluation parameters %s", args)
            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
                model.load_state_dict(
                    torch.load(os.path.join(args.output_dir, args.model_name), map_location=args.device))
                model.to(args.device)
                result = evaluate(args, tokenizer, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)

        return results


if __name__ == "__main__":
    main()

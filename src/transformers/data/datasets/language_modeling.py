import logging
import os
import pickle
import time

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Optional
from ...tokenization_utils import PreTrainedTokenizer
from ...modeling_utils import PreTrainedModel
from ...training_args import TrainingArguments
from .glue import glue_convert_examples_to_features
from ..processors.utils import InputExample,InputFeatures
from ...optimization import AdamW, get_linear_schedule_with_warmup
import random
import json
import copy

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger = logging.getLogger(__name__)


class MaskSelector:
    model: PreTrainedModel       

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments):

        self.model = model.to(args.device)
        self.args = args

        set_seed(self.args.seed)
    
    def predict(
        self, 
        mask_batch):
        #mask_batch size  [batch_size, seq_len]
        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        model.eval()
        for k, v in mask_batch.items():
            mask_batch[k] = v.to(self.args.device)
        with torch.no_grad():
            outputs = model(**mask_batch)
            logits = outputs[0]
        
        # src-0,tgt-1
        preds = logits.detach().cpu().numpy()
        preds = [x[1] for x in preds]
        # print("max_logits:{} min_logits:{}".format(max(preds), min(preds)))
        return np.argmax(preds)


class MaskGenerator:
    model: PreTrainedModel

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments):

        self.model = model.to(args.device)
        self.args = args
    
    def predict(self,
        batch, sample_rate=0.15):
        model = self.model
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)



        else:
            model = self.model
        model.eval()
        for k,v in batch.items():
            batch[k] = v.to(self.args.device)

        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0] #[batch_size, seq_len, hidden_size]
        
        sequence_size = batch["input_ids"].size(1)
        sample_size = int(sample_rate*sequence_size)
        all_preds = []
        for instance_logits in logits:
            logits_1 = [ (i,x[1]) for i, x in enumerate(instance_logits)]
            sorted_logits_1 = sorted(logits_1, key = lambda x:x[1])
            mask_index = sorted_logits_1[:sample_size]

            instance_mask = [0]*sequence_size

            for i in mask_index:
                instance_mask[i[0]] = 1
            all_preds.append(instance_mask)
            
        # 0 for unchanged token, 1 for masking token
        # preds = torch.argmax(logits, dim=-1)
        # print(torch.sum(preds))

        return all_preds

        
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class SelectCacheDataset(Dataset):
    def __init__(self, file_path:str):
        assert os.path.isfile(file_path)

        all_data = torch.load(file_path)
        self.examples = all_data
    
    def __len__(self):
        return len(self.examples["input_ids"])
    
    def __getitem__(self, i):
        return {
            "input_ids":self.examples["input_ids"][i],
            "labels":self.examples["labels"][i]
        }


class FullyLineByLineTextDataset(Dataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        file_path: str, 
        block_size: int,
        cache_dir: Optional[str] = None):

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else None,
            "cached_{}_{}_{}_{}".format(
                "select_lm", tokenizer.__class__.__name__, "200", "select_lm",
            ),
        )
        lock_path = cached_features_file + ".lock"
    
        if os.path.exists(cached_features_file) :
            start = time.time()
            self.features = torch.load(cached_features_file)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
        else:
            assert os.path.isfile(file_path)
            # Here, we do not cache the features, operating under the assumption
            # that we will soon use fast multithreaded tokenizers from the
            # `tokenizers` repo everywhere =)
            logger.info("Creating features from dataset file at %s", file_path)

            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

            examples = []
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % ("train", i)
                text_a = line
                label = "0"
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            # batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
            self.features = glue_convert_examples_to_features(
                        examples,
                        tokenizer,
                        max_length=512,
                        label_list=["0", "1"],
                        output_mode="classification",
                        task="cola"
                    )
            start = time.time()
            torch.save(self.features, cached_features_file)
            # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


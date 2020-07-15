from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from ..tokenization_utils import PreTrainedTokenizer
from ..modeling_utils import PreTrainedModel
from ..data.datasets.language_modeling import MaskSelector, MaskGenerator
from tqdm import tqdm


class DataCollator(ABC):
    """
    A `DataCollator` is responsible for batching
    and pre-processing samples of data as requested by the training loop.
    """

    @abstractmethod
    def collate_batch(self) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.

        Returns:
            A dictionary of tensors
        """
        pass


InputDataClass = NewType("InputDataClass", Any)



@dataclass
class DefaultDataCollator(DataCollator):
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        # In this method we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if hasattr(first, "label") and first.label is not None:
            if type(first.label) is int:
                labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, "label_ids") and first.label_ids is not None:
            if type(first.label_ids[0]) is int:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
            batch = {"labels": labels}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in vars(first).items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch


@dataclass
class DataCollatorForLanguageModeling(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForWeightedLanguageModeling(DataCollator):

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    weighted_vocab: list = None

    def collate_batch(self, examples:List[torch.Tensor]) -> torch.Tensor :
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels":labels}
        else:
            return {"input_ids": batch, "labels":labels}
    
    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor :
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)

        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    
    def mask_tokens(self, inputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone() #[batch, seq_len]
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Inevitable O(n^2) index
        for i in range(len(inputs)):
            probs = []
            for j in range(len(inputs[i])):
                input_id = int(inputs[i][j])
                probs.append( 1.0000 + float(self.weighted_vocab[input_id])) # not negative
            sum_prob = sum(probs)
            
            weighted_probs = []
            for j in range(len(inputs[i])):
                weighted_prob= (probs[j]*len(inputs[i])*1.000/sum_prob) * self.mlm_probability
                probability_matrix[i][j] = weighted_prob
                weighted_probs.append(weighted_prob)
            # print("max_prob:{} min_prob:{}".format(max(weighted_probs), min(weighted_probs)))
                
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]

        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForMaskGen(DataCollator):
    tokenizer: PreTrainedTokenizer
    generator: MaskGenerator
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples:List) -> Dict[str, torch.Tensor]:

        all_input_ids = torch.tensor([instance.input_ids for instance in examples], dtype=torch.long)
        all_attention_mask = torch.tensor([instance.attention_mask for instance in examples], dtype=torch.long)
        all_token_type_ids = torch.tensor([instance.token_type_ids for instance in examples], dtype=torch.long)

        generator_input = {
            "input_ids":all_input_ids,
            "attention_mask":all_attention_mask,
            "token_type_ids": all_token_type_ids
        }

        out = self.generator.predict(generator_input).float() # out shape same as input_ids
        print(out)
        all_labels = all_input_ids.clone()
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in all_labels.tolist()
        ]
        out.masked_fill(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = all_labels.eq(self.tokenizer.pad_token_id)
            out.masked_fill(padding_mask, value=0.0)
        
        masked_indices = torch.bernoulli(out).bool()
        all_labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(out, 0.8)).bool() & masked_indices

        all_input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernouli(torch.full(all_labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced

        random_words = torch.randint(len(self.tokenizer), all_labels.shape, dtype=torch.long)

        all_input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids":all_input_ids,
            "labels":all_labels
        }



@dataclass
class DataCollatorForDistillLM(DataCollator):
    tokenizer: PreTrainedTokenizer
    selector : MaskSelector
    mlm_sample_times: int = 16
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List) -> Dict[str, torch.Tensor]:
        all_inputs = []
        all_attention_mask = []
        all_token_type_ids = []
        all_labels = []
        for instance in examples:
            batch = {}
            for k,v in vars(instance).items():
                 batch[k] = torch.tensor([getattr(instance, k) for _ in range(self.mlm_sample_times)], dtype=torch.long)

            inputs, labels = self.mask_tokens(batch["input_ids"])
            selector_input = {   
                "input_ids":inputs,
                "attention_mask":batch["attention_mask"],
                "token_type_ids":batch["token_type_ids"],
            }

            out = self.selector.predict(selector_input)
            selected_instance = inputs[out] #[seq_len]

            # show examples
            selected_inputs = selected_instance.detach().cpu().numpy()
            selected_labels = labels[out].detach().cpu().numpy()

            # convert to sequence labelling 
            # print(selected_instance.shape)
            sl_labels = []
            for i in selected_labels:
                if i == -100:
                    sl_labels.append(0)
                else:
                    sl_labels.append(1)
            # print(sl_labels)
            all_inputs.append(instance.input_ids)
            all_attention_mask.append(instance.attention_mask)
            all_token_type_ids.append(instance.token_type_ids)
            all_labels.append(sl_labels)
            

        return {
            "input_ids":torch.tensor(all_inputs, dtype=torch.long),
            "attention_mask":torch.tensor(all_attention_mask, dtype=torch.long),
            "token_type_ids":torch.tensor(all_token_type_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long)
        }

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)



@dataclass
class DataCollatorForSelectLM(DataCollator):
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    selector: MaskSelector
    mlm_sample_times: int = 1
    mlm: bool = True
    mlm_probability: float = 0.15

    def collate_batch(self, examples: List) -> Dict[str, torch.Tensor]:
        # batch = self._tensorize_batch(examples)
        all_inputs = []
        all_labels = []
        for instance in examples:  
            batch = {}
            for k, v in vars(instance).items():
                batch[k] = torch.tensor([getattr(instance, k) for _ in range(self.mlm_sample_times)], dtype=torch.long)

            
            inputs, labels = self.mask_tokens(batch["input_ids"])
            selector_input = {
                "input_ids":inputs,
                "attention_mask":batch["attention_mask"],
                "token_type_ids":batch["token_type_ids"],
            }

            out = self.selector.predict(selector_input)
            selected_instance = batch["input_ids"][out]
            # show examples
            selected_inputs = selected_instance.detach().cpu().numpy()
            selected_labels = labels[0].detach().cpu().numpy()

            # print("mask inputs : {}".format(" ".join([self.tokenizer._convert_id_to_token(x) for x in selected_inputs])))

            # print("origin inputs : {}".format(" ".join([self.tokenizer._convert_id_to_token(x) for x in selected_labels])))

            all_inputs.append(selected_instance)
            all_labels.append(labels[0])
        # print(len(all_inputs), all_inputs[0].shape)
        # print(labels.shape)
        return {
            "input_ids":torch.stack(all_inputs, dim=0),
            "labels": torch.stack(all_labels, dim=0)
        }

        # if self.mlm:
        #     inputs, labels = self.mask_tokens(batch)
        #     return {"input_ids": inputs, "labels": labels}
        # else:
        #     return {"input_ids": batch, "labels": batch}

    def _tensorize_batch(self, examples: List[torch.Tensor]) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

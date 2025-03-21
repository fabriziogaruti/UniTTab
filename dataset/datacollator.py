from typing import Dict, List, Union
import torch
from transformers import DefaultDataCollator
import numpy as np


def _torch_collate_batch(examples, tokenizer):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)
    have_third_dim = examples[0].dim()==2
    if have_third_dim:
        third_dim = examples[0].size(1)
    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    # logger.debug(f'lengths in batch are {[x.size(0) for x in examples]}, padded to {max_length}')
    if have_third_dim:
        result = examples[0].new_full([len(examples), max_length, third_dim], tokenizer.pad_token_id)
    else:
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        result[i, : example.shape[0]] = example
    return result


class TransDataCollatorFinetuning(DefaultDataCollator):

    def __init__(self, tokenizer, seq_len):

        super().__init__()
        self.tokenizer = tokenizer

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_ids = [inp['input_ids'] for inp in examples]
        labels = [lbl['labels'] for lbl in examples]
        batch = _torch_collate_batch(input_ids, self.tokenizer)
        batch_lbl = torch.stack(labels, dim=0)

        return {"input_ids": batch, "labels": batch_lbl}


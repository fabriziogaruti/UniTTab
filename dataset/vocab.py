from collections import OrderedDict
import numpy as np
import torch
from functools import partial
from loguru import logger


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class EncoderVocab:
    def __init__(self, input_vocab, output_vocab, encoder_fit):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.encoder_fit = encoder_fit


class Vocabulary:
    def __init__(self, target_column_name="Is Fraud?"):
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.special_field_tag = "SPECIAL"
        self.target_column_name = target_column_name

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = []
        self.token2id[self.special_field_tag] = OrderedDict()
        self.fields_steps = {}
        self.lid2gid = None
        self.tok2gid = None

        self.filename = ''  # this field is set in the `save_vocab` method

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False):
        global_id, local_id = None, None

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id
        return global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        global_id, local_id = None, None
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]

        else:
            raise Exception(f"token {token} not found in field: {field_name}")

        if return_local:
            return local_id

        return global_id

    def set_field_steps(self, field, steps):
        logger.info(f'setting steps for field {field}')
        logger.info(f'num steps is {len(steps)}')
        self.fields_steps[field] = steps

    def set_field_keys(self, keys):

        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys.append(key)

        self.field_keys.append(self.special_field_tag)

    def get_field_ids(self, field_name, return_local=False):
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        selected_idx = 0
        if return_local:
            selected_idx = 1
        return [ids[idx][selected_idx] for idx in ids]


    def get_field_tokens(self, field_name, return_local=False):
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        return list(ids.keys())


    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        if not torch.is_tensor(global_ids):
            global_ids_torch = torch.from_numpy(global_ids)
        else:
            global_ids_torch = global_ids

        device = global_ids_torch.device

        def map_global_ids_to_local_ids(gid):
            lid = self.id2token[gid][2] if gid != -100 else -100
            # logger.debug(f'gid {gid}, lid {lid}')
            return lid

        def map_global_ids_to_tokens(gid):
            return int(self.id2token[gid][0]) if gid != -100 else -100

        if what_to_get == 'local_ids':
            local_ids = global_ids_torch.cpu().apply_(map_global_ids_to_local_ids).to(device)
            if not torch.is_tensor(global_ids):
                return local_ids.numpy()
            return local_ids
        elif what_to_get == 'tokens':
            tokens = global_ids_torch.cpu().apply_(map_global_ids_to_tokens).to(device)
            if not torch.is_tensor(global_ids):
                return tokens.numpy()
            return tokens
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def get_local_to_global_mapping(self):
        self.lid2gid = OrderedDict()
        self.tok2gid = OrderedDict()
        for field_name in self.token2id.keys():
            self.lid2gid[field_name] = OrderedDict()
            self.tok2gid[field_name] = OrderedDict()
            for token, (gid, lid) in self.token2id[field_name].items():
                self.lid2gid[field_name][lid] = (gid, token)
                self.tok2gid[field_name][token] = (gid, lid)

    def get_from_local_ids(self, local_ids, field_name, what_to_get='global_ids'):
        if not torch.is_tensor(local_ids):
            local_ids_torch = torch.from_numpy(local_ids)
        else:
            local_ids_torch = local_ids

        device = local_ids_torch.device
        if not self.lid2gid:
            self.get_local_to_global_mapping()

        def map_local_ids_to_global_ids(lid, field_name):
            return self.lid2gid[field_name][lid][0] if lid != -100 else -100

        def map_local_ids_to_tokens(lid, field_name):
            return int(self.lid2gid[field_name][lid][1]) if lid != -100 else -100

        if what_to_get == 'global_ids':
            mapping = partial(map_local_ids_to_global_ids, field_name=field_name)
            global_ids = local_ids_torch.cpu().apply_(mapping).to(device)
            if not torch.is_tensor(local_ids):
                return global_ids.numpy()
            return global_ids
        elif what_to_get == 'tokens':
            mapping = partial(map_local_ids_to_tokens, field_name=field_name)
            tokens = local_ids_torch.cpu().apply_(mapping).to(device)
            if not torch.is_tensor(local_ids):
                return tokens.numpy()
            return tokens
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def get_from_tokens(self, tokens, field_name, what_to_get='global_ids'):
        if not torch.is_tensor(tokens):
            local_ids_torch = torch.from_numpy(tokens)
        else:
            local_ids_torch = tokens

        device = local_ids_torch.device
        if not self.tok2gid:
            self.get_local_to_global_mapping()

        def map_tokens_to_global_ids(tok, field_name):
            return self.tok2gid[field_name][tok][0] if tok != -100 else -100

        def map_tokens_to_local_ids(tok, field_name):
            return self.tok2gid[field_name][tok][1] if tok != -100 else -100

        if what_to_get == 'global_ids':
            mapping = partial(map_tokens_to_global_ids, field_name=field_name)
            global_ids = local_ids_torch.cpu().apply_(mapping).to(device)
            if not torch.is_tensor(tokens):
                return global_ids.numpy()
            return global_ids
        elif what_to_get == 'local_ids':
            mapping = partial(map_tokens_to_local_ids, field_name=field_name)
            tokens = local_ids_torch.cpu().apply_(mapping).to(device)
            if not torch.is_tensor(tokens):
                return tokens.numpy()
            return tokens
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False):
        keys = self.field_keys.copy()

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        # TODO : remove the dependency of re-initializing here. retrieve from field_key = SPECIAL
        keys = ["unk_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_

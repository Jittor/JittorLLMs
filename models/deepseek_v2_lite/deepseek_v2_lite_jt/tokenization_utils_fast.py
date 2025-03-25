# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
see tokenization_utils.py
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from tokenizers import AddedToken


# Fast tokenizers (provided by HuggingFace tokenizer's library) can be saved in a single file
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# Slow tokenizers have an additional added tokens files
ADDED_TOKENS_FILE = "added_tokens.json"


MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}


class PreTrainedTokenizerFast():
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, *args, **kwargs):
        fast_tokenizer_file = kwargs.pop("tokenizer_file", None)

        self._tokenizer = TokenizerFast.from_file(fast_tokenizer_file)

        self._decode_use_source_tokenizer = False

        _truncation = self._tokenizer.truncation

        self._tokenizer.no_truncation()

        # Set the splitting mode for special tokens for the tokenizer to be used throughout the class.
        self._tokenizer.encode_special_tokens = False

        tokens_to_add = []
        encoder =  ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>']
        # if some of the special tokens are strings, we check if we don't already have a token
        tokens_to_add += [
            token for token in ['<s>', '</s>', '<unk>'] if token not in encoder and token not in tokens_to_add
        ]

        if len(tokens_to_add) > 0:
            # super hack: if a token.special is set, tokenizer ignores it for now so FIXME @ArthurZ
            # Accumulate added tokens into batches of special/non-special tokens, because calling add_tokens() for
            # individual tokens would repeatedly rebuild a trie, which can be slow.
            is_last_special = None
            tokens = []
            special_tokens = ['<s>', '</s>', '<unk>']
            for token in tokens_to_add:
                is_special = (
                    (token.special or str(token) in special_tokens)
                    if isinstance(token, AddedToken)
                    else str(token) in special_tokens
                )
                if is_last_special is None or is_last_special == is_special:
                    tokens.append(token)
                else:
                    self._add_tokens(tokens, special_tokens=is_last_special)
                    tokens = [token]
                is_last_special = is_special
            if tokens:
                self._add_tokens(tokens, special_tokens=is_last_special)

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        return self._tokenizer.decoder

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None:
            if self.verbose:
                logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None:
            if self.verbose:
                logger.error("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None:
            if self.verbose:
                logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None:
            if self.verbose:
                logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None:
            if self.verbose:
                logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None:
            if self.verbose:
                logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None:
            if self.verbose:
                logger.error("Using additional_special_tokens, but it is not set yet.")
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the BOS token")
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the EOS token")
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the UNK token")
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the SEP token")
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the PAD token")
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the CLS token")
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the MASK token")
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value if value is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index

    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)

        return self._tokenizer.add_tokens(new_tokens)

    def _batch_encode_plus(
        self,
        # batch_text_or_text_pairs: Union[
        #     List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        # ],
        batch_text_or_text_pairs: List[str],
        add_special_tokens: bool = True,
        # padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        split_special_tokens: bool = False,
    ) -> dict:

        encodings = self._tokenizer.encode_batch(
            batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            is_pretokenized=is_split_into_words,
        )

        tokens_and_encodings = [
            {
                "input_ids": [encoding.ids],
                "attention_mask": [encoding.attention_mask],
            }
            for encoding in encodings
        ]

        sanitized_tokens = {}
        for key in tokens_and_encodings[0].keys():
            stack = [e for item in tokens_and_encodings for e in item[key]]
            sanitized_tokens[key] = stack

        return sanitized_tokens

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:

        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text

from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

import codecs

logger = logging.getLogger(__name__)


@DatasetReader.register("tagger")
class TaggerDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    ```
    WORD POS-TAG CHUNK-TAG
    ```

    with a blank line indicating the end of each sentence
    and converts it into a `Dataset` suitable for sequence tagging.

    Each `Instance` contains the words in the `"tokens"` `TextField`.
    The values corresponding to the `tag_label`
    values will get loaded into the `"tags"` `SequenceLabelField`.
    And if you specify any `feature_labels` (you probably shouldn't),
    the corresponding values will get loaded into their own `SequenceLabelField` s.

    Registered as a `DatasetReader` with name "conll2000".

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label : `str`, optional (default=`chunk`)
        Specify `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels : `Sequence[str]`, optional (default=`()`)
        These labels will be loaded as features into the corresponding instance fields:
        `pos` -> `pos_tags` or `chunk` -> `chunk_tags`.
        Each will have its own namespace : `pos_tags` or `chunk_tags`.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme : `str`, optional (default=`BIO`)
        Specifies the coding scheme for `chunk_labels`.
        Valid options are `BIO` and `BIOUL`.  The `BIO` default maintains
        the original BIO scheme in the CoNLL 2000 chunking data.
        In the BIO scheme, B is a token starting a span, I is a token continuing a span, and
        O is a token outside of a span.
    label_namespace : `str`, optional (default=`labels`)
        Specifies the namespace for the chosen `tag_label`.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = "labels",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with codecs.open(file_path, "r", encoding="utf8") as f:
            for line in f:
                line  = line.rstrip()
                cols = line.split("\t")
                source = [Token(tok) for tok in cols[0].split(" ")]
                labels = [label for label in cols[1].split(" ")]
                yield self.text_to_instance(source, labels)


    def text_to_instance(  # type: ignore
        self, tokens: List[Token], tags: List[str] = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        # Add "tag label" to instance
        if tags != None:
            instance_fields["tags"] = SequenceLabelField(tags, sequence, self.label_namespace)

        return Instance(instance_fields)

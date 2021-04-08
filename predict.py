from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import DataLoader
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import Activation
from allennlp.training.trainer import Trainer
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from tagging.dataset_readers.tagger import TaggerDatasetReader
from argparse import ArgumentParser
import itertools 
import Tagger
import pickle
import copy
import codecs

torch.manual_seed(1)

def options():
    parser = ArgumentParser(description='This tagger predicts labels for each token in a given text on a line-by-line basis.')
    parser.add_argument("--output_file", dest="output_file", type=str, help="path to a train file")
    parser.add_argument("--input_file", dest="input_file", type=str, help="path to a train file")
    parser.add_argument("--tagger_model_file", dest="tagger_model_file", type=str, help="path to a model.")
    parser.add_argument("--tagger_param_file", dest="tagger_param_file", type=str, help="path to parameters.")
    parser.add_argument("--tagger_vocab_file", dest="tagger_vocab_file", type=str, help="path to a vocabularly file")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--gpuid", dest="gpuid", type=int, help="gpuid to run this tagger")
    opts = parser.parse_args()
    return opts

def load_model(model, params, model_file, gpuid):
    # select a bert specific indexer
    if params["with_bert"]:
        from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
        indexer=PretrainedTransformerMismatchedIndexer(model_name=params["bert_name"],
                max_length=params["bert_max_len"])
    # separate by spaces
    else:
        from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
        indexer=SingleIdTokenIndexer()
    # Select a device
    if gpuid >= 0 and torch.cuda.is_available():
        cuda_device = gpuid
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
        model = model.cpu()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model, indexer

def main():
    opts = options()
    # load zero pronoun detector
    with open(opts.tagger_param_file, mode='rb') as f:
        tagger_params = pickle.load(f)
    print(tagger_params)
    tagger_vocab = Vocabulary.from_files(opts.tagger_vocab_file)
    tagger_model = Tagger.build(tagger_params, tagger_vocab)
    tagger_model, tagger_indexer = load_model(tagger_model, tagger_params, opts.tagger_model_file, opts.gpuid)
    # prepare dataset readers
    tagger_reader = TaggerDatasetReader(token_indexers={"tokens" : tagger_indexer})
    
    with codecs.open(opts.input_file, "r", encoding="utf8") as f_in, codecs.open(opts.output_file, "w", encoding="utf8") as f_out:
        for line in f_in:
            line = line.strip()
            toks = [Token(tok) for tok in line.split(" ")]
            tagger_instance = tagger_reader.text_to_instance(toks)
            output = tagger_model.forward_on_instance(tagger_instance)
            f_out.write(" ".join(output["tags"]) + "\n")

if __name__ == "__main__":
    main()

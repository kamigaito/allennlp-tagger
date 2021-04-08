from typing import Iterator, List, Dict
import torch
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import Activation
from allennlp.data import PyTorchDataLoader
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from tagging.dataset_readers.tagger import TaggerDatasetReader
from argparse import ArgumentParser
import itertools 
import Tagger
import pickle

torch.manual_seed(1)

def options():
    parser = ArgumentParser(description='This tagger uses tsv formatted files for training.')
    parser.add_argument("--train_file", dest="train_file", type=str, default="./dataset/train.csv" , help="Path to a train file")
    parser.add_argument("--valid_file", dest="valid_file", type=str, default="./dataset/dev.csv" , help="Path to a validatioon file")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="./models" , help="Name of model directory to store parameters.")
    parser.add_argument("--dim_emb", dest="dim_emb", type=int, default=768, help="Dimension of word embeddings")
    parser.add_argument("--dim_hid", dest="dim_hid", type=int, default=256, help="Dimension of hidden layers")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.2, help="Dropout ratio")
    parser.add_argument("--epochs", dest="epochs", type=int, default=40, help="Maximum training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_amp", dest="use_amp", action='store_true', help="Use mixed precision")
    parser.add_argument("--num_gradient_accumulation_steps", dest="num_gradient_accumulation_steps", type=int, default=1, help="How many steps to accumulate gradients.")
    parser.add_argument("--num_enc_layers", dest="num_enc_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--with_bert", dest="with_bert", action='store_true', help="Use bert as an embedder")
    parser.add_argument("--bert_name", dest="bert_name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking", help="Name of the BERT you want to use")
    parser.add_argument("--bert_max_len", dest="bert_max_len", type=int, default=512, help="Maximum length to encode")
    parser.add_argument("--min_freq", dest="min_freq", type=int, default=2, help="Minimum frequency of words in the vocabulary (used in the setting without BERT)")
    parser.add_argument("--gpuid", dest="gpuid", type=int, default=0, help="gpuid")
    opts = parser.parse_args()
    return opts

def main():
    
    opts = options()

    # select a bert specific indexer
    if opts.with_bert:
        from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
        indexer=PretrainedTransformerMismatchedIndexer(model_name=opts.bert_name,
                max_length=opts.bert_max_len)
    # separate by spaces
    else:
        from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
        indexer=SingleIdTokenIndexer()

    reader = TaggerDatasetReader(token_indexers={"tokens" : indexer})
    train_dataset = reader.read(opts.train_file)
    valid_dataset = reader.read(opts.valid_file)
    params = Tagger.opts2params(opts)
    
    with open(opts.model_dir + "/params.pkl", mode='wb') as f:
        pickle.dump(params, f)
    
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset,
            min_count={'tokens': opts.min_freq})
    train_dataset.index_with(vocab)
    valid_dataset.index_with(vocab)
    train_data_loader = PyTorchDataLoader(train_dataset, 
            batch_sampler=BucketBatchSampler(
                train_dataset,
                batch_size=opts.batch_size,
                sorting_keys=["tokens"]))
    valid_data_loader = PyTorchDataLoader(valid_dataset, 
            batch_sampler=BucketBatchSampler(
                valid_dataset,
                batch_size=opts.batch_size,
                sorting_keys=["tokens"]))
    
    model = Tagger.build(params, vocab)
    if torch.cuda.is_available():
        cuda_device = opts.gpuid
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    # select an optimizer for fine-tuning
    if opts.with_bert:
        from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = HuggingfaceAdamWOptimizer(model_parameters=parameters,
                lr=0.0003,
                parameter_groups=[([".*transformer.*"],{"lr": 1e-05})])
    # optimizer for random initialization
    else:
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.001)


    trainer = GradientDescentTrainer(model=model,
                      optimizer=optimizer,
                      data_loader=train_data_loader,
                      validation_data_loader=valid_data_loader,
                      num_epochs=1,
                      use_amp=opts.use_amp,
                      num_gradient_accumulation_steps=opts.num_gradient_accumulation_steps,
                      cuda_device=cuda_device)
    
    vocab.save_to_files(opts.model_dir + "/vocab")

    best_f1 = 0.0
    for i in range(opts.epochs):
        epoch = i + 1
        print('Epoch: {}'.format(epoch))
        info = trainer.train()
        print(info)
        if info["validation_accuracy"] > best_f1:
            best_f1 = info["validation_accuracy"]
            with open(opts.model_dir + "/save_" + str(epoch) + ".save", 'wb') as f_model:
                torch.save(model.state_dict(), f_model)

if __name__ == "__main__":
    main()

import torch
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import Activation
from tagging.models.crf_tagger import CrfTagger

def opts2params(opts):
    params = {
        "dim_emb" : opts.dim_emb,
        "dim_hid" : opts.dim_hid,
        "num_enc_layers" : opts.num_enc_layers,
        "dropout" : opts.dropout,
        "with_bert" : opts.with_bert,
        "bert_name" : opts.bert_name,
        "bert_max_len" : opts.bert_max_len,
    }
    return params

def build(params, vocab):
    if params["with_bert"]:
        from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
        embedding = PretrainedTransformerMismatchedEmbedder(model_name=params["bert_name"],
            max_length=params["bert_max_len"])
    else:
        from allennlp.modules.token_embedders import Embedding
        embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=params["dim_emb"])
    embedder = BasicTextFieldEmbedder({"tokens": embedding})
    encoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(params["dim_emb"], params["dim_hid"], params["num_enc_layers"], dropout=params["dropout"], bidirectional=True, batch_first=True))
    model = CrfTagger(
                 vocab=vocab,
                 text_field_embedder=embedder,
                 encoder=encoder,
                 dropout=params["dropout"]
            )
    return model

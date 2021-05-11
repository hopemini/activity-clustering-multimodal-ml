import os
import sys
import json
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

from models.trainer import Trainer
from models.seq2seq import Seq2seq
from loss.loss import Perplexity
from optim.optim import Optimizer
from dataset import fields
from evaluator.predictor import Predictor

import numpy as np
import argparse

par = argparse.ArgumentParser()
par.add_argument("-i", "--iteration", default=5,
                 type=int, help="number of iteration")
args = par.parse_args()

for _iter in range(args.iteration):
    train_path = "data/train_data.txt"
    log_path = "log/pth/rico_data_autoencoder_128_h64_" + str(_iter)
    config_path = "models/config.json"

    '''
    config = { "max_len": 150,
               "embedding_size": 16,
               "hidden_size": 50,
               "input_dropout_p": 0,
               "dropout_p": 0,
               "n_layers": 1,
               "bidirectional": True,
               "rnn_cell": "lstm",
               "embedding": None,
               "update_embedding": False,
               "get_context_vector": False,
               "use_attention": true }
    '''

    optimizer = "Adam"
    seq2seq = None
    config_json = open(config_path).read()
    config = json.loads(config_json)
    print(json.dumps(config, indent=4))

    save_23_path = "../../data/seq2seq_23_" + str(_iter) + "/"
    if not os.path.exists(save_23_path):
        os.mkdir(save_23_path)
    save_23_path = save_23_path + "seq2seq_23_" + str(_iter)

    save_34_path = "../../data/seq2seq_34_" + str(_iter) + "/"
    if not os.path.exists(save_34_path):
        os.mkdir(save_34_path)
    save_34_path = save_34_path + "seq2seq_34_" + str(_iter)

    # Prepare dataset
    src = fields.SourceField()
    tgt = fields.TargetField()
    srcp = fields.SourceField()
    tgtp = fields.TargetField()
    max_len = config["max_len"]
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=[('src', src), ('srcp', srcp), ('tgt', tgt), ('tgtp', tgtp)],
        filter_pred=len_filter
    )

    src.build_vocab(train)
    tgt.build_vocab(train)
    srcp.build_vocab(train)
    tgtp.build_vocab(train)
    input_vocab = src.vocab
    output_vocab = tgt.vocab
    input_part_vocab = srcp.vocab
    output_part_vocab = tgtp.vocab

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    # Initialize model
    seq2seq = Seq2seq(config, len(src.vocab), len(tgt.vocab), tgt.sos_id, tgt.eos_id)
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    seq2seq.load_state_dict(torch.load(log_path))
    seq2seq.eval()

    predictor = Predictor(seq2seq, input_vocab, input_part_vocab,
                          output_vocab, output_part_vocab)
    # Dataset Load
    lines_23 = open('data/test_23_data.txt').read().strip().split('\n')
    pairs_23 = [[s for s in l.split('\t')] for l in lines_23]

    # Predict
    try:
        rsult = []
        c = 0
        name_dic = {}
        name = []
        over_data = []
        for pair in tqdm(pairs_23):
            name.append(pair[0])
            seq = pair[1].split(" ")
            if len(seq) > 150:
                over_data.append(pair[0])
            partition = pair[2].split(" ")
            tgt_seq, tgt_att_list, encoder_outputs = predictor.predict(seq,partition)
            if pair[1] == " ".join(tgt_seq).replace(" <eos>", ""):
                c += 1

            data = encoder_outputs[len(encoder_outputs)-1].tolist()
            rsult.append(data)

        rsult = np.array(rsult)
        print("accuracy : %f" % (c/len(pairs_23)))
        np.save(save_23_path + "_data.npy", rsult)
        name_dic["name"] = name
        name_json = json.dumps(name_dic)
        name_file = open(save_23_path + "_names.json","w")
        name_file.write(name_json)
        name_file.close()

    except KeyboardInterrupt:
        pass

    # Dataset Load
    lines_34 = open('data/test_34_data.txt').read().strip().split('\n')
    pairs_34 = [[s for s in l.split('\t')] for l in lines_34]

    # Predict
    try:
        rsult = []
        c = 0
        name_dic = {}
        name = []
        over_data = []
        for pair in tqdm(pairs_34):
            name.append(pair[0])
            seq = pair[1].split(" ")
            if len(seq) > 150:
                over_data.append(pair[0])
            partition = pair[2].split(" ")
            tgt_seq, tgt_att_list, encoder_outputs = predictor.predict(seq,partition)
            if pair[1] == " ".join(tgt_seq).replace(" <eos>", ""):
                c += 1

            data = encoder_outputs[len(encoder_outputs)-1].tolist()
            rsult.append(data)

        rsult = np.array(rsult)
        print("accuracy : %f" % (c/len(pairs_34)))
        np.save(save_34_path + "_data.npy", rsult)
        name_dic["name"] = name
        name_json = json.dumps(name_dic)
        name_file = open(save_34_path + "_names.json","w")
        name_file.write(name_json)
        name_file.close()

    except KeyboardInterrupt:
        pass


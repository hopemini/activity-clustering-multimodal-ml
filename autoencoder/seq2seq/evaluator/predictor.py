import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self, model, src_vocab, srcp_vocab, tgt_vocab, tgtp_vocab,):
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.srcp_vocab = srcp_vocab
        self.tgtp_vocab = tgtp_vocab

    def get_decoder_features(self, src_seq, srcp_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        srcp_id_seq = torch.LongTensor([self.srcp_vocab.stoi[tok] for tok in srcp_seq]).view(1, -1)
        tgtp_seq = srcp_seq
        tgtp_seq.insert(0,'<sos>')
        tgtp_seq.append('<eos>')
        for i in range(150-len(tgtp_seq)):
            tgtp_seq.append("")
        tgtp_id_seq = torch.LongTensor([self.tgtp_vocab.stoi[tok] for tok in tgtp_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
            srcp_id_seq = srcp_id_seq.cuda()
            tgtp_id_seq = tgtp_id_seq.cuda()

        with torch.no_grad():
            softmax_list, _, other = self.model(src_id_seq, input_part=srcp_id_seq,
                    target_part=tgtp_id_seq)

        return other

    def predict(self, src_seq, srcp_seq):
        other = self.get_decoder_features(src_seq, srcp_seq)

        length = other['length'][0]

        tgt_att_list = []
        encoder_outputs = []
        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        if 'attention_score' in list(other.keys()):
            tgt_att_list = [other['attention_score'][di][0].data[0].cpu().numpy() for di in range(length)]
            encoder_outputs = other['encoder_outputs'].cpu().numpy()

        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, tgt_att_list, encoder_outputs

    def predict_n(self, src_seq, n=1):
        other = self.get_decoder_features(src_seq)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result

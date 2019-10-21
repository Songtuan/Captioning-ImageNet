import torch
import torch.nn as nn
import allennlp.nn.beam_search as allen_beam_search

from functools import partial
import modules.beam_search as BeamSearch
from modules import *


class ShowTellModel(nn.Module):
    def __init__(self, vocab, attention_size=512, hidden_size=512, embed_dim=None,
                 pretrained_embedding=None, feature_size=2048, state_machine=None,
                 beam_size=3, seq_length=15):
        super(ShowTellModel, self).__init__()
        self.seq_length = seq_length
        self.vocab = vocab
        self.beam_size = beam_size
        self.encoder = Resnet(encoded_image_size=14)
        self.state_machine = state_machine if state_machine is not None else \
            BeamSearch.BeamSearch.build_default_state_machine(self.vocab)
        vocab_size = len(self.vocab)
        self.decoder_cell = DecoderAttCell(encoder_dim=feature_size, attention_dim=attention_size,
                                           embed_dim=embed_dim, decoder_dim=hidden_size, vocab_size=vocab_size,
                                           pretrained_embedding=pretrained_embedding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'], reduction='sum')

    def load_state_machine(self, state_machine):
        self.state_machine = state_machine

    def load_decoder(self, PATH):
        self.decoder_cell.load_state_dict(torch.load(PATH))

    def load_encoder(self, PATH):
        self.encoder.load_state_dict(torch.load(PATH))

    def step(self, tokens, states, img_feature):
        h = states['h']
        c = states['c']
        log_probs, logits, (h, c) = self.decoder_cell(tokens, img_feature, h, c)
        states['h'] = h
        states['c'] = c
        return log_probs, logits, states
        # return log_probs, states

    def forward(self, imgs, captions=None):
        # shape (batch_size, 2048, 14, 14), where 2048 is feature_size, 14 is projection size
        img_feature = self.encoder(imgs)

        if self.training:
            assert captions is not None, 'ground true captions is required during training'

            states = {'h': None, 'c': None}
            max_length = captions.shape[-1]
            loss = 0
            out_log_probs = []
            # test_logits = []  # used for test
            for t in range(max_length - 1):
                tokens = captions[:, t]
                log_probs, logits, states = self.step(tokens, states, img_feature)
                # test_logits.append(logits)

                loss += self.criterion(logits, captions[:, t])
                out_log_probs.append(log_probs)

            return out_log_probs, loss
        else:
            batch_size = img_feature.shape[0]
            done_beam = [[] for _ in range(batch_size)]
            beam_search = BeamSearch.BeamSearch(beam_size=3, state_machine=self.state_machine,
                                                end_token_idx=self.vocab['<end>'],
                                                seq_length=self.seq_length, vocab=self.vocab)

            beam_num = len(self.state_machine.get_states())
            seq = torch.LongTensor(self.seq_length, batch_size).zero_()
            seq_logprobs = torch.FloatTensor(self.seq_length, batch_size)

            for k in range(batch_size):
                img_feature_temp = img_feature[k:k + 1, :, :, :].expand(self.beam_size * beam_num,
                                                                        img_feature.shape[1], img_feature.shape[2],
                                                                        img_feature.shape[3])
                init_token = torch.tensor([self.vocab['<start>']]).expand(self.beam_size * beam_num).cuda()
                states = {'h': None, 'c': None}
                log_probs, _, states = self.step(init_token, states, img_feature_temp)
                # log_probs, states = self.step(init_token, states, img_feature_temp)

                get_logprobs = partial(self.step, img_feature=img_feature_temp)

                done_beam[k] = beam_search.search(hidden_states=states, log_probs=log_probs, get_logprobs=get_logprobs)

                seq[:, k] = done_beam[k][0]['seq']
                seq_logprobs[:, k] = done_beam[k][0]['logps']

            return seq, seq_logprobs

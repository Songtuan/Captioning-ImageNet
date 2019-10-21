import torch
import torch.nn as nn
import allennlp.nn.beam_search as allen_beam_search

from modules.updown_cell import UpDownCell

from functools import partial


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class UpDownCaptioner(nn.Module):
    def __init__(self, vocab, image_feature_size=2048, embedding_size=1000, hidden_size=1200,
                 attention_projection_size=768, seq_length=20, beam_size=3,
                 pretrained_embedding=None, state_machine=None):
        super(UpDownCaptioner, self).__init__()

        vocab_size = len(vocab)
        self.vocab = vocab
        self.seq_length = seq_length
        self.state_machine = state_machine
        self.image_feature_size = image_feature_size
        self.beam_size = beam_size

        # define up-down cell
        self._updown_cell = UpDownCell(image_feature_size=image_feature_size, embedding_size=embedding_size,
                                       hidden_size=hidden_size, attention_projection_size=attention_projection_size)
        # define embedding layer
        if pretrained_embedding is not None:
            # if use pre-trained word embedding
            self._embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding).float()
        else:
            self._embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                                 embedding_dim=embedding_size)

        # produce the logits which used to soft-max distribution
        self._output_layer = nn.Linear(hidden_size, vocab_size, bias=True)
        self._log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<unk>'])

    def _step(self, tokens, states, image_features):
        '''
        Implement single decode step
        :param image_features(torch.Tensor): image features produced by encoder,
        a tensor with shape (batch_size, num_boxes, feature_size)
        :param tokens(torch.Tensor): input tokens, a tensor with shape (batch_size)
        :param states(Dict[str, torch.Tensor]): a dict contains previous hidden state
        :return: a tuple (torch.Tensorm Dict[str, torch.Tensor])
        '''
        if image_features.shape[0] != tokens.shape[0]:
            batch_size, num_boxes, image_feature_size = image_features.shape
            net_beam_size = int(tokens.shape[0] / batch_size)
            assert net_beam_size == self.beam_size
            image_features = image_features.unsqueeze(1).repeat(1, net_beam_size, 1, 1)
            # batch_size, beam_size, num_boxes, image_feature_size = image_features.shape
            image_features = image_features.view(batch_size * net_beam_size, num_boxes, image_feature_size)

        token_embeddings = self._embedding_layer(tokens)
        logits, states = self._updown_cell(image_features, token_embeddings, states)
        logits = self._output_layer(logits)
        log_probs = self._log_softmax(logits)

        if self.training:
            # in training mode, we need logits to calculate loss
            return logits, states
        else:
            # in eval mode, we need log_probs distribution of words
            return log_probs, states

    def forward(self, image_features, targets=None):
        '''
        Implement forward propagation
        :param image_features(torch.Tensor): image features produced by encoder, a tensor
        with shape (batch_size, num_boxes, feature_size)
        :param targets(torch.Tensor): ground-true captions, a tensor with shape (batch_size, max_length)
        :return:
        '''
        output = {}
        batch_size = image_features.shape[0]
        states = None

        if self.training:
            # in training mode, ground-true targets should not be None
            assert targets is not None
            # max decoder step we need to perform
            max_step = targets.shape[-1] - 1
            # a tensor contains logits of each step
            logits_seq = torch.zeros(max_step, batch_size, len(self.vocab)).to(device)
            # we transpose targets tensor to shape (max_length, batch_size)
            # this is useful when we calculate loss
            targets = targets.permute(1, 0)

            for t in range(max_step):
                # perform decode step
                tokens = targets[t, :]
                # logits should has shape (batch_size, vocab_size)
                logits, states = self._step(image_features=image_features, tokens=tokens, states=states)
                # update logits_seq
                logits_seq[t] = logits

            # the ground-true targets should exclude the first token
            # '<start>' since out model do not produce this token at
            # the beginning of sequence
            gt = targets[1:, :]
            # we need to force the logits_seq has shape (batch_size * max_step, vocab_size)
            # and ground-true caption has shape (batch_size * max_step) so that them can
            # be accepted as arguments in softmax criterion
            gt = gt.reshape(-1)
            logits_seq = logits_seq.view(-1, len(self.vocab))
            loss = self.criterion(logits_seq, gt)

            # add loss to output dict
            output['loss'] = loss
        else:
            end_index = self.vocab['<boundary>'] if '<boundary>' in self.vocab else self.vocab['<end>']
            start_index = self.vocab['<boundary>'] if '<boundary>' in self.vocab else self.vocab['<start>']
            beam_search = allen_beam_search.BeamSearch(end_index=end_index,
                                                       max_steps=self.seq_length, beam_size=self.beam_size,
                                                       per_node_beam_size=self.beam_size)
            # TODO: using to(device) instead of .cuda()
            # init_tokens = torch.tensor([start_index]).expand(batch_size).cuda()
            init_tokens = torch.tensor([start_index]).expand(batch_size).to(device)
            step = partial(self._step, image_features=image_features)
            top_k_preds, log_probs = beam_search.search(start_predictions=init_tokens, start_state=states, step=step)
            preds = top_k_preds[:, 0, :]
            output['seq'] = preds

        return output

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH)['model'])

from abc import ABC

from modules import *
from modules.captioner import Captioner


class ShowTellCaptioner(Captioner):
    def __init__(self, vocab, attention_size=512, hidden_size=512, embed_dim=None,
                 pretrained_embedding=None, feature_size=2048, state_machine=None,
                 beam_size=3, seq_length=15):
        super(ShowTellCaptioner, self).__init__()
        vocab_size = len(vocab)
        self.seq_length = seq_length
        self.vocab = vocab
        self.beam_size = beam_size
        self.state_machine = state_machine
        self._cell = DecoderAttCell(encoder_dim=feature_size, attention_dim=attention_size,
                                    embed_dim=embed_dim, decoder_dim=hidden_size, vocab_size=vocab_size,
                                    pretrained_embedding=pretrained_embedding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])

    def load(self, PATH):
        self._cell.load_state_dict(torch.load(PATH))

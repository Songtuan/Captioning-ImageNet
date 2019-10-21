import torch
import torch.nn as nn
import torchvision


class Resnet(nn.Module):
    def __init__(self, encoded_image_size=None, fine_tune=False):
        '''
        Construct a resnet101 model and exclude the last pooling and linear layer
        :param encoded_image_size: Add an extra pooling layer to force the outputs
        have size (channels, projection_size, project_size). If None, do not add
        this layer
        :param fine_tune: Determine whether the model need to be fine-tune
        '''
        super(Resnet, self).__init__()
        resnet_base = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet_base.children())[: -2])

        self.projection_size = encoded_image_size
        if encoded_image_size is not None:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        if not fine_tune:
            # if we disable fine-tune, we need to freeze the parameters in resnet model
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, imgs):
        '''
        Propagate steps
        :param imgs: input images with expected shape (B, C, H, W), where B is batch
        size, C indicates channels, H and W are height and width respectively.
        :return: extracted features with size (B, 2048, H_new, W_new)
        '''
        features = self.resnet(imgs)
        if self.projection_size is not None:
            features = self.adaptive_pool(features)
        return features


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderAttCell(nn.Module):
    def __init__(self, encoder_dim, attention_dim, embed_dim, decoder_dim, vocab_size=None, pretrained_embedding=None):
        '''

        :param encoder_dim: size of extracted features
        :param attention_dim: intermediate projection size used to calculated attention
        :param embed_dim: size of word embedding
        :param decoder_dim: hidden size of LSTM
        :param pretrained_embedding: pre-trained word embedding matrix
        '''
        super(DecoderAttCell, self).__init__()
        if pretrained_embedding is not None:
            embed_dim = pretrained_embedding.shape[-1]
            self.embedding = self._load_embedding(pretrained_embedding=pretrained_embedding)
        else:
            assert vocab_size is not None and embed_dim is not None, \
                'vocab size and embedding size cannot be None if pre-trained embedding is not provided'
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        self.feature_size = encoder_dim
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim=decoder_dim, attention_dim=attention_dim)
        self.decode_step = nn.LSTMCell(input_size=encoder_dim + embed_dim, hidden_size=decoder_dim)
        self.init_h = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)
        self.init_c = nn.Linear(in_features=encoder_dim, out_features=decoder_dim)
        self.f_beta = nn.Linear(in_features=decoder_dim, out_features=encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_features=decoder_dim, out_features=vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, tokens, img_feature, states=None):
        word_features = self.embedding(tokens)

        feature_size = img_feature.shape[1]
        batch_size = img_feature.shape[0]
        assert feature_size == self.feature_size

        # reshape img_feature to have shape:
        # (batch_size, num_of_features/num_of_pixels, feature_size)
        img_feature = img_feature.permute(0, 2, 3, 1)
        img_feature = img_feature.view(batch_size, -1, feature_size)

        if states is None:
            # if h and c is None, initialize them by using linear layer
            h, c = self.init_hidden(img_feature)
        else:
            h = states['h']
            c = states['c']

        # calculate attention base on image feature and prev hidden states
        attention_features, _ = self.attention(img_feature, h)
        gate = self.sigmoid(self.f_beta(h))
        attention_features = gate * attention_features

        lstm_input = torch.cat((word_features, attention_features), dim=1)
        h, c = self.decode_step(lstm_input, (h, c))
        logits = self.fc(h)
        log_probs = self.log_softmax(logits)
        states = {'h': h, 'c': c}
        return log_probs, logits, states



    def init_hidden(self, img_feature):
        img_feature = img_feature.mean(dim=1)
        h = self.init_h(img_feature)
        c = self.init_c(img_feature)
        return h, c

    def _load_embedding(self, pretrained_embedding):
        embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        return embedding

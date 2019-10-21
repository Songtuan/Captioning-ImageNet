import torch.nn as nn


class CaptioningModel(nn.Module):
    def __init__(self, encoder, captioner):
        super(CaptioningModel, self).__init__()
        # encoder is used to produce image features
        self.encoder = encoder
        # captioner is used to generate image captions
        self.captioner = captioner

        # particularly, we do not want to fine-tune
        # encoder part
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, imgs, targets=None):
        self.encoder.eval()  # usually we do not need to fine-tune encoder

        # produce image features
        # if encoder is faster-rcnn img_features should has
        # shape: (batch_size, num_boxes, feature_size)
        # otherwise, img_features should has
        # shape: (batch_size, feature_height, feature_width)
        img_features = self.encoder(imgs)

        if self.training:
            # if in training mode, ground-true target captions
            # cannot be empty
            assert targets is not None

            # set the captioner to training mode
            self.captioner.train()

            # get the output dict of captioner
            output = self.captioner(img_features, targets)
            # in training mode, output dict must has
            # key 'loss' which contain the loss of captioner
            assert 'loss' in output
        else:
            # if in eval mode
            # set the captioner to eval mode
            self.captioner.eval()
            # get the output of captioner
            output = self.captioner(img_features)
            # in the eval mode, output dict must has
            # key 'seq'
            assert 'seq' in output

        return output

import torch
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.utils import load_state_dict_from_url

from collections import OrderedDict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list

model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}

class FasterRCNN_Encoder(nn.Module):
    def __init__(self, out_dim=None, fine_tune=False):
        super(FasterRCNN_Encoder, self).__init__()
        backbone = resnet_fpn_backbone('resnet50', False)
        self.faster_rcnn = FasterRCNN(backbone, num_classes=91, rpn_post_nms_top_n_train=200,
                                              rpn_post_nms_top_n_test=100)
        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'], progress=True)
        self.faster_rcnn.load_state_dict(state_dict)

        # modify the last linear layer of the ROI pooling if there is
        # a special requirement of output size
        if out_dim is not None:
            self.faster_rcnn.roi_heads.box_head.fc7 = nn.Linear(in_features=1024, out_features=out_dim)

        # in captioning task, we may not want fine-tune faster-rcnn model
        if not fine_tune:
            for param in self.faster_rcnn.parameters():
                param.requires_grad = False

    def forward(self, images, targets=None):
        '''
        Forward propagation of faster-rcnn encoder
        Args:
            images: List[Tensor], a list of image data
            targets: List[Tensor], a list of ground-truth bounding box data,
                     used only in fine-tune
        Returns:
            proposal features after ROI pooling and RPN loss
        '''
        images, targets = self.faster_rcnn.transform(images, targets)
        # the base features produced by backbone network, i.e. resnet50
        features = self.faster_rcnn.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        # proposals produced by RPN, i.e. the coordinates of bounding box
        # which contain foreground objects
        proposals, proposal_losses = self.faster_rcnn.rpn(images, features, targets)
        # get the corresponding features of the proposals produced by RPN and perform roi pooling
        box_features = self.faster_rcnn.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        # project the features to shape (batch_size, num_boxes, feature_dim)
        box_features = self.faster_rcnn.roi_heads.box_head(box_features)
        return box_features, proposal_losses


class MaskRCNN_Benchmark(nn.Module):
    def __init__(self):
        super(MaskRCNN_Benchmark, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.model = build_detection_model(cfg)
        # load the pre-trained model
        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        # make sure maskrcnn_benchmark is in eval mode
        self.model.eval()

    def forward(self, imgs):
        imgs = to_image_list(imgs, cfg.DATALOADER.SIZE_DIVISIBILITY)
        # the returned features of maskrcnn_benchmark is the result of
        # roi pooling without average pooling
        features, _ = self.model(imgs)
        # perform average pooling to get the target result of roi pooling
        features = self.avgpool(features)
        # reshape feature size to (batch_size, num_boxes, feature_size)
        features = features.view(features.shape[0], -1)
        features = features.view(-1, 100, 2048)

        return features


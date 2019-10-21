import os
import json
import torch
import argparse

from tqdm import tqdm

from allennlp.training.metrics import BLEU

from dataset import CaptionDataset
from modules.faster_rcnn import MaskRCNN_Benchmark
from modules.captioner.UpDownCaptioner import UpDownCaptioner
from models.CaptioningModel import CaptioningModel

from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg


# tensorboard summary writter
writer = SummaryWriter()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def eval_batch(batch_data, model, evaluator):
    """
    perform evaluation over one batch data
    :param batch_data: Dict[str, torch.Tensor], batch data read from data loader
    :param model: Captioning model
    :evaluator: allennlp.training.metrics.BLEU, evaluate bleu_4 score
    :return: 
    """
    model.eval()

    # get evaluation images and ground-true target
    images = batch_data['image']
    targets = batch_data['caption']
    images = images.to(device)
    targets = targets.to(device)

    output = model(images)
    predicts = output['seq']
    evaluator(predictions=predicts, gold_targets=targets)


def train_batch(batch_data, model, optimizer):
    """
    perform training on one single batch data
    :param batch_data: Dict[str, torch.Tensor], batch data read from data-loader
    :param model: Captioning model
    :param optimizer: torch.nn.optimizer
    :return: torch.Tensor loss in this batch
    """
    # get training images and ground-true target
    images = batch_data['image']
    targets = batch_data['caption']
    images = images.to(device)
    targets = targets.to(device)

    # force model in training mode
    # and clean the grads of model
    model.train()
    model.zero_grad()
    # get the output of captioning model
    # and make sure 'loss' is in the return dict
    output = model(images, targets)
    assert 'loss' in output

    # perform back-propagation
    # and clip the grads
    batch_loss = output['loss']
    batch_loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # add batch loss to tensorboard
    writer.add_scalar('loss', batch_loss)
    
    return batch_loss


parser = argparse.ArgumentParser('Training Parameters')
parser.add_argument('--epochs', type=int, default=30, help='setting the number of training epochs')
parser.add_argument('--lr', type=float, default=4e-4, help='setting the initial learning rate')
parser.add_argument('--batch_size', type=int, default=50, help='setting the batch size')
parser.add_argument('--check_point', default='UpDownCaptioner.pth', help='check point path')
parser.add_argument('--save_dir', default='CaptioningModel.pth', help='path to save the model')


if __name__ == '__main__':
    opt = parser.parse_args()
    epochs = opt.epochs
    lr = opt.lr
    batch_size = opt.batch_size
    check_point = opt.check_point
    save_dir = opt.save_dir

    # load vocabulary
    vocabulary_path = 'vocab/updown_vocab.json'
    with open(vocabulary_path, 'r') as v:
        vocab = json.load(v)

    # load training set
    training_set_path = os.path.join('data', 'TRAIN.hdf5')
    training_set = CaptionDataset(training_set_path)

    # load eval set
    eval_set_path = os.path.join('data', 'VAL.hdf5')
    eval_set = CaptionDataset(eval_set_path)

    # build data-loaders for both training set and eval set
    # make both of them iterable
    training_loader = DataLoader(dataset=training_set, batch_size=batch_size)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size)

    # build encoder
    encoder = MaskRCNN_Benchmark()

    # build decoder
    decoder = UpDownCaptioner(vocab=vocab)
    decoder.load(check_point)

    # build complete model
    model = CaptioningModel(encoder=encoder, captioner=decoder)
    model = model.to(device)

    # build optimizers
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    # initialize best bleu score
    # we will store the model which
    # reach has the best score
    bleu_score_best = 0

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0

        for batch_data in tqdm(training_loader):
            # perform training over one batch
            batch_loss = train_batch(batch_data=batch_data, model=model, optimizer=optimizer)
            epoch_loss += batch_loss.item()
        
        epoch_loss = epoch_loss / len(training_loader)

        # perform evaluation
        evaluator = BLEU(exclude_indices={vocab['<unk>'], vocab['<boundary>']})

        with torch.no_grad():
            for batch_data in eval_loader:
                eval_batch(batch_data=batch_data, model=model, evaluator=evaluator)

        bleu_score = evaluator.get_meric()
        if bleu_score > bleu_score_best:
            bleu_score_best = bleu_score
            torch.save(model.state_dict(), save_dir)
        
        print('\n')
        print('Epoch: {0:2d} | Epoch Loss: {1:7.3f} | BLEU_4 Score: {2:5.2f}'.format(epoch, epoch_loss, bleu_score))



        




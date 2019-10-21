import h5py
import json
import os
import numpy as np
import argparse

from PIL import Image
from random import seed, choice, sample
from tqdm import tqdm



def create_input_file(vocab, data_name='flickr8k', data_split_file='dataset_flickr8k.json',
                      image_folder=None, captions_per_img=1):

    with open(data_split_file) as j:
        formatted_input = json.load(j)

    train_imgs = []  # record the path to training images
    train_imgs_caps = []  # record the captions associate with training images

    val_imgs = []  # record the path to val images
    val_imgs_caps = []  # record the captions associate with val images

    test_imgs = []  # record the path to testing images
    test_imgs_caps = []  # record the captions associate with testing images

    max_cap_length = 0  # used to record the max caption length

    unk_token = 0

    for img in formatted_input['images']:
        # iterate through each image in dataset
        captions = []
        file_name = os.path.join(image_folder, img['filename']) if data_name != 'coco' else \
            os.path.join(image_folder, img['filepath'], img['filename'])
        for c in img['sentences']:
            # record each caption
            captions.append(c['tokens'])
            if len(c['tokens']) > max_cap_length:
                max_cap_length = len(c['tokens'])

        if img['split'] in {'train', 'restval'}:
            # if current image is training image
            # we record both image path and captions
            train_imgs.append(file_name)
            train_imgs_caps.append(captions)
        elif img['split'] == 'val':
            # if current image is val image
            # we record both image path and captions
            val_imgs.append(file_name)
            val_imgs_caps.append(captions)
        else:
            # if current image is testing image
            # we record both image path and captions
            test_imgs.append(file_name)
            test_imgs_caps.append(captions)

    assert max_cap_length > 0
    max_cap_length += 1  # the 1 add here is hold for 'EOS' token

    seed(123)
    for imgs, caps, split in [(train_imgs, train_imgs_caps, 'TRAIN'),
                              (val_imgs, val_imgs_caps, 'VAL'),
                              (test_imgs, test_imgs_caps, 'TEST')]:
        # create hdf5 file for each TRAIN, VAL and TEST dataset

        with h5py.File(split + '.hdf5') as h:
            # create dataset to store images' data
            imgs_dataset = h.create_dataset('images', (len(imgs), 600, 800, 3), dtype='uint8')
            # dt refer to a data type which has variable size
            # images_list = []
            # dt = h5py.vlen_dtype(np.dtype('float32'))
            # imgs_dataset = h.create_dataset('images', (len(imgs), ), dtype=dt)
            # create dataset to store captions, which should have size
            # (num_images * num_captions_per_img, max_cap_len + 1), the 1 add after max_cap_len is hold for
            # 'SOS' token (start of sentence)
            caps_dataset = h.create_dataset('captions', (len(imgs) * captions_per_img, max_cap_length + 1), dtype=int)
            # test dataset to store original caption tokens
            caps_unencode_dataset = h.create_dataset('captions_uncode',
                                                     (len(imgs) * captions_per_img, max_cap_length + 1), dtype='S10')

            h.attrs['captions_per_image'] = captions_per_img

            for idx, img_name in enumerate(tqdm(imgs)):
                # iterate through each image in split dataset
                # read the image data
                pil_image = Image.open(img_name).convert('RGB')
                pil_image = pil_image.resize((800, 600))
                img = np.array(pil_image)[:, :, [2, 1, 0]]
                # record img data
                # imgs_dataset[idx] = img
                assert img.shape[-1] == 3
                imgs_dataset[idx] = img

                # we copy the captions associate with current image
                img_caps = caps[idx].copy()
                img_caps_uncode = caps[idx].copy()

                # sample the captions to make sure each image associates with captions_per_img number
                # of captions
                if len(img_caps) < captions_per_img:
                    img_caps = img_caps + [choice(img_caps) for _ in range(captions_per_img - len(img_caps))]
                else:
                    img_caps = sample(img_caps, k=captions_per_img)

                if len(img_caps_uncode) < captions_per_img:
                    img_caps_uncode = img_caps_uncode + [choice(img_caps_uncode)
                                                         for _ in range(captions_per_img - len(img_caps_uncode))]

                assert len(img_caps) == captions_per_img

                for j, cap in enumerate(img_caps):
                    # iterate through each caption
                    # reformat each caption so that 'PAD' token
                    # appear after 'EOS' token and each caption
                    # has length max_cap_length
                    formatted_cap = [vocab['<unk>']] * max_cap_length
                    formatted_cap_uncode = ['<unk>'] * max_cap_length
                    for k in range(len(cap)):
                        if cap[k] not in vocab:
                            formatted_cap[k] = vocab['<unk>']
                            unk_token += 1
                        else:
                            formatted_cap[k] = vocab[cap[k]]
                        formatted_cap_uncode[k] = cap[k]
                    formatted_cap[len(cap)] = vocab['<boundary>']
                    formatted_cap = [vocab['<boundary>']] + formatted_cap

                    formatted_cap_uncode[len(cap)] = '<boundary>'
                    formatted_cap_uncode = ['<boundary>'] + formatted_cap_uncode

                    img_caps[j] = np.array(formatted_cap)
                    img_caps_uncode[j] = np.string_(formatted_cap_uncode)

                for j in range(captions_per_img):
                    # calculate the total indices of current caption
                    cap_total_idx = idx * captions_per_img + j
                    caps_dataset[cap_total_idx] = img_caps[j]
                    caps_unencode_dataset[cap_total_idx] = img_caps_uncode[j]

            # h.create_dataset('images', data=images_list, dtype=dt)

    print(unk_token)


parser = argparse.ArgumentParser('create dataset files')
parser.add_argument('--data_name', default='coco', help='name of dataset')
parser.add_argument('--data_split_file', default='dataset_coco.json', help='data split file to be read')
parser.add_argument('--vocab_path', default='updown_vocab.json', help='vocabulary')
parser.add_argument('--image_folder', default='images', help='folder which contain images')

dir_main = os.path.join(__file__, '../..')

if __name__ == '__main__':
    opt = parser.parse_args()
    vocab_path = os.path.join(dir_main, 'vocab', opt.vocab_path)
    with open(vocab_path, 'r') as v:
        vocab = json.load(v)
    # vocab = vocab.get_token_to_index_vocabulary()
    create_input_file(vocab=vocab, data_name=opt.data_name,
                      data_split_file=opt.data_split_file, image_folder=opt.image_folder)

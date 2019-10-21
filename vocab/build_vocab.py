import numpy as np
import json


def build():
    with open('glove.6B.50d.txt', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        embedding = np.zeros([len(lines) + 4, 50])
        for idx, line in enumerate(lines):
            line = line.strip('\n').split(' ')
            word = line[0]
            vector = np.array(line[1:]).astype(np.float)
            vocab[word] = idx + 4
            embedding[idx, :] = vector
        embedding[3] = np.mean(embedding, axis=0)
        assert embedding.shape == (len(vocab), 50)
        print(embedding[5])
    return vocab, embedding


if __name__ == '__main__':
    vocab, embedding = build()
    with open('vocab.json', 'w') as j:
        json.dump(vocab, j)
    np.save('embedding', embedding)


from beam_state_machine import *
from state_machine import *
from nltk.corpus import wordnet as wn


def decode_str(vocab, cap):
    """
    map a caption to words
    :param vocab: the input vocabulary
    :param cap: caption, list
    :return: list
    """
    caption = []
    id_to_word = {vocab[word]: word for word in vocab.keys()}
    for token_id in cap:
        if token_id not in [vocab['<start>'], vocab['<end>'], vocab['<pad>']]:
            caption.append(id_to_word[token_id])
    return caption


def map_to_wordnet(target_id, wordnet_file):
    with open(wordnet_file) as f:
        for line in f:
            wordnet_id, words = line.split('\t')
            if wordnet_id == target_id:
                return words.strip().split(',')
    return None


def get_hypernyms(wordnet_id):
    '''
    get all hypernyms of a specific wordnet id

    Args:
        wordnet_id: a wordnet id

    Returns:
        phrases: a list of phrases(lemma names) which are the hypernyms of input wordner id
    '''

    def get(syns):
        '''
        a helper function which get all the hypernyms of input synsets recursively

        Args:
            syns: the list input synsets

        Returns:
            hypers: the list of all hypernyms
        '''
        if syns == []:
            # if the current synset is empty, finish search
            return []
        else:
            hypers = []
            for syn in syns:
                # get the hypernyms of each synset in the list
                hypers += syn.hypernyms()

            hypers += get(hypers)  # get the hypernyms of current hypernyms

            return hypers

    syn = wn.synset_from_pos_and_offset(wordnet_id[0], int(wordnet_id[1:]))
    hypers = get([syn])
    phrases = syn.lemma_names()
    for hyper in hypers:
        phrases = phrases + hyper.lemma_names()

    for idx, phrase in enumerate(phrases):
        phrases[idx] = phrase.replace('_', ' ')

    for idx, phrase in enumerate(phrases):
        phrases[idx] = phrase.lower()

    return phrases


def build_state_machine(phases, vocab):
    id_to_word = {vocab[word]: word for word in vocab.keys()}
    vocab_idx = [int(idx) for idx in id_to_word.keys()]  # the list of word ids, e.g. [1, 2, 3, 4, ..., vocab_size]
    state_machine = StateMachine(events={'input': InputEvent()})
    state_machine.add_state('init')
    state_machine.add_state('final')
    cond_table = {('init', 'final'): TransitCondition(), ('init', 'init'): TransitCondition(vocab_idx[:]),
                  ('final', 'final'): TransitCondition(vocab_idx[:])}
    cond_word = {('init', 'final'): [], ('init', 'init'): list(vocab.keys()),
                 ('final', 'final'): list(vocab.keys())}
    num_mid_states = 0
    state_idx_mapping = {0: 'init'}

    for phase in phases:
        words = phase.split()
        prev_s = 'init'

        for i, word in enumerate(words):

            if word not in vocab:
                break
            word_id = vocab[word]
            if i == len(words) - 1:
                current_s = 'final'
            else:
                num_mid_states += 1
                current_s = 'mid_{}'.format(num_mid_states)
                state_idx_mapping[num_mid_states] = current_s

            state_machine.add_state(current_s)

            if (prev_s, current_s) in cond_table:
                cond_table[(prev_s, current_s)].update(word_id)
                cond_word[(prev_s, current_s)].append(word)
            else:
                cond_table[(prev_s, current_s)] = TransitCondition([word_id])
                cond_word[(prev_s, current_s)] = [word]

            if prev_s == 'init':
                cond_table[('init', 'init')].update(word_id, is_remove=True)
                if word in cond_word[('init', 'init')]:
                    cond_word[('init', 'init')].remove(word)
            else:
                if (prev_s, 'init') not in cond_table:
                    cond_table[(prev_s, 'init')] = TransitCondition(vocab_idx[:])
                    cond_word[(prev_s, 'init')] = list(vocab.keys())
                cond_table[(prev_s, 'init')].update(word_id, is_remove=True)
                if word in cond_word[(prev_s, 'init')]:
                    cond_word[(prev_s, 'init')].remove(word)

            prev_s = current_s

    state_idx_mapping[num_mid_states + 1] = 'final'

    for source_s, dest_s in cond_table:
        state_machine.add_transition(source_s, dest_s, 'input', cond_table[(source_s, dest_s)])

    return state_machine, state_idx_mapping

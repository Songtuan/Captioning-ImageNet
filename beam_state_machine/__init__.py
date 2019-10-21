from state_machine import *


class TransitCondition(Condition):
    def __init__(self, word_ids=None):
        super().__init__()
        self.word_ids = word_ids if word_ids is not None else []
        if word_ids is not None:
            assert type(word_ids) is list, 'initial word_ids must be list'

    def update(self, word_id, is_remove=False):
        if is_remove:
            if word_id in self.word_ids:
                self.word_ids.remove(word_id)
        else:
            self.word_ids.append(word_id)

    def _eval(self, inputs):
        return True if inputs.val in self.word_ids else False


class InputEvent(Event):
    def __init__(self, val=None):
        self.val = val if val is not None else None

    def inputs(self, val):
        self.val = val
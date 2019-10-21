import argparse
import json
import os
from typing import Dict, List

from mypy_extensions import TypedDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Build a vocabulary out of COCO train2017 captions json file."
)

parser.add_argument(
    "-c",
    "--captions-jsonpath",
    default="captions_train2017.json",
    help="Path to COCO train2017 captions json file.",
)
parser.add_argument("-t", "--word-count-threshold", type=int, default=5)
parser.add_argument(
    "-o",
    "--output-dirpath",
    default="updown_vocab.json",
    help="Path to a (non-existent directory to save the vocabulary.",
)


# ------------------------------------------------------------------------------------------------
# All the punctuations in COCO captions, we will remove them.
# fmt: off
PUNCTUATIONS: List[str] = [
    "''", "'", "``", "`", "(", ")", "{", "}", ".", "?", "!", ",", ":", "-", "--", "...", ";"
]
# fmt: on

# Special tokens which should be added (all, or a subset) to the vocabulary.
# We use the same token for @@PADDING@@ and @@UNKNOWN@@ -- @@UNKNOWN@@.
SPECIAL_TOKENS: List[str] = ['<unk>', '<boundary>']

# Type for each COCO caption example annotation.
CocoCaptionExample = TypedDict("CocoCaptionExample", {"id": int, "image_id": int, "caption": str})
# ------------------------------------------------------------------------------------------------


def build_caption_vocabulary(
    caption_json: List[CocoCaptionExample], word_count_threshold: int = 5
) -> List[str]:
    r"""
    Given a list of COCO caption examples, return a list of unique captions tokens thresholded
    by minimum occurence.
    """

    word_counts: Dict[str, int] = {}

    # Accumulate unique caption tokens from all caption sequences.
    for item in tqdm(caption_json):
        caption: str = item["caption"].lower().strip()
        caption_tokens: List[str] = word_tokenize(caption)
        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]

        for token in caption_tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1

    all_caption_tokens = sorted(
        [key for key in word_counts if word_counts[key] >= word_count_threshold]
    )
    caption_vocabulary = sorted(list(all_caption_tokens))
    # caption_vocabulary = {token: idx for idx, token in enumerate(caption_vocabulary)}
    return caption_vocabulary


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Loading annotations json from {args.captions_jsonpath}...")
    captions_json = json.load(open(args.captions_jsonpath))["annotations"]

    print("Building caption vocabulary...")
    caption_vocabulary = build_caption_vocabulary(
        captions_json, args.word_count_threshold
    )
    caption_vocabulary = SPECIAL_TOKENS + caption_vocabulary
    print(f"Caption vocabulary size (with special tokens): {len(caption_vocabulary)}")
    caption_vocabulary = {token: idx for idx, token in enumerate(caption_vocabulary)}

    # Write the vocabulary to separate namespace files in directory.
    print('Writing the vocabulary to {args.output_dirpath}...')
    with open(args.output_dirpath, 'w') as j:
        json.dump(caption_vocabulary, j)
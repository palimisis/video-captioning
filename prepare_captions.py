import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_vocab(all_words, min_feq=1):
    # use collections.Counter() to build vocab
    all_words = all_words.most_common()
    word2ix = {"<pad>": 0, "<unk>": 1}
    for ix, (word, feq) in enumerate(tqdm(all_words, desc="building vocab"), start=2):
        if feq < min_feq:
            continue
        word2ix[word] = ix
    ix2word = {v: k for k, v in word2ix.items()}

    # output info
    print("number of words in vocab: {}".format(len(word2ix)))
    print("number of <unk> in vocab: {}".format(len(all_words) - len(word2ix)))

    return word2ix, ix2word


def parse_msvd_labels(file, captions_file, gts_file):
    data = []

    with open(file, "r") as file:
        for line_number, line in enumerate(file):
            if line_number < 8:
                continue  # Skip the first 8 lines
            line_data = line.strip().split(
                " ", 1
            )  # Split each line into two parts based on the first space
            data.append(line_data)

    # Create a DataFrame from the list
    df = pd.DataFrame(data, columns=["ID", "Label"])

    print("There are totally {} descriptions".format(len(df)))

    counter = Counter()
    captions = []
    filenames = []
    gts = {}  # for eval.py
    max_cap_ids = {}  # for eval.py
    for _, name, label in tqdm(df[["ID", "Label"]].itertuples(), desc="reading labels"):
        file_name = name  # + ".avi"
        filenames.append(file_name)
        # process caption
        tokenized = label.lower()
        tokenized = re.sub(r"[~\\/().!,;?:]", " ", tokenized)
        gts_token = tokenized
        tokenized = tokenized.split()
        tokenized = ["<sos>"] + tokenized + ["<eos>"]
        counter.update(tokenized)  # put words into a set
        captions.append(tokenized)
        # gts
        if file_name in gts:
            # if max_cap_ids[file_name] <= 10:
            max_cap_ids[file_name] += 1
            gts[file_name].append(
                {
                    "image_id": file_name,
                    "cap_id": max_cap_ids[file_name],
                    "caption": label,
                    "tokenized": gts_token,
                }
            )
        else:
            max_cap_ids[file_name] = 0
            gts[file_name] = [
                {
                    "image_id": file_name,
                    "cap_id": 0,
                    "caption": label,
                    "tokenized": gts_token,
                }
            ]

    # build vocab
    word2ix, ix2word = build_vocab(counter)

    # turn words into index (1 is <unk>)
    captions = [
        [word2ix.get(w, word2ix["<unk>"]) for w in caption]
        for caption in tqdm(captions, desc="turing words into index")
    ]

    # build dict   filename: [captions]
    caption_dict = {}
    for name, cap in zip(filenames, captions):
        if name not in caption_dict.keys():
            caption_dict[name] = []
        caption_dict[name].append(cap)

    print(len(caption_dict))

    # split dataset
    data_split = [1400, 450, -1]  # train valid test
    vid_names = list(caption_dict.keys())
    np.random.shuffle(vid_names)
    train_split = vid_names[: data_split[0]]
    valid_split = vid_names[data_split[0] : data_split[0] + data_split[1]]
    test_split = vid_names[data_split[0] + data_split[1] :]

    print(
        "train:{} valid:{} test:{}".format(
            len(train_split), len(valid_split), len(test_split)
        )
    )
    with open(gts_file, "w+", encoding="utf-8") as f:
        json.dump({"gts": gts}, f)

    # save files
    with open(captions_file, "w+", encoding="utf-8") as f:
        json.dump(
            {
                "word2ix": word2ix,
                "ix2word": ix2word,
                "captions": caption_dict,
                "splits": {
                    "train": train_split,
                    "valid": valid_split,
                    "test": test_split,
                },
            },
            f,
        )


if __name__ == "__main__":
    parse_msvd_labels(
        file=r"./msvd/AllVideoDescriptions.txt",
        captions_file=r"./msvd/captions.json",
        gts_file=r"./msvd/gts.json",
    )

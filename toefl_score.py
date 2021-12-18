import kenlm
import json
from urllib.parse import urlparse

from cc_net import text_normalizer, perplexity
from sentencepiece import SentencePieceProcessor


def count_token(lines):
    c = 0
    for line in lines:
        c += len(line.split())
    return c


model_file_path = '/mnt/nvme/files/nlp-archive/cc-net/data/lm_sp/en.arpa.bin'

model = kenlm.Model(model_file_path)

sp = SentencePieceProcessor()
sp.load("/mnt/nvme/files/nlp-archive/cc-net/data/lm_sp/en.sp.model")

# in_file = open("sample-text.txt", 'r')

# text = in_file.readline()

# normalized = text_normalizer.normalize(text)

# pieces = sp.encode_as_pieces(normalized)
# print(perplexity.pp(model.score(" ".join(pieces)), len(pieces)))


f = open('toefl.jsonl')
# text, title, url

avg_ppl_by_website = {}
website_count = {}
token_counts = {}

# 99th threshold: 3590
thresholds = [99999, 1570, 930, 660, 520, 440, 380, 340, 300, 250]

buckets = {}
for i in range(1, 11):
    buckets[i] = 0


for line in f.readlines():
    if not line or len(line) < 3:
        continue

    js_dict = json.loads(line.strip())
    prompt = js_dict['prompt']
    text = js_dict['text']

    token_count = count_token(text)

    lines = text.split('\n')
    lines = list(filter(bool, lines))

    article_score = 0.0
    article_length = 0
    for sent in lines:
        normalized = text_normalizer.normalize(sent)
        pieces = sp.encode_as_pieces(normalized)
        length = len(pieces)

        article_score += model.score(" ".join(pieces))
        article_length += length

    if article_length > 0 and prompt == "P8":
        ppl = perplexity.pp(article_score, article_length)

        for i in range(9, -1, -1):
            if ppl <= thresholds[i]:
                buckets[i+1] = buckets[i+1] + 1
                break


l = []
for i in range(1, 11):
    l.append(buckets[i])

print(l)
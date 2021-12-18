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


f = open('articles-high-dedup.txt')
# text, title, url

ppl_by_website = {}
website_count = {}
token_counts = {}

# 99th threshold: 3590
thresholds = [99999, 1570, 930, 660, 520, 440, 380, 340, 300, 250]

buckets = {}
for i in range(1, 11):
    buckets[i] = 0

high_ppl_text = open("bucket-10-texts.txt", "w+")

for line in f.readlines():
    if not line or len(line) < 3:
        continue

    js_dict = json.loads(line.strip())
    url = urlparse(js_dict['url']).netloc
    text = js_dict['text']

    token_count = count_token(text)

    if url in website_count.keys():
        website_count[url] = website_count.get(url) + 1
        token_counts[url] = token_counts.get(url) + token_count
    else:
        website_count[url] = 1
        token_counts[url] = token_count

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

    # count articles in different ppl buckets
    if article_length > 0:
        ppl = perplexity.pp(article_score, article_length)

        for i in range(9, -1, -1):
            if ppl > 1570:
                # check what text is in lowest bucket
                high_ppl_text.write(text + "\n")

            if ppl <= thresholds[i]:
                buckets[i+1] = buckets[i+1] + 1
                break

    # avg ppl by website
    if url in ppl_by_website.keys():
        ppl_by_website[url] = ppl_by_website.get(url) + ppl
    else:
        ppl_by_website[url] = ppl


final_ppl = {}
for address in website_count.keys():
    address_ppl = ppl_by_website[address] / website_count[address]
    final_ppl[address] = address_ppl

sorted_ppl = dict(sorted(final_ppl.items(), key=lambda item: item[1]))

out_file = open("ppl.txt", "w+")

for key in sorted_ppl.keys():
    if token_counts[key] > 5000:
        out_file.write(key + " " + str(sorted_ppl[key]) + " \t TOKENS: " + str(token_counts[key]) + "\n")


f.close()
out_file.close()

l = []
for i in range(1, 11):
    l.append(buckets[i])

print(l)

high_ppl_text.close()

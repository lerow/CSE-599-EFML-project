import hashlib
import json


input_file_path = "satire.txt"
output_file_path = "satire-dedup.txt"

completed_lines_hash = set()

output_file = open(output_file_path, "w")


for line in open(input_file_path, "r"):
    line = line.strip()
    if not line or len(line) < 3:
        continue

    js_dict = json.loads(line)
    text = js_dict['text']

    hashValue = hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    if (hashValue not in completed_lines_hash) and (len(text) > 200):
      output_file.write(line)
      output_file.write('\n')
      completed_lines_hash.add(hashValue)


output_file.close()

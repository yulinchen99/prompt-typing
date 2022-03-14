import json

def process(data):
    # data = "fewnerd"
    verbalizer = {}
    with open(f"../../data/{data}/tag_mapping.txt")as f:
        lines = f.readlines()
    for line in lines:
        label, label_words = line.strip().split("\t")
        verbalizer[label] = label_words.split("/")
    with open(f"./{data}/verbalizer.json", "w")as fw:
        json.dump(verbalizer, fw)

    with open(f"../../data/{data}/tags.txt")as f:
        lines = f.readlines()
        tags = [line.strip() for line in lines]
    with open(f"./{data}/labels.json", "w")as fw:
        json.dump(tags, fw)

# process("fewnerd")
# process("bbn")
# process("ontonote")
process("openentity")
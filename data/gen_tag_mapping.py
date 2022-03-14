# %%
TYPE_NUM_DICT = {"open": 10331, "onto": 89, "wiki": 4600, "kb": 130, "gen": 9}

with open("./release/ontology/types.txt", encoding="utf-8")as f:
    lines = f.readlines()
gen_tags = lines[:TYPE_NUM_DICT["gen"]]
fine_tags = lines[TYPE_NUM_DICT["gen"]:TYPE_NUM_DICT["kb"]]

new_lines = []
new_lines2 = []

for tags in [gen_tags, fine_tags]:
    for t in tags:
        words = t.strip().split("_")
        for w in words:
            if w not in new_lines2: #and w != "of" and w != "in":
                new_lines2.append(w)
                new_lines.append(w + "\t" + w)
    print(len(new_lines))

for line in lines[TYPE_NUM_DICT["kb"]:]:
    words = line.strip().split("_")
    for w in words:
        if w not in new_lines2 and w != "of" and w != "in":
            new_lines2.append(w)
            new_lines.append(w + "\t" + w)

# new_lines = list(set(new_lines))
    # new_lines.append("\t".join([line.strip(), line.strip().replace("_", "/")]))
    # if "_" in line:
    #     print(line.strip())
with open("./openentity/tag_mapping.txt", "w", encoding="utf-8")as fw:
    fw.writelines("\n".join(new_lines))
with open("./openentity/tags.txt", "w", encoding="utf-8")as fw:
    fw.writelines("\n".join(new_lines2))
# %%

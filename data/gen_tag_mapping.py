# %%
with open("./release/ontology/types.txt", encoding="utf-8")as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    new_lines.append("\t".join([line.strip(), line.strip().replace("_", "/")]))
    if "_" in line:
        print(line.strip())
with open("./openentity/tag_mapping.txt", "w", encoding="utf-8")as fw:
    fw.writelines("\n".join(new_lines))
# %%

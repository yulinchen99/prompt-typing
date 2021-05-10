def load_tag_list(filepath):
    with open(filepath)as f:
        lines = f.readlines()
        type_list = [line.strip() for line in lines]
    return type_list
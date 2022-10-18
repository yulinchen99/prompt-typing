import copy
def f1(p, r):
    if p == 0. or r == 0.:
        return 0.
    return 2*p*r/(p+r)

def label_path(t, isfewnerd=False):
    if isfewnerd:
        return [t.split('-')[0], t]
    else:
        types = t.split("/")
        if len(types) == 3:
            return ["/"+types[1], t]
        if len(types) == 4:
            return ["/"+types[1], "/"+types[1]+"/"+types[2], t]
        return [t]

def strict(labels, predictions):
    cnt = 0
    for label, pred in zip(labels, predictions):
        cnt += set(label) == set(pred)
    acc = cnt/len(labels)
    #print("Strict Accuracy: %s" % acc)
    return acc

def loose_macro(labels, predictions):
    p = 0.
    r = 0.
    for label, pred in zip(labels, predictions):
        label = set(label)
        pred = set(pred)
        if len(pred) > 0:
            p += len(label.intersection(pred))/len(pred)
        if len(label) > 0:
            r += len(label.intersection(pred))/len(label)
    p /= len(labels)
    r /= len(labels)
    f = f1(p, r)
    #print("Loose Macro:")
    #print("Precision %s Recall %s F1 %s" % (p, r, f))
    return p, r, f

def loose_micro(labels, predictions):
    cnt_pred = 0
    cnt_label = 0
    cnt_correct = 0
    for label, pred in zip(labels, predictions):
        label = set(label)
        pred = set(pred)
        cnt_pred += len(pred)
        cnt_label += len(label)
        cnt_correct += len(label.intersection(pred))
    if cnt_pred > 0:
        p = cnt_correct/cnt_pred
    else:
        p = 0.0
    r = cnt_correct/cnt_label
    f = f1(p, r)
    #print("Loose Micro:")
    #print("Precision %s Recall %s F1 %s" % (p, r, f))
    return p, r, f

def get_metrics(label, pred, idx2tag, isfewnerd=False):
    label = [label_path(idx2tag[l], isfewnerd) for l in label]
    pred = [label_path(idx2tag[p], isfewnerd) for p in pred]
    acc = strict(label, pred)
    macro = loose_macro(label, pred)
    micro = loose_micro(label, pred)
    return acc, {'p':micro[0], 'r':micro[1], 'f':micro[2]}, {'p':macro[0], 'r':macro[1], 'f':macro[2]}

def get_openentity_metrics(label, pred, idx2tag=None, string=False):
    if not string:
        label = [[idx2tag[l] for l in la] for la in label]
        pred = [[idx2tag[p] for p in pr] for pr in pred]
    # print(label)
    # print(pred)
    acc = strict(label, pred)
    macro = loose_macro(label, pred)
    micro = loose_micro(label, pred)
    return acc, {'p':micro[0], 'r':micro[1], 'f':micro[2]}, {'p':macro[0], 'r':macro[1], 'f':macro[2]}

def get_openentity_metrics_for_prompt(y_true, y_pred, idx2tag, idx2oritag, oritag2idx, ori_tag_list):
    # merge possible combinations in y_pred
    tri_word_tags = ["chief_executive_officer", "latter_day_saints"]
    tri_words = [w.split("_") for w in tri_word_tags]
    merged_pr_tags = []
    for pr in y_pred:
        pr_tag = [idx2tag[p] for p in pr]
        merged_pr_tag = copy.deepcopy(pr_tag)
        for i in range(len(pr_tag)):
            for j in range(i+1, len(pr_tag)):
                if pr_tag[i] + "_" + pr_tag[j] in oritag2idx:
                    merged_pr_tag.append(pr_tag[i] + "_" + pr_tag[j])
                if pr_tag[j] + "_" + pr_tag[i] in oritag2idx:
                    merged_pr_tag.append(pr_tag[j] + "_" + pr_tag[i])
                if pr_tag[i] + "_of_" + pr_tag[j] in oritag2idx:
                    merged_pr_tag.append(pr_tag[i] + "_of_" + pr_tag[j])
                if pr_tag[j] + "_of_" + pr_tag[i] in oritag2idx:
                    merged_pr_tag.append(pr_tag[j] + "_of_" + pr_tag[i])
                if pr_tag[i] + "_in_" + pr_tag[j] in oritag2idx:
                    merged_pr_tag.append(pr_tag[i] + "_in_" + pr_tag[j])
                if pr_tag[j] + "_in_" + pr_tag[i] in oritag2idx:
                    merged_pr_tag.append(pr_tag[j] + "_in_" + pr_tag[i])
                for k, words in enumerate(tri_words):
                    if all(map(lambda w:w in pr_tag, words)):
                        merged_pr_tag.append(tri_word_tags[k])
        merged_pr_tag = list(set(merged_pr_tag).intersection(set(ori_tag_list)))
        merged_pr_tags.append(merged_pr_tag)
    label = [[idx2tag[l] for l in la] for la in y_true]
    print("predicted:", merged_pr_tags[:10])
    print("label:", label[:10])
    return get_openentity_metrics(label, merged_pr_tags, string=True)

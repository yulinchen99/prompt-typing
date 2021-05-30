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
    print("Strict Accuracy: %s" % acc)
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
    print("Loose Macro:")
    print("Precision %s Recall %s F1 %s" % (p, r, f))
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
    p = cnt_correct/cnt_pred
    r = cnt_correct/cnt_label
    f = f1(p, r)
    print("Loose Micro:")
    print("Precision %s Recall %s F1 %s" % (p, r, f))
    return p, r, f

def get_metrics(label, pred, idx2tag, isfewnerd=False):
    label = [label_path(idx2tag[l], isfewnerd) for l in label]
    pred = [label_path(idx2tag[p], isfewnerd) for p in pred]
    acc = strict(label, pred)
    macro = loose_macro(label, pred)
    micro = loose_micro(label, pred)
    return acc, {'p':micro[0], 'r':micro[1], 'f':micro[2]}, {'p':macro[0], 'r':macro[1], 'f':macro[2]}



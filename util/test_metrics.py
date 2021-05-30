import pytest 
from metrics import strict, loose_micro, loose_macro, get_metrics, label_path, f1

@pytest.fixture()
def isfewnerd():
    return True


def idx2tag(isfewnerd):
    if not isfewnerd:
        d = {0:'/coarse1/fine1', 1:'/coarse1', 2:'/coarse2/fine1', 3:'/coarse3/fine3'}
    else:
        d = {0:'coarse1-fine/1', 1:'coarse1', 2:'coarse2-fine/1', 3:'coarse3-fine/3'}
    return d

@pytest.fixture()
def pred():
    return [0,1,2,3]

@pytest.fixture()
def true():
    return [0,0,2,3]

@pytest.fixture()
def labels(isfewnerd):
    labels = idx2tag(isfewnerd).values()
    return labels

def test_label_path(labels, isfewnerd):
    for label in labels:
        print(label)
        print(label_path(label))
        if isfewnerd:
            assert set(label_path(label, isfewnerd=isfewnerd)) == set([label, label.split('-')[0]])
        else:
            assert set(label_path(label)) == set([label, '/'+label.split('/')[1]])

def test_strict_acc(true, pred, isfewnerd):
    true = [label_path(idx2tag(isfewnerd)[t], isfewnerd) for t in true]
    pred = [label_path(idx2tag(isfewnerd)[p], isfewnerd) for p in pred]
    assert strict(true, pred) == 0.75

def test_loose_micro(true, pred, isfewnerd):
    true = [label_path(idx2tag(isfewnerd)[t], isfewnerd) for t in true]
    pred = [label_path(idx2tag(isfewnerd)[p], isfewnerd) for p in pred]
    assert loose_micro(true, pred) == (1, 0.875, f1(0.875, 1))

def test_loose_macro(true, pred, isfewnerd):
    true = [label_path(idx2tag(isfewnerd)[t], isfewnerd) for t in true]
    pred = [label_path(idx2tag(isfewnerd)[p], isfewnerd) for p in pred]
    assert loose_macro(true, pred) == (1, 0.875, f1(0.875, 1))
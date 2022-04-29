import numpy as np

from make_semi_supervised_dataset import osss_subset, ss_subset, class_filtered_dataset


def test_osss_subset():
    N = 100000
    M = 10
    sM = 5
    ratio = 0.2
    labels = np.random.choice(list(range(M)), size=N)
    res = osss_subset(labels, ratio=ratio, subset_class=sM)
    assert len(np.unique(res)) == sM + 1
    for i in range(sM):
        assert np.abs((res == i).sum() - (N/M)*ratio) < (N/M)*ratio/M
    assert np.abs((res == -1).sum() - N*((2-ratio)*(sM/M))) < N*((2-ratio)*(sM/M))/M


def test_ss_subset():
    N = 100000
    M = 10
    ratio = 0.2
    labels = np.random.choice(list(range(M)), size=N)
    res = ss_subset(labels, ratio=ratio)
    assert len(np.unique(res)) == M + 1
    for i in range(M):
        assert np.abs((res == i).sum() - (N/M)*ratio) < (N/M)*ratio/M
    assert np.abs((res == -1).sum() - N*(1-ratio)) < (N*(1-ratio))/M


def test_class_filtered_dataset():
    N = 100000
    M = 10
    sM = 5
    labels = np.random.choice(list(range(M)), size=N)    
    imgs = np.stack([labels, labels, labels], -1)
    assert labels.shape == (N,)
    assert imgs.shape == (N, 3)
    filtered_imgs, filtered_labels = class_filtered_dataset(imgs, labels, list(range(sM)))
    assert np.sum(filtered_labels >= sM) == 0
    assert np.sum(filtered_imgs >= sM) == 0    

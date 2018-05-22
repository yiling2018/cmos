import numpy as np
import scipy.io
import os

path = 'E:\\Datasets\\wiki\\my\\'
text_size = 8
alog_size = 93
log_size = 151
pred_size = 275
m5e_size = 1117
m6h_size = 1041
m7d_size = 640

# for training
class TrainDataSet(object):
    def __init__(self, img, txt, tri):
        """Construct a DataSet.
        """
        self._img1 = img['img_tr1']
        self._img2 = img['img_tr2']
        self._img3 = img['img_tr3']
        self._img4 = img['img_tr4']
        self._img5 = img['img_tr5']
        self._img6 = img['img_tr6']
        self._txt = txt['txt_tr']
        self._pos = 0
        self._num_sample = tri.shape[0]
        self._tri = tri

    def next_batch(self, batch_size):
        if self._pos + batch_size >= self._num_sample:
            idx1 = np.arange(self._pos, self._num_sample)
            self._pos = self._pos + batch_size - self._num_sample
            idx2 = np.arange(0, self._pos)
            idx = np.hstack((idx1, idx2))
        else:
            idx = np.arange(self._pos, self._pos + batch_size)
            self._pos += batch_size

        img1_1 = self._img1[self._tri[idx, 0], :]
        img1_2 = self._img1[self._tri[idx, 1], :]
        img1_3 = self._img1[self._tri[idx, 2], :]
        img2_1 = self._img2[self._tri[idx, 0], :]
        img2_2 = self._img2[self._tri[idx, 1], :]
        img2_3 = self._img2[self._tri[idx, 2], :]
        img3_1 = self._img3[self._tri[idx, 0], :]
        img3_2 = self._img3[self._tri[idx, 1], :]
        img3_3 = self._img3[self._tri[idx, 2], :]
        img4_1 = self._img4[self._tri[idx, 0], :]
        img4_2 = self._img4[self._tri[idx, 1], :]
        img4_3 = self._img4[self._tri[idx, 2], :]
        img5_1 = self._img5[self._tri[idx, 0], :]
        img5_2 = self._img5[self._tri[idx, 1], :]
        img5_3 = self._img5[self._tri[idx, 2], :]
        img6_1 = self._img6[self._tri[idx, 0], :]
        img6_2 = self._img6[self._tri[idx, 1], :]
        img6_3 = self._img6[self._tri[idx, 2], :]
        txt_1 = self._txt[self._tri[idx, 0], :]
        txt_2 = self._txt[self._tri[idx, 1], :]
        txt_3 = self._txt[self._tri[idx, 2], :]

        img1 = np.vstack((img1_1, img1_2, img1_3))
        img2 = np.vstack((img2_1, img2_2, img2_3))
        img3 = np.vstack((img3_1, img3_2, img3_3))
        img4 = np.vstack((img4_1, img4_2, img4_3))
        img5 = np.vstack((img5_1, img5_2, img5_3))
        img6 = np.vstack((img6_1, img6_2, img6_3))
        txt = np.vstack((txt_1, txt_2, txt_3))

        return img1, img2, img3, img4, img5, img6, txt


def get_train_data():
    img_fea = scipy.io.loadmat(os.path.join(path, 'icptv4_multi_norzm_py.mat'))
    txt_fea = scipy.io.loadmat(os.path.join(path, 'txt_norzm_py.mat'))
    triplets = np.load('sample_tri_wiki.npy')
    return TrainDataSet(img_fea, txt_fea, triplets)


# for testing
class TestDataSet(object):

    def __init__(self, img, txt, label):
        """construct a dataset.
        """
        self._img1 = img['img_te1']
        self._img2 = img['img_te2']
        self._img3 = img['img_te3']
        self._img4 = img['img_te4']
        self._img5 = img['img_te5']
        self._img6 = img['img_te6']
        self._txt = txt['txt_te']
        self._label = label['label_te']
        self._idx_in_epoch = 0
        self._num_examples = label['label_te'].shape[0]

    def next_batch(self):
        return self._img1, self._img2, self._img3, self._img4, self._img5, self._img6, self._txt, self._label


def get_test_data():
    img_fea = scipy.io.loadmat(os.path.join(path, 'icptv4_multi_norzm_py.mat'))
    txt_fea = scipy.io.loadmat(os.path.join(path, 'txt_norzm_py.mat'))
    label = scipy.io.loadmat(os.path.join(path, 'data_norzm.mat'))
    return TestDataSet(img_fea, txt_fea, label)


# generate triplets for training
def gen_tri_uc():
    mat = scipy.io.loadmat(os.path.join(path, 'data_norzm.mat'))
    label = mat['label_tr']
    #
    num_sample = int(1e5)
    triplets = np.zeros((num_sample, 3), dtype=np.int32)
    for i in range(num_sample):
        c_idx = np.random.randint(0, label.shape[0])
        while 1:
            p_idx = np.random.randint(0, label.shape[0])
            if label[c_idx] == label[p_idx]:
                break
        while 1:
            n_idx = np.random.randint(0, label.shape[0])
            if not label[c_idx] == label[n_idx]:
                break
        triplets[i, 0] = c_idx
        triplets[i, 1] = p_idx
        triplets[i, 2] = n_idx
    np.save('sample_tri_wiki', triplets)


if __name__ == "__main__":
    gen_tri_uc()
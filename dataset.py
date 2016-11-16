import itertools as it
import numpy as np


def random_test_input():
    """
    this generates random test input, useful for debugging
    """
    sz = 224
    channels = 3
    init_val = tf.random_normal(
        (batch_size, sz, sz, channels),
        dtype=tf.float32,
        stddev=1
    )
    images = tf.Variable(init_val)
    labels = tf.Variable(tf.ones([batch_size], dtype=tf.int32))
    return images, labels


def get_cifar10(batch_size=16):
    print("loading cifar10 data ... ")

    from skdata.cifar10.dataset import CIFAR10
    cifar10 = CIFAR10()
    cifar10.fetch(True)

    trn_labels = []
    trn_pixels = []
    for i in range(1,6):
        data = cifar10.unpickle("data_batch_%d" % i)
        trn_pixels.append(data['data'])
        trn_labels.extend(data['labels'])

    trn_pixels = np.vstack(trn_pixels)
    trn_pixels = trn_pixels.reshape(-1, 3, 32, 32).astype(np.float32)

    tst_data = cifar10.unpickle("test_batch")
    tst_labels = tst_data["labels"]
    tst_pixels = tst_data["data"]
    tst_pixels = tst_pixels.reshape(-1, 3, 32, 32).astype(np.float32)

    print("trn.shape=%s tst.shape=%s" % (trn_pixels.shape, tst_pixels.shape))

    print("computing mean & stddev for cifar10 ...")
    mu = np.mean(trn_pixels, axis=(0,2,3))
    std = np.std(trn_pixels, axis=(0,2,3))
    print("cifar10 mu  = %s" % mu)
    print("cifar10 std = %s" % std)
    print("whitening cifar10 pixels ... ")

    trn_pixels[:, :, :, :] -= mu.reshape(1, 3, 1, 1)
    trn_pixels[:, :, :, :] /= std.reshape(1, 3, 1, 1)

    tst_pixels[:, :, :, :] -= mu.reshape(1, 3, 1, 1)
    tst_pixels[:, :, :, :] /= std.reshape(1, 3, 1, 1)

    # transpose to tensorflow's bhwc order assuming bchw order
    trn_pixels = trn_pixels.transpose(0, 2, 3, 1)
    tst_pixels = tst_pixels.transpose(0, 2, 3, 1)

    trn_set = batch_iterator(it.cycle(zip(trn_pixels, trn_labels)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    tst_set = (np.vstack(tst_pixels), np.array(tst_labels))

    return trn_set, tst_set

def batch_iterator(iterable, size, cycle=False, batch_fn=lambda x: x):
    """
    Iterate over a list or iterator in batches
    """
    batch = []

    # loop to begining upon reaching end of iterable, if cycle flag is set
    if cycle is True:
        iterable = it.cycle(iterable)

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)


if __name__ == '__main__':
    trn, tst = get_cifar10()

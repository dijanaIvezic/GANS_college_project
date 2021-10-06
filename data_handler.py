from tensorflow.keras import datasets
import cv2
import numpy as np

def LAB_to_BGR(im):
    L, a, b = cv2.split(im)
    L = (L+1) * 50
    a = a*127.0
    b = b*127.0

    im_lab = cv2.merge([L,a,b])
    im_bgr = cv2.cvtColor(im_lab, cv2.COLOR_Lab2BGR)
    return im_bgr

def calculate_accuracy(real, fake, tresh):
    accs = []
    for i in range(len(real)):
        real_bgr = LAB_to_BGR(real[i])
        fake_bgr = LAB_to_BGR(fake[i])

        diffs = np.abs(real_bgr-fake_bgr)
        diffs = (diffs < tresh).astype("int")
        diffs = np.sum(diffs)

        sh = list(real_bgr.shape)
        n = np.prod(sh)

        accs.append(diffs/n)

    return np.sum(accs)/len(accs)

def get_data():

    (train_images, _), (test_images, _) = datasets.cifar10.load_data()

    def proces_dataset(images):
        def convert(im):
            im_color = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im_color = im_color.astype("float32") / 255.0
            im_lab = cv2.cvtColor(im_color, cv2.COLOR_BGR2LAB)

            L, a, b = cv2.split(im_lab)
            L = L/50.0 - 1
            a = a/127.0
            b = b/127.0

            im_lab = cv2.merge([L,a,b])

            L = np.expand_dims(L, 2)

            return (im_lab, L)

        def apply(data):
            n = data.shape[0]
            im_labs = np.empty(shape=(n,32,32,3), dtype="float32")
            Ls = np.empty(shape=(n,32,32,1), dtype="float32")

            for i in range(n):
                im_lab, L = convert(data[i])
                im_labs[i] = im_lab
                Ls[i] = L
            
            return (im_labs, Ls)

        return apply(images)

    train_lab, train_L = proces_dataset(train_images)
    test_lab, test_L = proces_dataset(test_images)

    return (train_lab, train_L, test_lab, test_L)
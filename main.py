import GAN
import data_handler
import cv2
import numpy as np

train_lab, train_L, test_lab, test_L = data_handler.get_data()

gan = GAN.GAN()

#gan.train(train_lab, train_L, test_lab, test_L)

gan.load_generator()
gen = gan.generator

n = 40
N = 50
wanted_shape = (320,320)

generated = np.empty((0,wanted_shape[1],3), dtype="float32")
true = np.empty((0,wanted_shape[1],3), dtype="float32")
gray = np.empty((0,wanted_shape[1],3), dtype="float32")

for i in range(n,N):
    generated_image = gen(test_L[i:i+1]).numpy()[0]

    generated_image = data_handler.LAB_to_BGR(generated_image)
    true_image = data_handler.LAB_to_BGR(test_lab[i])

    gray_image = np.repeat((test_L[i]+1)/2, 3, axis=2)

    generated_image = cv2.resize(generated_image, wanted_shape)
    true_image = cv2.resize(true_image, wanted_shape)
    gray_image = cv2.resize(gray_image, wanted_shape)

    generated = np.append(generated, generated_image, axis=0)
    true = np.append(true, true_image, axis=0)
    gray = np.append(gray, gray_image, axis=0)

together = np.hstack((true, gray, generated))
together = (together*255).astype("uint8")

cv2.imwrite("together.png", together)

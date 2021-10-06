import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, optimizers, losses
import numpy as np
import pandas as pd
import data_handler

#GAN network based on Keras
#network is also based on original paper at: https://arxiv.org/abs/1803.05400

class GAN():
    def __init__(self) -> None:
        pass
    
    def create_generator(self):
        #kernel sizes are from original paper, taken from their github repository
        #github: https://github.com/ImagingLab/Colorizing-with-GANs
        kernels_gen_encoder = [
            (64, 1, 0),
            (128, 2, 0),
            (256, 2, 0),
            (512, 2, 0),
            (512, 2, 0)
        ]

        kernels_gen_decoder = [
            (512, 2, 0.5),
            (256, 2, 0.5),
            (128, 2, 0),
            (64, 2, 0)
        ]

        encoder_layers = []

        #input is a gray image 32 by 32 pixels
        #greyed CIFAR-10 image
        input_layer = layers.Input(shape=(32,32,1))
        layer = input_layer

        for i in range(len(kernels_gen_encoder)):
            info = kernels_gen_encoder[i]
            layer = layers.Conv2D(filters=info[0], kernel_size=4, strides=info[1], padding="same")(layer)
            layer = layers.BatchNormalization()(layer)
            layer = layers.LeakyReLU()(layer)

            encoder_layers.append(layer)

        for i in range(len(kernels_gen_decoder)):
            info = kernels_gen_decoder[i]
            layer = layers.Conv2DTranspose(filters=info[0], kernel_size=4, strides=info[1], padding="same")(layer)
            layer = layers.BatchNormalization()(layer)
            layer = layers.ReLU()(layer)

            j = info[2]
            if(j > 0):
                layer = layers.Dropout(j)(layer)

            #network is based on U-Net architecture, we concatenate layers to keep important information
            layer = layers.concatenate([encoder_layers[len(encoder_layers)-i-2], layer], axis=3)

        #output layer is an LAB image
        layer = layers.Conv2D(filters=3, kernel_size=1, strides=1, activation=activations.tanh)(layer)

        model = keras.Model(inputs=input_layer, outputs=layer)

        return model

    def create_discriminator(self):
        #kernel sizes are based on cited work above
        kernels_dis = [
            (64, 2, 0),
            (128, 2, 0),
            (256, 2, 0),
            (512, 1, 0)
        ]

        #network input is a gray image concatenated with colored image in LAB color space
        input_layer = layers.Input(shape=(32,32,4))
        layer = input_layer

        for i in range(len(kernels_dis)):
            info = kernels_dis[i]
            layer = layers.Conv2D(filters=info[0], kernel_size=4, strides=info[1], padding="same")(layer)
            if(i > 0):
                layer = layers.BatchNormalization()(layer)
            layer = layers.ReLU()(layer)

        #output is a number which states whether an image is real or fake
        layer = layers.Conv2D(filters=1, kernel_size=4, strides=1)(layer)

        model = keras.Model(inputs=input_layer, outputs=layer)

        return model

    #function to be called for every minibatch
    #minibatch size is 128 just like in original paper
    def train_step(self, color_images, gray_images):
        #at every minibatch we take half images to train discriminator and other half for generator
        #models are trained one after another
        n_pic = color_images.shape[0]
        n = n_pic//2
  
        with tf.GradientTape() as tape:
            #load a portion of gray images from minibatch
            gray = gray_images[:n]
            #generate LAB predictions based on gray images
            generated_images = self.generator(gray)
            #concatenate real images just like fake ones with respective gray image
            colored = tf.concat([gray, color_images[:n]], axis=3)
            generated = tf.concat([gray, generated_images], axis=3)
            #half of the images are real, and other half is fake
            images = tf.concat([colored, generated], axis=0)

            #label true and fake images
            #one sided label smoothing just like in paper
            labels = tf.concat([(tf.ones((n,))-0.1), tf.zeros(n,)], axis=0)
        
            #squeeze prediction to single value between 0 and 1
            predictions = tf.squeeze(self.discriminator(images))
            #evaluate loss function on discriminator
            d_loss = self.loss(labels, predictions)
            d_loss = tf.reduce_mean(d_loss)

            #calculate gradient
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            #propagate the gradient based on Adam optimiser
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            #training generator, similar as discriminator
            gray = gray_images[n:]
            generated_images = self.generator(gray)
            colored = color_images[n:]
            generated = tf.concat([gray, generated_images], axis=3)

            labels = tf.ones((n_pic-n,)) - 0.1

            predictions = tf.squeeze(self.discriminator(generated))
            g_loss = self.loss(labels, predictions)
            g_loss = tf.reduce_mean(g_loss)
            #gg_loss represents difference between real and generated fake image
            gg_loss = tf.reduce_mean(tf.abs(colored-generated_images)) * self.lambda_w
            g_loss = g_loss + gg_loss

            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return (d_loss.numpy(), g_loss.numpy())

    def train(self, color_images, gray_images, test_color, test_gray, batch_size=128, epochs=20):
        self.loss = losses.BinaryCrossentropy(from_logits=True)

        self.initial_learning_rate = 2e-4
        self.decay_steps = 5e5
        self.decay_rate = 0.1
        #initiating optimisers with Exponential Decay to help convergence
        #optimizer also clips gradient to prevent hopping
        self.lr_decay_g = optimizers.schedules.ExponentialDecay(initial_learning_rate=self.initial_learning_rate, decay_steps=self.decay_steps, decay_rate=self.decay_rate)
        self.g_optimizer = optimizers.Adam(learning_rate=self.lr_decay_g, beta_1=0.5, clipnorm=1.0)
        self.d_optimizer = optimizers.Adam(learning_rate=self.lr_decay_g, beta_1=0.5, clipnorm=1.0)

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        #lambda hyerparameter from the paper, weight next to difference between real and fake image
        self.lambda_w = 100.0

        n_all = color_images.shape[0]
        n = n_all//batch_size
        rest = n_all - n * batch_size

        #indices to chose random pictures from testing dataset
        indices = np.random.choice(a=test_gray.shape[0], size=5, replace=False)

        data_g_loss = []
        data_d_loss = []
        data_acc = []

        for e in range(epochs):
            d_losses = []
            g_losses = []
            for i in range(n):
                r = i*batch_size
                R = (i+1)*batch_size
                d_loss, g_loss = self.train_step(color_images[r:R], gray_images[r:R])
                d_losses.append(d_loss)
                g_losses.append(g_loss)
            d_loss, g_loss = self.train_step(color_images[-rest:], gray_images[-rest:])
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            d_loss = np.sum(d_losses)/(n+1)
            g_loss = np.sum(g_losses)/(n+1)
            
            fake = self.generator(test_gray[indices]).numpy()
            acc = data_handler.calculate_accuracy(test_color[indices], fake, 0.025)

            #calculate, save and show losses and accuracies for each epoch
            print("Epoch {}: d_loss={}, g_loss={}, acc={}".format(e+1, d_loss, g_loss, acc))

            data_d_loss.append(d_loss)
            data_g_loss.append(g_loss)
            data_acc.append(acc)

        data = {"d_loss":data_d_loss, "g_loss":data_g_loss, "acc":data_acc}

        df = pd.DataFrame(data)
        df.to_csv("data.csv")

        self.generator.save_weights("weights.h5")

    def load_generator(self, filename="weights.h5"):
        self.generator = self.create_generator()
        self.generator.load_weights(filename)


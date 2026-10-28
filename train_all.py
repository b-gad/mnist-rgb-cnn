"""
Reproduction complète du projet MNIST RGB :
  1. génération du dataset MNIST RGB (reproductible, graine fixée)
  2. CNN 10 classes
  3. CNN binaire pair/impair
  4. transfert d'apprentissage (features Dense(100) puis Dense(40))
Les poids sont sauvegardés dans le répertoire de travail.
"""
import os
import random
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WORKING_DIR = "2025-2026-S5-Projet-MNIST-RGB"
os.makedirs(WORKING_DIR, exist_ok=True)

EPOCHS, BATCH = int(os.environ.get("EPOCHS", 15)), 64   # EPOCHS=2 pour un test rapide

# ---------------------------------------------------------------- dataset
LIST_RGB = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
                     [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)


def make_rgb(X, rng):
    """Fond uniforme dans [0.2, 0.8]^3 (arrondi à 2 décimales), chiffre de l'une des 8 couleurs."""
    n = X.shape[0]
    fond = np.round(rng.uniform(0.2, 0.8, size=(n, 3)), 2).astype(np.float32)
    ind = rng.integers(0, 8, size=n)
    chiffre = LIST_RGB[ind]
    masque = (X > 0)[..., None]                       # (n, 28, 28, 1)
    XRGB = np.where(masque, chiffre[:, None, None, :], fond[:, None, None, :])
    return XRGB.astype(np.float32), fond, ind


(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
rng = np.random.default_rng(SEED)
XtrainRGB, fond_tr, ind_tr = make_rgb(Xtrain, rng)
XtestRGB, fond_te, ind_te = make_rgb(Xtest, rng)

# Paramètres de coloration (très légers) : permettent de régénérer le dataset exact
np.savez_compressed(os.path.join(WORKING_DIR, "params_coloration.npz"),
                    fond_train=fond_tr, ind_train=ind_tr,
                    fond_test=fond_te, ind_test=ind_te)

Ytrain_nn, Ytest_nn = to_categorical(Ytrain), to_categorical(Ytest)
Ytrain_bin, Ytest_bin = Ytrain % 2, Ytest % 2
Ytrain_bin_nn, Ytest_bin_nn = to_categorical(Ytrain_bin), to_categorical(Ytest_bin)


# ----------------------------------------------------------------- modèles
def build_cnn(binaire=False):
    m = Sequential()
    m.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 3)))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(32, (3, 3), activation='relu'))
    m.add(Flatten())
    m.add(Dense(100, activation='relu', name='couche_1'))
    m.add(Dense(40, activation='relu', name='couche_2'))
    m.add(Dense(10, activation='softmax'))
    if binaire:                                       # architecture identique au rapport
        m.add(Dense(2, activation='softmax'))
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def fit_and_save(m, Xtr, Ytr, Xte, Yte, name, epochs=EPOCHS):
    m.fit(Xtr, Ytr, epochs=epochs, batch_size=BATCH,
          validation_data=(Xte, Yte), verbose=2)
    m.save_weights(os.path.join(WORKING_DIR, f"{name}.weights.h5"))
    _, acc = m.evaluate(Xte, Yte, verbose=0)
    print(f"[{name}] accuracy test : {acc * 100:.2f} %\n")
    return m


# 1) CNN 10 classes
model = fit_and_save(build_cnn(), XtrainRGB, Ytrain_nn, XtestRGB, Ytest_nn,
                     "weights_cnn_10classes")

# 2) CNN binaire pair/impair
model_bin = fit_and_save(build_cnn(binaire=True), XtrainRGB, Ytrain_bin_nn,
                         XtestRGB, Ytest_bin_nn,
                         "weights_cnn_binaire")

# 3) Transfert : features de couche_1 (100) -> petit modèle
feat100 = Model(inputs=model.inputs, outputs=model.layers[-3].output)
Ftr100, Fte100 = feat100.predict(XtrainRGB), feat100.predict(XtestRGB)

petit = Sequential([Dense(50, activation='relu', input_shape=(100,)),
                    Dense(2, activation='softmax')])
petit.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fit_and_save(petit, Ftr100, Ytrain_bin_nn, Fte100, Ytest_bin_nn,
             "weights_petit_modele")

# 4) Question 7 : features de couche_2 (40) -> petit modèle v2
feat40 = Model(inputs=model.inputs, outputs=model.layers[-2].output)
Ftr40, Fte40 = feat40.predict(XtrainRGB), feat40.predict(XtestRGB)

petit_v2 = Sequential([Dense(20, activation='relu', input_shape=(40,)),
                       Dense(2, activation='softmax')])
petit_v2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fit_and_save(petit_v2, Ftr40, Ytrain_bin_nn, Fte40, Ytest_bin_nn,
             "weights_petit_modele_v2", epochs=min(10, EPOCHS))

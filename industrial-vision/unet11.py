from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np, os
import cv2

# Dataset
X = []; Y = []

image_path = r"C:\Users\rakul\OneDrive\Documents\Desktop\quizgen\dataset\images"
mask_path = r"C:\Users\rakul\OneDrive\Documents\Desktop\quizgen\dataset\masks"

for f in os.listdir(image_path):

    # Input image
    X.append(cv2.resize(cv2.imread(os.path.join(image_path, f)),
                        (128,128))/255.0)

    # Mask image
    Y.append(np.expand_dims(cv2.resize(
        cv2.imread(os.path.join(mask_path, f), 0),
        (128,128))/255.0, axis=-1))

# Convert list to array
X = np.array(X)
Y = np.array(Y)

# Input
i = Input((128,128,3))

# Encoder
c1 = Conv2D(32,(3,3),activation='relu',padding='same')(i)
p1 = MaxPooling2D((2,2))(c1)

# Bottleneck
c2 = Conv2D(64,(3,3),activation='relu',padding='same')(p1)

# Decoder
u1 = UpSampling2D((2,2))(c2)
m1 = concatenate([u1,c1])

# Output
o = Conv2D(1,(1,1),activation='sigmoid')(m1)

# Model
model = Model(i,o)

# Compile
model.compile(optimizer='adam',loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X,Y,epochs=5,batch_size=2)

# Save
model.save("unet.h5")
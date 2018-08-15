from generator import gen_id_card
import numpy as np
import random
import string

# characters = string.digits + string.ascii_uppercase
characters = string.digits
# print(characters)

width, height, n_class = 140, 20, len(characters)


def gen(batch_size=32, n_len=11):
    genObj = gen_id_card(width, height)
    x = np.zeros((batch_size, height, width, 1), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    for i in range(batch_size):
        image_data, label, vec = genObj.gen_image()
        x[i] = image_data
        y[i] = [int(_) for _ in label]
    return x, y


batch_x, batch_y = gen(200)

print(batch_x.shape)
print(batch_y.shape)

np.save("x_train.npy", batch_x)
np.save("y_train.npy", batch_y)

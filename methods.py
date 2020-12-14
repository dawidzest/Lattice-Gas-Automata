import numba
from numba import jit, njit
from PIL import Image, ImageTk
import numpy as np
import random


@jit(nopython=True, parallel=True)
def fill_matrix(img_matrix, state_matrix):
    for x in numba.prange(0, state_matrix.shape[0]-1):
        for y in numba.prange(0, state_matrix.shape[1]-1):
            state = state_matrix[x, y]
            if y == 0 or x == 0 or y == state_matrix.shape[1]-1 or x == state_matrix.shape[0]-1:
                state[0] = 1
                state_matrix[x, y] = state
                img_matrix[x, y] = [255, 255, 255]
            if y == 150 and (0 < x < (((state_matrix.shape[0]-1)/2)-25) or \
                             (((state_matrix.shape[0]-1)/2)+25 < x < state_matrix.shape[0]-1)):
                state[0] = 1
                state_matrix[x, y] = state[0]
                img_matrix[x, y] = [255, 255, 255]
            random_gen_1 = random.randint(0, 100)
            treshold = 92
            if 0 < y < 150 and 0 < x < (state_matrix.shape[0]-1) and random_gen_1 > treshold:
                random_gen_2 = random.randint(1, 4)
                if random_gen_2 == 1:
                    state[1] = 1
                elif random_gen_2 == 2:
                    state[2] = 1
                elif random_gen_2 == 3:
                    state[3] = 1
                elif random_gen_2 == 4:
                    state[4] = 1
                state_matrix[x, y] = state
                img_matrix[x, y] = [255, 0, 0]

    return img_matrix, state_matrix


def simulate(canvas, img_matrix, state_matrix, master, frame):
    new_state_matrix = np.zeros([state_matrix.shape[0], state_matrix.shape[1], 5], dtype=np.uint8)
    new_img_matrix = np.zeros([img_matrix.shape[0], img_matrix.shape[1], 3], dtype=np.uint8)
    new_state_matrix, new_img_matrix = state_sim(state_matrix, new_state_matrix, img_matrix, new_img_matrix)
    img = ImageTk.PhotoImage(image=Image.fromarray(new_img_matrix))
    canvas.image = img
    canvas.create_image(0, 0, anchor="nw", image=img)
    master.after(2000, lambda: simulate(canvas, new_img_matrix, new_state_matrix, master, frame))


@jit(nopython=True, parallel=True)
def state_sim(state_matrix, new_state_matrix, img_matrix, new_img_matrix):
    for x in numba.prange(0, state_matrix.shape[0]-1):
        for y in numba.prange(0, state_matrix.shape[1]-1):
            state = state_matrix[x, y]
            if state[0] == 1:
                new_state_matrix[x, y] = [1, 0, 0, 0, 0]
                new_img_matrix[x, y] = [255, 255, 255]
            else:
                new_state = state
                state_up = state_matrix[x - 1, y]
                state_right = state_matrix[x, y + 1]
                state_down = state_matrix[x + 1, y]
                state_left = state_matrix[x, y - 1]
                new_state_up = state_matrix[x - 1, y]
                new_state_right = state_matrix[x, y + 1]
                new_state_down = state_matrix[x + 1, y]
                new_state_left = state_matrix[x, y - 1]

                if state_up[3] == 1:
                    new_state[3] = 1
                    new_state_up[3] = 0
                    if new_state_up[0] == 0 or new_state_up[1] == 0 or new_state_up[2] == 0 or new_state_up[3] == 0 or new_state_up[4] == 0:
                        new_img_matrix[x - 1, y] = [0, 0, 0]
                if state_right[4] == 1:
                    new_state[4] = 1
                    new_state_right[4] = 0
                    if new_state_right[0] == 0 or new_state_right[1] == 0 or new_state_right[2] == 0 or new_state_right[3] == 0 or new_state_right[4] == 0:
                        new_img_matrix[x, y + 1] = [0, 0, 0]
                if state_down[1] == 1:
                    new_state[1] = 1
                    new_state_down[1] = 0
                    if new_state_down[0] == 0 or new_state_down[1] == 0 or new_state_down[2] == 0 or new_state_down[3] == 0 or new_state_down[4] == 0:
                        new_img_matrix[x + 1, y] = [0, 0, 0]
                if state_left[2] == 1:
                    new_state[2] = 1
                    new_state_left[2] = 0
                    if new_state_left[0] == 0 or new_state_left[1] == 0 or new_state_left[2] == 0 or new_state_left[3] == 0 or new_state_left[4] == 0:
                        new_img_matrix[x, y - 1] = [0, 0, 0]
                """if new_state[1] == 1 and new_state[3] == 1:
                    new_state[1] = 0
                    new_state[2] = 1
                    new_state[3] = 0
                    new_state[4] = 1
                elif new_state[2] == 1 and new_state[4] == 1:
                    new_state[1] = 1
                    new_state[2] = 0
                    new_state[3] = 1
                    new_state[4] = 0"""
                new_state_matrix[x, y] = new_state
                if new_state[1] == 1 or new_state[2] == 1 or new_state[3] == 1 or new_state[4] == 1:
                    new_img_matrix[x, y] = [255, 0, 0]

    return new_state_matrix, new_img_matrix

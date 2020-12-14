"""
Simulating of the Lattice Gas Automata
-----------------------------
=====================================================

This application is my project for Discrete Modeling.
Dawid Kosior, Computational Engineering

=====================================================
"""

from numba import jit
from numba import jitclass
from joblib import Parallel, delayed
import multiprocessing
import tkinter as tk
import cv2
import numpy as np
import random
import sys
import os


@jit(nopython=True)
def make_walls(matrix):
    for x in range (0, matrix.shape[0]):
        for y in range (0, matrix.shape[1]):
            if 4 < x < 596 and 4 < y < 596:
                if x == 5 or x == 595 or y == 5 or y == 595:
                    matrix[x, y] = 100
                if x == 150 and (y < 270 or y > 330):
                    matrix[x, y] = 100
                if 4 < x < 149 and 4 < y < 599:
                    random_gen = random.randint(0, 100)
                    if random_gen > 92:
                        random_gen_2 = random.randint(0, 3)
                        if random_gen_2 == 0:
                            matrix[x, y] = 0b0001
                        elif random_gen_2 == 1:
                            matrix[x, y] = 0b0010
                        elif random_gen_2 == 2:
                            matrix[x, y] = 0b0100
                        elif random_gen_2 == 3:
                            matrix[x, y] = 0b1000
    return matrix


class Window:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master, background="white")

        self.matrix = np.zeros([600, 600])

        self.canvas = tk.Canvas(self.frame, width=600, height=600, background="white")
        self.matrix = make_walls(self.matrix)
        self.make_canvas()

        self.canvas.pack(side="left", fill="y")
        self.frame.pack()

    def make_canvas(self):
        for x in range (0, 600):
            for y in range (0, 600):
                if self.matrix[x, y] == 100:
                    x1, y1 = (x - 1), (y - 1)
                    x2, y2 = (x + 1), (y + 1)
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
                if 0 < self.matrix[x, y] <= 4:
                    x1, y1 = (x - 1), (y - 1)
                    x2, y2 = (x + 1), (y + 1)
                    self.canvas.create_oval(x1, y1, x2, y2, fill="red", outline="red")

    """def sim(self):
        for x in range (5, 595):
            for y in range (5, 595):
                if self.matrix[x-1, y] == 0b1000 or self.matrix[x-1, y] == 0b1100 or self.matrix[x-1, y] == 0b1000 \
                or self.matrix[x - 1, y] == 0b1000"""
#def main():
root = tk.Tk()
root.withdraw()

top = tk.Toplevel(root)
top.protocol("WM_DELETE_WINDOW", root.destroy)

app = Window(top)
top.title("LGA")
top.mainloop()

#if __name__ == '__main__':
#    main()

"""
Simulating of the Lattice Gas Automata
-----------------------------
=====================================================

This application is my project for Discrete Modeling.
Dawid Kosior, Computational Engineering

=====================================================
"""


from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import methods


class Window:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master, background="white")

        self.img_matrix = np.zeros([600, 600, 3], dtype=np.uint8)
        self.state_matrix = np.zeros([600, 600, 5], dtype=np.uint8)
        self.img_matrix, self.state_matrix = methods.fill_matrix(self.img_matrix, self.state_matrix)

        self.img = ImageTk.PhotoImage(image=Image.fromarray(self.img_matrix))

        self.canvas = tk.Canvas(self.frame, width=600, height=600)
        self.canvas.pack(side="left", fill="y")
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)
        self.frame.pack()

        self.master.after(50, lambda: methods.simulate(self.canvas, self.img_matrix, self.state_matrix, self.master, self.frame))



def main():
    root = tk.Tk()
    root.withdraw()

    top = tk.Toplevel(root)
    top.protocol("WM_DELETE_WINDOW", root.destroy)

    app = Window(top)
    top.title("LGA")
    top.mainloop()


if __name__ == '__main__':
    main()

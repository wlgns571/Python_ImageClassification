from tkinter import *
from PIL import Image,ImageDraw
import keras
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

def fn_img(imgPath):

    img = Image.open(imgPath).convert("L")
    width, height = img.size
    print(width, height)
    img_resize = img.resize((28, 28))
    plt.imshow(img_resize)
    plt.show()

    return np.array(img_resize).reshape(1, 28, 28, 1).astype('float32') / 255


class ImageGenerator:
    def __init__(self, parent, posx, posy, *kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = 'up'
        self.xold = None
        self.yold = None
        self.drawing_area = Canvas(self.parent, width=self.sizex, height=self.sizey + 10)
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind("<Motion>", self.fn_motion)
        self.drawing_area.bind("<ButtonPress-1>", self.fn_b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.fn_b1up)
        self.drawing_area.place(x=self.sizex/7, y=self.sizey/7)
        self.button = Button(self.parent, text="Done!", width=8, command=self.fn_save)
        self.button.place(x=30, y=self.sizey +50)
        self.button1 = Button(self.parent, text="Clear!", width=8, command=self.fn_clear)
        self.button1.place(x=135, y=self.sizey + 50)

        self.text = Text(self.parent, width=14, height=1)
        self.text.insert(INSERT, "예상 숫자: ")
        self.text.pack()
        self.text.place(x=90, y=4)

        self.image = Image.new("RGB",(200,200),(255,255,255))
        self.draw = ImageDraw.Draw(self.image)

        self.model = keras.models.load_model("./cnn_model/15-0.0251.hdf5")

    def fn_check(self):
        if os.path.exists(self.filename):
            predictions = self.model.predict(fn_img(self.filename))
            print(predictions)
            print(np.argmax(predictions, 1))
            return np.argmax(predictions, 1).tolist()

    def fn_save(self):
        self.filename = "./img.jpg"
        self.image.save(self.filename)
        num = self.fn_check()
        msg = "예상숫자 :" + str(num[0])
        print(msg)
        self.text.delete('1.0', END)
        self.text.insert(INSERT, msg)

    def fn_clear(self):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw((self.image))

    def fn_b1down(self, event):
        self.b1 = "down"

    def fn_b1up(self, event):
        self.b1 = "up"
        self.xold =None
        self.yold = None

    def fn_motion(self, event):
        if self.b1 =="down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth='false', width=5, fill='black')
                self.draw.line(((self.xold, self.yold), (event.x, event.y)), (0, 128, 0), width=5)
        self.xold =event.x
        self.yold =event.y

if __name__ == '__main__':
    root = Tk()
    root.wm_geometry("%dx%d+%d+%d"%(265, 300, 10, 10))
    root.config(bg='gray')
    ImageGenerator(root, 5, 5)
    root.mainloop()


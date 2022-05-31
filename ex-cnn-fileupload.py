from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter.messagebox as mbox
import keras
import numpy as np
model = keras.models.load_model('./cnn_model/10-0.0257.hdf5')
root = Tk()
root.title('number image convert')

def fn_img(imgPath):
    img = Image.open(imgPath).convert('L')
    width, height = img.size
    print(width, height)
    img_resize = img.resize((28, 28))
    return np.array(img_resize).reshape(1, 28, 28, 1).astype('float32') /255

def fn_num():
    prediction = model.predict(fn_img(filepath))
    num = np.argmax(prediction, 1).tolist()
    mbox.showinfo('convert',str(num[0]))

def fn_open():
    global my_image
    global root
    global filepath
    try:
        if 'normal' == root.state():
            root.destroy()
    except Exception as e:
        print(str(e))
    finally:
        root = Tk()
        open_btn = Button(root, text='파일열기', command=fn_open).pack()
        check_btn = Button(root, text='번호는?', command=fn_num).pack()
        root.filename=filedialog.askopenfilename(initialdir=''
            ,title='파일선택'
            ,filetypes=(('png files', '*.PNG'),('jpg files','*.jpg')
                        ,('all files','*.*')))
        Label(root, text=root.filename).pack()
        my_image = ImageTk.PhotoImage(Image.open(root.filename))
        Label(root, image=my_image).pack()
        filepath = root.filename
        print(filepath)
my_btn = Button(root, text='파일열기', command=fn_open).pack()
root.mainloop()
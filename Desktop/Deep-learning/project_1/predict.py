import cv2 as cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib as joblib
from train import *
from tkinter import *
# loading Python Imaging Library
from PIL import ImageTk, Image
# To get the dialog box to open when required
from tkinter import filedialog
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import tkinter as tk
from tkinter.font import Font


#   Giao diện
#   ----------------------------------------------------------------------------------------

#   Tạo cửa sổ upload
root = Tk()

#   Đặt tiêu đề chp hình tải lên
root.title("Chương trình tính toán đơn giản")

#   Thiết lập độ phân giải
root.geometry("555x470")

#   Cho phép thay đổi kích thước, giống như
root.resizable(width=True, height=True)

myFont = Font(weight="bold")
# myFont1 = Font(family="Calibri", size = 18)

member = Label(root, text="Nhóm 08: Khương, Khoa, Châu ")
# member = Label(root, text="Dành cho học sinh mẫu giáo =))")
member.grid(row=30, column=0)

#   Thiết lập ảnh mặc định
img = ImageTk.PhotoImage(Image.open("./img_for_gui/img_upload.png"))
panel = Label(root, image=img)
panel.grid(row=0, column=0)

img1 = ImageTk.PhotoImage(Image.open("./img_for_gui/result.png"))
panel = Label(root, image=img1)
panel.grid(row=1, column=0)


def open_img():
    global img_new
    img_new = filedialog.askopenfilename(title='pen')
    # Select the Imagename  from a folder
    # img_new = openfilename()
    # img1 = cv2.imread(img_new)
    # opens the image
    img = Image.open(img_new)
    # # resize the image and apply a high-quality down sampling filter
    img = img.resize((380, 190), Image.ANTIALIAS)
    # print(img)
    # # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # print(img)
    # create a label
    panel = Label(root, image=img)
    # # set the image as img
    panel.image = img
    panel.grid(row=0, column=0)


def open_img_predict():
    global ketqua
    global bieuthuc
    img = cv2.imread(img_new)
    img, ketqua, error, bieuthuc = main(img)
    label_kq = Label(root, text=bieuthuc)
    label_kq.grid(row=2, column=0)
    img = Image.fromarray(img)  # CHuyen image np array to PIL image
    # # opens the image
    # img = Image.open(x)
    # # # resize the image and apply a high-quality down sampling filter
    img = img.resize((380, 190), Image.ANTIALIAS)
    # # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # print(img)
    # create a label
    panel = Label(root, image=img)
    # # set the image as img
    panel.image = img
    panel.grid(row=1, column=0)
    return bieuthuc
# Create a button and place it into the window using grid layout


btn_upload = Button(root, text='Load image', command=open_img,
                    height=12, width=22, bg="#FFF8DC", fg='black')
btn_predict = Button(root, text='Calculate', command=open_img_predict,
                     height=12, width=22, bg="#FFF8DC", fg='black')
button_quit = Button(root, text="Exit", command=root.quit,
                     width=22, height=3, bg="#1a74e8", fg='red')


btn_upload.grid(row=0, column=1)
btn_predict.grid(row=1, column=1)
button_quit.grid(row=2, column=1)
root.mainloop()

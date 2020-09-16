import cv2 as cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tkinter import *
# loading Python Imaging Library 
from PIL import ImageTk, Image 
# To get the dialog box to open when required  
from tkinter import filedialog


def get_digit_data(path):#:, digit_list, label_list):
    digit_list = []
    label_list = []
    for number in range(12):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            digit_list.append(img)
            label_list.append(int(number))
    return digit_list,label_list

#lấy dữ liệu train
digit_path_train = "Desktop/data_svm_train/"
digit_list, label_list = get_digit_data(digit_path_train)
X_train = np.array(digit_list, dtype=np.float32)
y_train = np.array(label_list)

#lấy dữ liệu test
digit_path_test = "Desktop/data_svm_test/"
digit_list, label_list = get_digit_data(digit_path_test)
X_test = np.array(digit_list, dtype=np.float32)
y_test = np.array(label_list)

#   Rút trích đặc trưng cho tập train
#   Giai thích tham số:
#   pixels_per_cell là kích thước của 1 cell (đơn vị pixel)
#   pixels_per_cell = 5,5 trên ảnh 60,30 vậy là có 6 * 12 = 72 cell

#--------------------------------------------------------------------------

# #   Rút trích đặt trưng chp tập train
# X_train_feature = []
# for i in range(len(X_train)):
#     feature = hog(X_train[i],orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
#     X_train_feature.append(feature)
   
# X_train_feature = np.array(X_train_feature,dtype = np.float32)


# #   Rút trích đặc trưng cho tập test
# X_test_feature = []
# for i in range(len(X_test)):
#     feature = hog(X_test[i],orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
#     X_test_feature.append(feature)

# X_test_feature = np.array(X_test_feature,dtype=np.float32)

#--------------------------------------------------------------------------

#   Hàm rút trích đặc trưng
import cv2 as cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_digit_data(path):#:, digit_list, label_list):
    digit_list = []
    label_list = []
    for number in range(12):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            digit_list.append(img)
            label_list.append(int(number))
    return digit_list,label_list

#lấy dữ liệu train
digit_path_train = "./data_svm_train/"
digit_list, label_list = get_digit_data(digit_path_train)
X_train = np.array(digit_list, dtype=np.float32)
y_train = np.array(label_list)

#lấy dữ liệu test
digit_path_test = "./data_svm_test/"
digit_list, label_list = get_digit_data(digit_path_test)
X_test = np.array(digit_list, dtype=np.float32)
y_test = np.array(label_list)

#--------------------------------------------------------------------------

# #   Rút trích đặt trưng chp tập train
# X_train_feature = []
# for i in range(len(X_train)):
#     feature = hog(X_train[i],orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
#     X_train_feature.append(feature)
   
# X_train_feature = np.array(X_train_feature,dtype = np.float32)


# #   Rút trích đặc trưng cho tập test
# X_test_feature = []
# for i in range(len(X_test)):
#     feature = hog(X_test[i],orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
#     X_test_feature.append(feature)

# X_test_feature = np.array(X_test_feature,dtype=np.float32)

#--------------------------------------------------------------------------

#   Hàm rút trích đặc trưng
def feature(x):
    X_feature = []
    if len(x.shape) == 2:
        feature = hog(x,orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
        X_feature.append(feature)
    else:
        for i in range(len(x)):  
            feature = hog(x[i],orientations=9,pixels_per_cell=(5,5),cells_per_block=(1,1),block_norm="L2")
            X_feature.append(feature)      
    X_feature = np.array(X_feature)
    return (X_feature)

#   Hàm dự đoán nhãn
def predict(x):
    X_feature = feature(x)
    y_pred = model.predict(X_feature)
    return (y_pred)

#   Lấy ra các đặc trưng của tập X_train và X_test
X_train_feature = feature(X_train)
X_test_feature = feature(X_test)

#   Import model
model = LinearSVC(C=10)
#   Xây dựng mô hình
model.fit(X_train_feature,y_train)

#   Dự đoán nhãn cho tập X_test
y_predict = model.predict(X_test_feature)

#   In ra độ chính xác
print(accuracy_score(y_test,y_predict))


def get_digit_predicted(image):
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im,thre = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)
    #   Tìm các contours
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #   Tìm 3 contours có diện tích lớn nhất
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    area_sort_3 = area_sort[:3]
    contours_3 = []
    for i in area_sort_3:
        contours_3.append(contours[i])
    #   Tìm tọa độ boudingRect của các contours
    rects = [cv2.boundingRect(cnt) for cnt in contours_3]
    #   Sắp xếp contours từ trái sang phải dựa vào tọa độ X
    contours_LTR = []
    rects_sort = sorted(rects)
    for i in range(len(rects_sort)):
        for j in range(len(rects)):
            if rects_sort[i] == rects[j]:
                contours_LTR.append(contours_3[j])
    #   react_LTR là x,y,w,h tương ứng tọa độ, rộng và cao của các bounding box 
    rects_LTR = [cv2.boundingRect(cnt) for cnt in contours_LTR]
    contours = contours_LTR
    #   Tạo danh sách lưu số và dấu
    list_digit = []
    #   Duyệt qua các contours xác định nhãn
    for i in range(len(contours)):
        #   Hàm vẽ boundingbox hình chữ nhật bao quanh contours
        #   X,Y là tọa độ góc trên bên trái của hcn
        (x,y,w,h) = cv2.boundingRect(contours[i])
        h = h + 6
        w = w + 6
        x = x - 3
        y = y - 3
        #   Tại đây em tăng y giảm x để góc trên tách lên 1 ít, tránh khi vẽ bị cho hình
        #   Tăng h và w để có boundingbox rộng hơn
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #   Hàm vẽ bounding box 
        roi = thre[y:y+h,x:x+w]
        #   thre tạo ra 1 matran 
        roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (60, 30), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # rút trích đặc trưng cho contour
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(5, 5), cells_per_block=(1, 1),block_norm="L2")
        nbr = model.predict(np.array([roi_hog_fd], np.float32))
        list_digit.append(int(nbr[0]))
        kytu = ""
        if list_digit[i] == 10:
            kytu = '+'
        elif list_digit[i] == 11:
            kytu = '-'
        else:
            kytu = str(int(list_digit[i]))
        cv2.putText(image, kytu, (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    return list_digit

#   Lấy toán hạng và dấu 
def get_operation(image):
    list_digit = get_digit_predicted(image)
    toanhang = []
    pheptoan = []
    for digit in list_digit:
        if digit == 10:
            pheptoan.append('+')
        elif digit == 11:
            pheptoan.append('-')
        else:
            toanhang.append(digit)
    return toanhang,pheptoan

#   Hàm tính kết quả
def kq(image):
    toanhang,pheptoan = get_operation(image)
    error = []
    ketqua = 0
    #   Xác định xem có nhận được phép toán và toán hạng hay không
    if len(pheptoan) < 1 :
        error.append("Không nhận được dấu")
    elif len(pheptoan) > 1 :
        error.append("Nhận nhiều hơn 1 dấu")
    elif len(toanhang) > 2:
        error.append("Nhận nhiều hơn 2 toán tử")
    elif len(toanhang) < 2:
        error.append("Nhận ít hơn 2 toán tử")
    else:
        if pheptoan[0] == "+":
            ketqua = ketqua + ( toanhang[0] + toanhang[1])
        elif pheptoan[0] == "-":
            ketqua = ketqua + ( toanhang[0] - toanhang[1])
    return error,ketqua

#   Hàm lấy ra vị trí để vẽ
#   Hàm này lấy ra tọa độ, sau đó lấy toạn độ x + 200 để vẽ kết quả
def get_location(image):
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im,thre = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)
    #   Tìm các contours
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #   Tìm 3 contours có diện tích lớn nhất
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    area_sort[:3]
    contours_3 = []
    for i in area_sort:
        contours_3.append(contours[i])
    #   Tìm tọa độ boudingRect của các contours
    rects = [cv2.boundingRect(cnt) for cnt in contours_3]
    #   Sắp xếp contours từ trái sang phải dựa vào tọa độ X
    contours_LTR = []
    rects_sort = sorted(rects)
    for i in range(len(rects_sort)):
        for j in range(len(rects)):
            if rects_sort[i] == rects[j]:
                contours_LTR.append(contours_3[j])
    #    Lấy toạn độ x,y chiều rộng w và chiều cao h của các boundingbox
    rects_LTR = [cv2.boundingRect(cnt) for cnt in contours_LTR]
    return rects_LTR

#   Hàm hiển thị ảnh có kết quả
def show(image):
    error,ketqua = kq(image)
    rects_LTR = get_location(image)
    if len(error) != 0:
        x = rects_LTR[2][0] + 160
        y = rects_LTR[1][1]
        kytu = "error"
        cv2.putText(image, kytu, (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    else:
        x1 = rects_LTR[2][0]  + 300
        y1 = rects_LTR[1][1]
        x2 = x1 + 60
        y2 = y1
        cv2.putText(image, "=", (x1, y1),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        cv2.putText(image, str(int(ketqua)), (x2, y2),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    return image      

def main(image):
    img = show(image)    
    cv2.imshow("im",img)
    cv2.waitKey()


#   Lấy toán hạng và dấu 
def get_operation(image):
    list_digit = get_digit_predicted(image)
    toanhang = []
    pheptoan = []
    for digit in list_digit:
        if digit == 10:
            pheptoan.append('+')
        elif digit == 11:
            pheptoan.append('-')
        else:
            toanhang.append(digit)
    return toanhang,pheptoan

#   Hàm tính kết quả
def kq(image):
    toanhang,pheptoan = get_operation(image)
    error = []
    ketqua = 0
    #   Xác định xem có nhận được phép toán và toán hạng hay không
    if len(pheptoan) < 1 :
        error.append("Không nhận được dấu")
    elif len(pheptoan) > 1 :
        error.append("Nhận nhiều hơn 1 dấu")
    elif len(toanhang) > 2:
        error.append("Nhận nhiều hơn 2 toán tử")
    elif len(toanhang) < 2:
        error.append("Nhận ít hơn 2 toán tử")
    else:
        if pheptoan[0] == "+":
            ketqua = ketqua + ( toanhang[0] + toanhang[1])
        elif pheptoan[0] == "-":
            ketqua = ketqua + ( toanhang[0] - toanhang[1])
    return error,ketqua

#   Hàm lấy ra vị trí để vẽ
#   Hàm này lấy ra tọa độ, sau đó lấy toạn độ x + 200 để vẽ kết quả
def get_location(image):
    im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im,thre = cv2.threshold(im_gray,90,255,cv2.THRESH_BINARY_INV)
    #   Tìm các contours
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #   Tìm 3 contours có diện tích lớn nhất
    area_cnt = [cv2.contourArea(cnt) for cnt in contours]
    area_sort = np.argsort(area_cnt)[::-1]
    area_sort[:3]
    contours_3 = []
    for i in area_sort:
        contours_3.append(contours[i])
    #   Tìm tọa độ boudingRect của các contours
    rects = [cv2.boundingRect(cnt) for cnt in contours_3]
    #   Sắp xếp contours từ trái sang phải dựa vào tọa độ X
    contours_LTR = []
    rects_sort = sorted(rects)
    for i in range(len(rects_sort)):
        for j in range(len(rects)):
            if rects_sort[i] == rects[j]:
                contours_LTR.append(contours_3[j])
    #    Lấy toạn độ x,y chiều rộng w và chiều cao h của các boundingbox
    rects_LTR = [cv2.boundingRect(cnt) for cnt in contours_LTR]
    return rects_LTR

#   Hàm hiển thị ảnh có kết quả
def show(image):
    error,ketqua = kq(image)
    rects_LTR = get_location(image)
    if len(error) != 0:
        x = rects_LTR[2][0] + 160
        y = rects_LTR[1][1]
        kytu = "error"
        cv2.putText(image, kytu, (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    else:
        x1 = rects_LTR[2][0]  + 300
        y1 = rects_LTR[1][1]
        x2 = x1 + 60
        y2 = y1
        cv2.putText(image, "=", (x1, y1),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        cv2.putText(image, str(int(ketqua)), (x2, y2),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    return image,ketqua   

def main(image):
    img,ketqua = show(image)  
    # error,result = kq(img)
    # if len(error) == 0:
    #     print("Kết quả của phép toán: ",result) 
    # else:
    #     print("error: ",error)  
    # cv2.imshow("im",img)
    # cv2.waitKey()
    return (img,ketqua)





#   Giao diện
#   ----------------------------------------------------------------------------------------

#   Tạo cửa sổ upload
root = Tk()

#   Đặt tiêu đề chp hình tải lên
root.title("Image Loader")

#   Thiết lập độ phân giải
root.geometry("1024x512")

#   Cho phép thay đổi kích thước, giống như
root.resizable(width = True, height = True)

def open_img_predict(): 
    # Select the Imagename  from a folder  
    x = openfilename()
    img = cv2.imread(x)
    img,ketqua = main(img) 
    img = Image.fromarray(img) #CHuyen image np array to PIL image
    # # opens the image 
    # img = Image.open(x)
    # # # resize the image and apply a high-quality down sampling filter 
    img = img.resize((512, 256), Image.ANTIALIAS) 
    # # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img)
    # print(img)
    # create a label 
    panel = Label(root, image = img) 
    # # set the image as img  
    panel.image = img 
    panel.grid(row = 4)


def open_img(): 
    # Select the Imagename  from a folder  
    x = openfilename()
    img = cv2.imread(x)
    # opens the image
    img = Image.open(x)
    # # resize the image and apply a high-quality down sampling filter 
    img = img.resize((512, 256), Image.ANTIALIAS) 
    # print(img)
    # # PhotoImage class is used to add image to widgets, icons etc 
    img = ImageTk.PhotoImage(img)
    # print(img)
    # create a label 
    panel = Label(root, image = img) 
    # # set the image as img  
    panel.image = img 
    panel.grid(row = 4)
    


def openfilename(): 
    # open file dialog box to select image 
    # The dialogue box has a title "Open" 
    filename = filedialog.askopenfilename(title ='"pen')
    # print(filename)
    return filename

# Create a button and place it into the window using grid layout 
btn = Button(root, text ='open image', command = open_img).grid(row = 1, columnspan = 4) 
btn_2 = Button(root, text = 'predict', command = open_img_predict).grid(row = 3, columnspan = 4) 

root.mainloop() 


# img = cv2.imread("./data_svm_new/3cong8.jpg")
# # cv2.imshow("im",img)
# # cv2.waitKey()
# main(img)
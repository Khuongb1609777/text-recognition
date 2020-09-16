#   Nguyễn Nhật Khương b1609777

# --------------------------------------------------
#   Hàm rút trích đặc trưng
import cv2 as cv2
import os
import numpy as np
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class_name_digit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_name_op = ['+', '-']


def get_digit_data(path):  # :, digit_list, label_list):
    digit_list = []
    label_list = []
    digit_op_list = []
    label_op_list = []
    for number in range(10):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            digit_list.append(img)
            label_list.append(int(number))
    for number in range(10, 12):
        for img_org_path in glob.iglob(path + str(number) + '/*.jpg'):
            img = cv2.imread(img_org_path, 0)
            img = np.array(img)
            digit_op_list.append(img)
            label_op_list.append(int(number))
    return digit_list, digit_op_list, label_list, label_op_list


# lấy dữ liệu train
digit_path_train = "./data_svm_train/"
digit_list, digit_op_list, label_list, label_op_list = get_digit_data(
    digit_path_train)
X_digit_train = np.array(digit_list, dtype=np.float32)
X_op_train = np.array(digit_op_list, dtype=np.float32)
y_digit_train = np.array(label_list)
y_op_train = np.array(label_op_list)

# lấy dữ liệu test
digit_path_test = "./data_svm_test/"
digit_list, digit_op_list, label_list, label_op_list = get_digit_data(
    digit_path_test)
X_digit_test = np.array(digit_list, dtype=np.float32)
X_op_test = np.array(digit_op_list, dtype=np.float32)
y_digit_test = np.array(label_list)
y_op_test = np.array(label_op_list)

# -----------------------------------------------------------------------

#   Hàm rút trích đặc trưng


def feature(x):
    X_feature = []
    if len(x.shape) == 2:
        feature = hog(x, orientations=9, pixels_per_cell=(
            5, 5), cells_per_block=(2, 2), block_norm="L2")
        X_feature.append(feature)
    else:
        for i in range(len(x)):
            feature = hog(x[i], orientations=9, pixels_per_cell=(
                5, 5), cells_per_block=(2, 2), block_norm="L2")
            X_feature.append(feature)
    X_feature = np.array(X_feature)
    return (X_feature)


#   Hàm dự đoán nhãn
# def predict(x):
#     X_feature = feature(x)
#     y_pred = model1.predict(X_feature)
#     return (y_pred)

#   Lấy ra các đặc trưng của tập X_train và X_test
X_digit_train_feature = feature(X_digit_train)
X_op_train_feature = feature(X_op_train)
X_digit_test_feature = feature(X_digit_test)
X_op_test_feature = feature(X_op_test)

# #Xác định chỉ số C tốt nhất_-----------------------------
# parameter_candidates = [
#   {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear','rbf','poly']},]

# model1 = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
# model1.fit(X_digit_train_feature, y_digit_train)
# print('Best score model_1:', model1.best_score_)
# print('Best C model_1:',model1.best_estimator_.C)
# print(model1.best_params_)


# model2 = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
# model2.fit(X_op_train_feature, y_op_train)
# print('Best score model_2:', model2.best_score_)
# print('Best C model_2:',model2.best_estimator_.C)
# print(model2.best_params_)


model1 = SVC(kernel='linear', C=0.1)
model2 = SVC(kernel='linear', C=0.001)

model1.fit(X_digit_train_feature, y_digit_train)
model2.fit(X_op_train_feature, y_op_train)

y_digit_predict = model1.predict(X_digit_test_feature)
y_op_predict = model2.predict(X_op_test_feature)

y_test = np.append(y_digit_test, y_op_test, axis=0)
y_predict = np.append(y_digit_predict, y_op_predict, axis=0)


# titles_options = [("Ma trận nhầm lẫn trên tập số", None),
#                   ("Chuẩn hóa về 1", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(model1,X_digit_test_feature ,y_digit_test ,
#                                  display_labels=class_name_digit,
#                                  cmap= plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#     print(title)
#     print(disp.confusion_matrix)

titles_options = [("Ma trận nhầm lẫn trên tập dấu", None),
                  ("Chuẩn hóa về 1", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(model2, X_op_test_feature, y_op_test,
                                 display_labels=class_name_op,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)

plt.show()


#   In ra độ chính xác
print("Độ chính xác trên tập test ", 100*accuracy_score(y_test, y_predict))

#   Hàm rút trích đặc trưng cho ảnh (Hàm này dùng cho ảnh có phép tính)


def get_digit_predicted(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)
    # Hàm threshold có 4 tham số, tham số đầu là ảnh xám, 2 là ngưỡng, 3 là giá trị khi kích hoạt ngưỡng, 4 là toại nhị phân
    #   Thres_binary_inv giá trị lớn hơn ngưỡng gán 0, nhở hơn gán MAXVAL
    im, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)
    #   Tìm các contours
    contours, hierachy = cv2.findContours(
        thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if i == 1:
            (x, y, w, h) = cv2.boundingRect(contours[i])
            if(w >= 3*h):
                w1 = w + 4
                h1 = w1
                x1 = x - 2
                y1 = y - 2
            else:
                if(h >= w):
                    h1 = h
                    w1 = h1
                    x1 = x
                    y1 = y
                else:
                    w1 = w
                    h1 = w1
                    x1 = x
                    y1 = y
            #   Tại đây em tăng y giảm x để góc trên tách lên 1 ít, tránh khi vẽ bị cho hình
            #   Tăng h và w để có boundingbox rộng hơn
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #   Hàm vẽ bounding box x,y là tọa độ goc trên bên trái
            cut_img = thre[y1:y1+h1, x1:x1+w1]
            #   thre tạo ra 1 matran
            cut_img = np.pad(cut_img, (20, 20), 'constant',
                             constant_values=(0, 0))
            cut_img = cv2.resize(
                cut_img, (30, 30), interpolation=cv2.INTER_AREA)
            cut_img = cv2.dilate(cut_img, (3, 3))
            # rút trích đặc trưng cho contour
            cut_img_hog_fd = hog(cut_img, orientations=9, pixels_per_cell=(
                5, 5), cells_per_block=(2, 2), block_norm="L2")
            # print(cut_img_hog_fd.shape)
            # print(cut_img_hog_fd.shape)
            nbr = model2.predict(np.array([cut_img_hog_fd], np.float32))
            list_digit.append(int(nbr[0]))
            kytu = ""
            if list_digit[i] == 10:
                kytu = '+'
            elif list_digit[i] == 11:
                kytu = '-'
            else:
                kytu = str(int(list_digit[i]))
            cv2.putText(image, kytu, (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        else:
            (x, y, w, h) = cv2.boundingRect(contours[i])
            w1 = w+4
            h1 = h+4
            y1 = y-2
            x1 = x-2
            #   Tại đây em tăng y giảm x để góc trên tách lên 1 ít, tránh khi vẽ bị cho hình
            #   Tăng h và w để có boundingbox rộng hơn
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #   Hàm vẽ bounding box x,y là tọa độ goc trên bên trái
            cut_img = thre[y1:y1+h1, x1:x1+w1]
            #   thre tạo ra 1 matran
            cut_img = np.pad(cut_img, (20, 20), 'constant',
                             constant_values=(0, 0))
            cut_img = cv2.resize(
                cut_img, (30, 60), interpolation=cv2.INTER_AREA)
            cut_img = cv2.dilate(cut_img, (3, 3))
            # rút trích đặc trưng cho contour
            cut_img_hog_fd = hog(cut_img, orientations=9, pixels_per_cell=(
                5, 5), cells_per_block=(2, 2), block_norm="L2")
            # print(cut_img_hog_fd.shape)
            nbr = model1.predict(np.array([cut_img_hog_fd], np.float32))
            list_digit.append(int(nbr[0]))
            kytu = ""
            if list_digit[i] == 10:
                kytu = '+'
            elif list_digit[i] == 11:
                kytu = '-'
            else:
                kytu = str(int(list_digit[i]))
            cv2.putText(image, kytu, (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, 6, (0, 0, 255), 3)
    return list_digit, rects_LTR

#   Lấy toán hạng và dấu


def get_operation(image):
    list_digit, rects_LTR = get_digit_predicted(image)
    toanhang = []
    pheptoan = []
    for digit in list_digit:
        if digit == 10:
            pheptoan.append('+')
        elif digit == 11:
            pheptoan.append('-')
        else:
            toanhang.append(digit)
    return toanhang, pheptoan, rects_LTR

#   Hàm tính kết quả


def kq(image):
    toanhang, pheptoan, rects_LTR = get_operation(image)
    error = []
    ketqua = 0
    #   Xác định xem có nhận được phép toán và toán hạng hay không
    if len(pheptoan) < 1:
        error.append("Không nhận được dấu")
    elif len(pheptoan) > 1:
        error.append("Nhận nhiều hơn 1 dấu")
    elif len(toanhang) > 2:
        error.append("Nhận nhiều hơn 2 toán tử")
    elif len(toanhang) < 2:
        error.append("Nhận ít hơn 2 toán tử")
    else:
        if pheptoan[0] == "+":
            ketqua = ketqua + (toanhang[0] + toanhang[1])
        elif pheptoan[0] == "-":
            ketqua = ketqua + (toanhang[0] - toanhang[1])
    return error, ketqua, rects_LTR, toanhang, pheptoan

# #   Hàm lấy ra vị trí để vẽ
# #   Hàm này lấy ra tọa độ, sau đó lấy toạn độ x + 200 để vẽ kết quả
# def get_location(image):
#     im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     im_blur = cv2.GaussianBlur(im_gray,(5,5),0)
#     im,thre = cv2.threshold(im_blur,90,255,cv2.THRESH_BINARY_INV)
#     #   Tìm các contours
#     contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     #   Tìm 3 contours có diện tích lớn nhất
#     area_cnt = [cv2.contourArea(cnt) for cnt in contours]
#     area_sort = np.argsort(area_cnt)[::-1]
#     area_sort_3 = area_sort[:3]
#     contours_3 = []
#     for i in area_sort_3:
#         contours_3.append(contours[i])
#     #   Tìm tọa độ boudingRect của các contours
#     rects = [cv2.boundingRect(cnt) for cnt in contours_3]
#     #   Sắp xếp contours từ trái sang phải dựa vào tọa độ X
#     contours_LTR = []
#     rects_sort = sorted(rects)
#     for i in range(len(rects_sort)):
#         for j in range(len(rects)):
#             if rects_sort[i] == rects[j]:
#                 contours_LTR.append(contours_3[j])
#     #    Lấy toạn độ x,y chiều rộng w và chiều cao h của các boundingbox
#     rects_LTR = [cv2.boundingRect(cnt) for cnt in contours_LTR]
#     return rects_LTR

#   Hàm hiển thị ảnh có kết quả


def show(image):
    error, ketqua, rects_LTR, toanhang, pheptoan = kq(image)
    # rects_LTR = get_location(image)
    if len(error) != 0:
        x = rects_LTR[2][0] + 60
        y = rects_LTR[1][1]
        kytu = "error"
        cv2.putText(image, kytu, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, 6, (0, 0, 255), 3)
    else:
        x1 = rects_LTR[2][0] + rects_LTR[2][3] + 30
        w1 = rects_LTR[2][2]
        y1 = rects_LTR[1][1] + 60
        x2 = x1 + w1 + 30
        y2 = y1
        cv2.putText(image, "=", (x1, y1),
                    cv2.FONT_HERSHEY_DUPLEX, 6, (0, 0, 255), 3)
        cv2.putText(image, str(int(ketqua)), (x2, y2),
                    cv2.FONT_HERSHEY_DUPLEX, 6, (0, 0, 255), 3)
    return image, ketqua, error, toanhang, pheptoan


def main(image):
    img, ketqua, error, toanhang, pheptoan = show(image)
    # print(type(error))
    # cv2.imshow("im",img)
    # cv2.waitKey()
    bieuthuc = "Kết quả: " + str(int(toanhang[0])) + " " + str(
        pheptoan[0]) + " " + str(int(toanhang[1])) + " " + "=" + " " + str(ketqua)
    return img, ketqua, error, bieuthuc

# img = cv2.imread("./data_svm_new/6tru1_new.jpg")
# # cv2.imshow("im",img)
# # cv2.waitKey()
# main(img)

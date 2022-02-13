
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
####################################################

####################################################

# start of GUI
root = tk.Tk()
# general characteristics of the GUI
APP_WIDTH = 920 #minimal width of the GUI
APP_HEIGHT = 534 #minimal height of the gui
root.title("Face Recognition")
root.minsize(APP_WIDTH,APP_HEIGHT)
root["bg"]="#B1D0E0"
####################################

####################################

#functions

#Load The Data
imgs = np.load('data/olivetti_faces.npy')
label = np.load('data/olivetti_faces_target.npy')


def showdata():
    fig1 = plt.figure(figsize=(20, 10))
    columns = 10
    rows = 4
    for i in range(1, columns * rows + 1):
        img = imgs[10 * (i - 1), :, :]
        fig1.add_subplot(rows, columns, i)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.title("person {}".format(i), fontsize=16)
        plt.axis('off')

    plt.suptitle("The DataSet", fontsize=22)
    plt.show()

#Splite The Data
labs=label.reshape(-1,1)
imag_train, imag_test, label_train ,label_test = train_test_split(imgs,labs,test_size=0.2,random_state=40)
imag_train = imag_train.reshape(-1,64,64,1)   # adding the bais
imag_test = imag_test.reshape(-1,64,64,1)

label_train = to_categorical(label_train ,num_classes = 40)   # convert string to interger for each different class
label_test = to_categorical(label_test , num_classes= 40)   #classfication

#The Model
model= Sequential()
# Convolution
model.add(Conv2D(32,(3,3), activation='relu',input_shape =(64,64,1)))
# Pooling
model.add(MaxPooling2D(pool_size = (2,2)))
# Convolution
model.add(Conv2D(64,(3,3), activation='relu',input_shape =(64,64,1)))
# Pooling
model.add(MaxPooling2D(pool_size = (2,2)))
# flattening
model.add(Flatten())
# Full connection
model.add(Dense(256 , activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500 , activation='relu'))
model.add(Dropout(0.6))



model.add(Dense(40 , activation='sigmoid')) #the output layer with 40 different faces

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Generate the photos
datagen = ImageDataGenerator(
         rotation_range=40,
         width_shift_range=0.2,
         height_shift_range=0.2,
         rescale=1./255,
         shear_range=0.2,
         zoom_range=0.2,
         horizontal_flip=True,
         fill_mode='nearest')

datagen.fit(imag_train)

def train():
    model.fit(imag_train,label_train,batch_size=64,epochs=35)
    messagebox.showinfo('Result', 'Training dataset completed!!!')



def showPhoto():
    global imag_test,imag_train
    m = model.evaluate(imag_train, label_train) ###########################################################
    m = round((m[1] * 100), 2)  # Accurcy
    string_id = Number_entry.get()
    int_id = int(string_id)
    pred = model.predict(imag_test)
    num = np.argmax(pred[int_id])
    num = num + 1
    imag_test = imag_test.reshape(-1, 64, 64)
    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(imag_test[int_id, :], cmap=plt.get_cmap('gray'))
    plt.title("person: " + str(num) + " and The Accurcy is : " + str(m), fontsize=16)
    plt.suptitle("The Result", fontsize=22)
    plt.axis('off')
    canvas = FigureCanvasTkAgg(fig2,master=root)
    canvas.draw()
    canvas.get_tk_widget().place(relx=0.03, rely=0.052,anchor="nw")

####################################

####################################

# GUI elements

recognition_button = tk.Button(root, text = "Recognize",command = showPhoto,
                                bg = "#6998AB", fg = "white", activebackground = '#1A374D')
recognition_button.place(relx=0.97,rely=0.390,relheight=0.05,relwidth=0.2,anchor="ne")

recognition_button.focus()

first_seperator = ttk.Separator(root, orient="horizontal")
first_seperator.place(relx=0.97, rely=0.365,relwidth = 0.2, anchor = "ne")


showdata_button = tk.Button(root, text = "Show DataSet",command = showdata,
                                bg = "#6998AB", fg = "white", activebackground = '#1A374D')
showdata_button.place(relx=0.97,rely=0.480,relheight=0.05,relwidth=0.2,anchor="ne")
showdata_button.focus()

fourth_seperator = ttk.Separator(root, orient="horizontal")
fourth_seperator.place(relx=0.97, rely=0.460,relwidth = 0.2, anchor = "ne")

train_button = tk.Button(root, text = "Train",command = train,
                                bg = "#6998AB", fg = "white", activebackground = '#1A374D')
train_button.place(relx=0.97,rely=0.570,relheight=0.05,relwidth=0.2,anchor="ne")

train_button.focus()

fifth_seperator = ttk.Separator(root, orient="horizontal")
fifth_seperator.place(relx=0.97, rely=0.550,relwidth = 0.2, anchor = "ne")

second_seperator = ttk.Separator(root, orient="horizontal")
second_seperator.place(relx=0.97, rely=0.055,relwidth = 0.2, anchor = "ne")

MESSAGE = tk.StringVar()
MESSAGE.set("To get recognized,\nenter the Number!")
message_label=tk.Label(root,textvariable=MESSAGE, wraplength = "5c", bg="#6998AB", fg="white")
message_label.place(relx=0.97,rely=0.080,relwidth=0.2,relheight=0.16,anchor="ne")
message_label.config(font=(None, 11))

third_seperator = ttk.Separator(root, orient="horizontal")
third_seperator.place(relx=0.97, rely=0.265,relwidth = 0.2, anchor = "ne")

Number = tk.IntVar()
Number_entry = ttk.Entry(root, textvariable=Number)
Number_entry.place(relx=0.97, rely=0.290,relheight=0.05,relwidth = 0.2, anchor = "ne")
####################################

####################################
#create the GUI.
root.mainloop()
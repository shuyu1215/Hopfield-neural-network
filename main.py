#!/usr/bin/env python
# coding: utf-8
import tkinter as tk
from os import listdir,system
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

mypath = './dataset/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file_name = ''

def load_trainData(file_name):
    train_pic = []
    train_picture = []
    with open(file_name,'r') as f :
        for line in f.readlines():
            line = line.replace(" ","0").strip("\n")
            #print('line:',line)
            if len(line) == 0:
                for i in range(len(train_pic)):
                    if train_pic[i] == '0':
                        train_pic[i] = -1
                    else:
                        train_pic[i] = 1
                #print('train_pic:',train_pic)
                train_picture.append(np.array(train_pic))
                train_pic.clear()
            else:
                train_pic += line
        
        train_col = len(line)
        
        for i in range(len(train_pic)):
            if train_pic[i] == '0':
                 train_pic[i] = -1
            else:
                 train_pic[i] = 1
        #print('train_pic:',train_pic)
        train_picture.append(np.array(train_pic))
        #print('train_picture:',train_picture)
        
        train_picture_num = len(train_picture)
        train_dim = train_picture[0].shape[0]
        train_row = int(train_dim/train_col)
    return train_dim, train_picture, train_col

def load_testData(testFile_name,Dim):
    test_pic = []
    test_picture = []
    with open(testFile_name,'r') as f :
        for line in f.readlines():
            if len(test_pic) == Dim:
                for i in range(len(test_pic)):
                    if test_pic[i] == '0':
                        test_pic[i] = -1
                    else:
                        test_pic[i] = 1
                test_picture.append(np.array(test_pic))
                test_pic.clear()
            else:
                line = line.replace(" ","0").strip("\n")
                test_pic += line
        
        test_col = len(line)
        
        for i in range(len(test_pic)):
            if test_pic[i] == '0':
                 test_pic[i] = -1
            else:
                 test_pic[i] = 1
        test_picture.append(np.array(test_pic))
        test_picture_num = len(test_picture)
        return test_picture
        
def caculateWeight(picture,Dim):
    zero_M = np.zeros((Dim,Dim))
    identity_M = np.eye(Dim,dtype = int)
    basic_weight = np.zeros((Dim,Dim))
    for A in picture:
        A_T = A.reshape(Dim,1)
        zero_M += A_T * A
    weight = (1/Dim) * (zero_M) - (len(picture)/Dim) * (identity_M)
    return weight

def caculateThreshold(weight):
    threshold = []
    for w in weight:
        num = 0
        for a in w:
            num += a
        threshold.append(round(num,3))
    return threshold

def association(weight,threshold,test_pic,Dim):
    test_result = []
    input_M = test_pic
    check = True
    iteration = 0
    count = 0
    while(check):
        count += 1
        test = input_M.reshape(Dim,1)
        temp_M = weight.dot(test)
        for i in range(0,Dim):
            if temp_M[i] > threshold[i]:
                test_result.append(1)
            elif temp_M[i] == threshold[i]:
                test_result.append(int(test[i]))
            else:
                test_result.append(-1)
        check_same = (test_result == input_M).all()
        input_M = []
        last_M = []
        last_M = test_result
        input_M = np.array(test_result)
        test_result = []
        if iteration > 0:
            if check_same == True:
                check = False
        iteration += 1
    return last_M

class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.windows = master
        self.grid()
        self.mypath = './dataset/'
        self.files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.create_windows()

    def get_list(self,event):
        global file_name
        self.index = self.listbox.curselection()[0]
        self.selected = self.listbox.get(self.index)
        file_name = 'dataset/'+self.selected

    def create_windows(self):
        self.windows.title("Hopfield_HW3")

        self.listbox = tk.Listbox(windows, width=20, height=6)
        self.listbox.grid(row=0, column=0,columnspan=2,stick=tk.W+tk.E)

        self.yscroll = tk.Scrollbar(command=self.listbox.yview, orient=tk.VERTICAL)
        self.yscroll.grid(row=0, column=2, sticky=tk.W+tk.E)
        self.listbox.configure(yscrollcommand=self.yscroll.set)

        for item in self.files:
            self.listbox.insert(tk.END, item)

        self.listbox.bind('<ButtonRelease-1>', self.get_list)
        
        self.iteration = tk.Label(windows, text="Iteration:").grid(row=12,column=0, sticky=tk.W+tk.E)
        
        self.quit = tk.Button(windows, text='Quit', command=windows.quit).grid(row=1, column=0, sticky=tk.W+tk.E)
        self.show = tk.Button(windows, text='Show', command=self.show_entry_fields).grid(row=1, column=1, sticky=tk.W+tk.E)

        self.result_figure = Figure(figsize=(5,4), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, self.windows)
        self.result_canvas.draw()
        self.result_canvas.get_tk_widget().grid(row=6, column=0, columnspan=3, sticky=tk.W+tk.E)
        
    def show_entry_fields(self):
        str_test, str_output = run(file_name)
        self.text1 = tk.Label(windows, text="input:").grid(row=2,column=0, sticky=tk.W)
        self.text1 = tk.Label(windows, text="output:").grid(row=2,column=1, sticky=tk.W)
        for i in range(0,len(str_test)):
            self.test2 = tk.Label(windows, text=str_test[i]).grid(row=i+3,column=0, sticky=tk.N+tk.S)
            self.text2 = tk.Label(windows, text=str_output[i]).grid(row=i+3,column=1, sticky=tk.N+tk.S)


def run(file_name):
    Dim,picture,col = load_trainData(file_name)
    if file_name == 'dataset/Basic_Training.txt':
        testFile_name = 'dataset/Basic_Testing.txt'
    elif file_name == 'dataset/Bonus_Training.txt':
        testFile_name = 'dataset/Bonus_Testing.txt'
    testPicture = load_testData(testFile_name,Dim)
    test_picture = []
    str_test = []
    str_output = []
    test_picture.append(np.array([1,-1,1]))
    test_picture.append(np.array([-1,1,-1]))
    test_Dim = 3
    weight = caculateWeight(picture,Dim)
    threshold = caculateThreshold(weight)
    for test_pic in testPicture:
        output = association(weight,threshold,test_pic,Dim)
        strr = print_result(test_pic,col)
        # print('input:')
        # print(strr)
        str_test.append(strr)
        strr = ''
        strr = print_result(output,col)
        # print('output:')
        # print(strr)
        str_output.append(strr)
    return str_test,str_output

def print_result(output,col):
    count = 0
    str_temp = ''
    for element in output:
        if int(element) == 1:
            str_temp += str(element)
        elif int(element) == -1:
            str_temp += '  '
        count += 1
        if count == col:
            str_temp += '\n'
            count = 0
    return str_temp


if __name__ == "__main__":
    windows = tk.Tk()
    app = Application(windows)
    windows.mainloop()






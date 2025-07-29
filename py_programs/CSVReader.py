#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:15:43 2025

@author: tim
"""

import csv
import shutil
from pathlib import Path


PATH_CSV = './emissions.csv'
csvfile =  open(PATH_CSV, newline='')
source = Path(PATH_CSV)
PATH_SAVE_DATAS = "./emissions_datas"



'''spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

for row in spamreader:
    a = 6
    if len(row)> a :
        print(row[a])
    #print(', '.join(row))'''
    
class CSVfile():
    
    
    def __init__(self, path = './emissions.csv'):
        self.file =  open(path, newline='')
        self.line_red = csv.reader(self.file, delimiter=',', quotechar='/')
        # we get the content beacause the iterator is not readable several times
        self.categories = self.line_red.__next__() 
        self.n_column = len(self.categories)
        self.content = [] 
        for line in self.file:
            self.content.append(line.split(","))
            
    ## print the labels of the columns
    def print_categories(self):
        print(self.categories)

    # TODO test
    def get_columns(self, name_cols):
        result = []
        for name_col in name_cols:
            result.append(self.get_column(name_col))
        return result

    # 
    def get_column(self, name_col : str):
        col = []
        for i in range(self.n_column):
            if self.categories[i] == name_col:
                for n_line in range(len(self.content)):
                    col.append(self.content[n_line][i])
        return col

    def print_columns(self, name_cols):
        T="id_result"
        for i in range(1,len(name_cols)):
            T+="/"+ name_cols[i]
        print(T)
        result = self.get_columns(name_cols)
        for i in range(len(result[0])):
            T = str(i)
            for col in result:
                T +=", "+col[i]
            print(T)
            '''        for i in range(self.n_column):
            if self.categories[i] == name_col:
                #print(self.categories[i], " found")
                for n_line in range(len(self.content)):
                    print(self.content[n_line][i])'''
            print()
    
    def save_file_and_clean(self, name_saved_file, path_saev_file = PATH_SAVE_DATAS):
        #TODO may not work on Windows
        path_new_file =path_saev_file+"/"+name_saved_file
        f= open(path_new_file, 'x')
        f.close()
        shutil.move(PATH_CSV, path_new_file)
    
                    

'''file1 = CSVfile()
file1.print_categories()
file1.print_columns(["tracking_mode", "timestamp"])
#TODO test this below
file1.save_file_and_clean("test2.csv", PATH_SAVE_DATAS)'''


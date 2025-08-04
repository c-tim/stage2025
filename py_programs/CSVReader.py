#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:15:43 2025

@author: tim
"""

import csv
import shutil
from pathlib import Path
import DataTools


PATH_CSV = './emissions.csv'
#csvfile =  open(PATH_CSV, newline='')
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
        self.categories = ["id_col"] + self.line_red.__next__() 
        self.n_column = len(self.categories)
        self.content = [] 
        i = 2 #start at 2 because the first line is for the categories
        for line in self.file:
            #print("line ", i, " : ", line)
            self.content.append([str(i)] + line.split(","))
            i+=1
            
    ## print the labels of the columns
    def print_categories(self):
        print(self.categories)
    
    def extract_data(self, name_cols, condFilter : dict() = None):
        # we remove the ids of the line that doesnt match the conditions in condFilter
        id_line_kept = range(len(self.content))
        
        name_cols = DataTools.str_to_singleton(name_cols)
        
        if condFilter is not None : 
            for col_name in condFilter:             
                # changing single input in list for generalization
                if type(condFilter[col_name]) == str :
                    condFilter[col_name] = [condFilter[col_name]]
                
                col_data = self.get_column(col_name)   
                list_id_remove_next_iteration=[]
                for i in id_line_kept:
                    #if the string has a ':' we get the first part only 
                    first_part_label =  (col_data[i].split(':'))[0]
                    
                    if first_part_label not in condFilter[col_name]:
                        list_id_remove_next_iteration.append(i)
                new_list = []
                for id_ancient_list in id_line_kept:
                    if id_ancient_list not in list_id_remove_next_iteration:
                        new_list.append(id_ancient_list)
                id_line_kept = new_list
        return self.get_columns(name_cols, id_line_kept)
                

    # TODO test
    def get_columns(self, name_cols, filter_col = None):
        result = []
        for name_col in name_cols:
            result.append(self.get_column(name_col, filter_col))
        return result

    # 
    def get_column(self, name_col : str, filter_col = None):
        col = []
        
        # filter col contains the ids of the line filtered by extract_data
        if filter_col is None :
            filter_col = range(len(self.content))
        
        for i in range(self.n_column):
            if self.categories[i] == name_col:
                for n_line in filter_col:
                    col.append(self.content[n_line][i])
        return col

    def print_columns(self, name_cols, condFilter = None):
        if "id_col" not in name_cols:
            name_cols.insert(0, "id_col")
        T=name_cols[0]
        for i in range(1,len(name_cols)):
            T+="/"+ name_cols[i]
        print(T)
        result = self.extract_data(name_cols, condFilter)
        for i in range(len(result[0])):
            #T = str(i)
            T= result[0][i]
            '''for col in result:
                T +=", "+col[i]'''
            for n_col in range(1,len(result)):
                T += ", "+result[n_col][i]
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
file1.print_columns(["id_col", "tracking_mode", "timestamp"])
file1.print_columns(["tracking_mode", "timestamp"], {"project_name":"test"})
print("and again")
file1.print_columns(["id_col", "tracking_mode", "timestamp"], {"project_name":["test"]})
'''
#TODO test this below
#file1.save_file_and_clean("test2.csv", PATH_SAVE_DATAS)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:15:43 2025

@author: tim
"""

import csv
import shutil
import DataTools
import DataValidation
    
class CSVfile():
    
    PATH_SAVE_DATAS = "../emissions_datas"
    
    def create_file(path):
        path = DataValidation.addSuffixIfNecessary(path, ".csv")
        try:
            open(path, 'x')
        except :
            print("Error file already created")
        return CSVfile(path)
    
    def open_file(path):
        return CSVfile(path)


    def __init__(self, path = './emissions.csv'):
       
        self.path = path
        self.refresh_read_file()
    
    def refresh_read_file(self):
        self.file =  open(self.path, 'r',newline='')
        self.line_red = csv.reader(self.file, delimiter=',', quotechar='/')
        # we get the content beacause the iterator is not readable several times
        '''if len(self.line_red)>0:
            self.categories = ["id_col"] + self.line_red.__next__() 
        else :
            self.categories = ["id_col"]'''
        try :
            self.categories = self.line_red.__next__() 

        except :
            self.categories = ["id_col"]    #works for now
        if self.categories[0] != "id_col":
            self.categories[0]="id_col"

            
        self.n_column = len(self.categories)
        self.content = [] 
        i = 2 #start at 2 because the first line is for the categories
        for line in self.file:
            #print("line ", i, " : ", line)
            self.content.append([str(i)] + line.split(","))
            i+=1
        self.file.close()

    
            
    ## print the labels of the columns
    def get_categories(self):
        '''
        Print the label of each categorie.
        '''
        print(self.categories)
        return self.categories
    
    def extract_data(self, name_cols, condFilter : dict() = None):
        """
        Get the datas from the file.

        Parameters
        ----------
        name_cols : List(str)
            The name of the columns to extract.
        condFilter : dict(), optional
            Apply a filter on the research (ex : {"project_name":"test"}). The default is None.

        Returns
        -------
        List(data)

        """
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
                

    def get_columns(self, name_cols, filter_col = None):
        result = []
        for name_col in name_cols:
            result.append(self.get_column(name_col, filter_col))
        return result

    def get_column(self, name_col : str, filter_col = None):
        col = []
        
        # filter col contains the ids of the line filtered by extract_data
        if filter_col is None :
            filter_col = range(len(self.content))
        
        id_col = self.get_id_column(name_col)
        
        for n_line in filter_col:
            print(self.content[n_line])
            col.append(self.content[n_line][id_col])
        return col
    
    def get_id_column(self, name_col):
        print(self.categories)

        for i in range(len(self.categories)):
            if self.categories[i] == name_col:
                return i

    def get_columns_value(self, name_cols, condFilter = None):
        name_cols=DataTools.str_to_singleton(name_cols)
        '''if "id_col" not in name_cols:
            name_cols.insert(0, "id_col")'''
        result = self.extract_data(name_cols, condFilter)
        columns_values = []
        for i in range(len(result[0])):
            #T = str(i)
            T= str(result[0][i])
            '''for col in result:
                T +=", "+col[i]'''
            for n_col in range(1,len(result)):
                T += ", "+str(result[n_col][i])
            columns_values.append(T)
        return columns_values

    def print_columns(self, name_cols, condFilter = None):
        
        name_cols=DataTools.str_to_singleton(name_cols)
        '''if "id_col" not in name_cols:
            name_cols.insert(0, "id_col")'''
        T=name_cols[0]
        for i in range(1,len(name_cols)):
            T+="/"+ name_cols[i]
        result = self.extract_data(name_cols, condFilter)
        for i in range(len(result[0])):
            #T = str(i)
            T= str(result[0][i])
            '''for col in result:
                T +=", "+col[i]'''
            for n_col in range(1,len(result)):
                T += ", "+str(result[n_col][i])
        return T
    
    
    def add_columns(self, name_categories, values_categories = None):
        
        if values_categories is None:
            for cat in name_categories:
                self.add_column(cat)
            return
        for cat, value in name_categories, values_categories:
            self.add_column(cat, value)
    
    def add_column(self, name_categorie : str, values_categories = None):
        self.categories.append(name_categorie)
        if values_categories is None :
            for i in range(len(self.content)):
                self.content[i].append('Null')
        else :
            
            if len(self.content) == 0:
                print("blanck file, adding lines")
                
            elif len(self.content) == 0:
                print("blanck file, adding lines")
            
            for i in range(len(values_categories)):
                if i >= len(self.content):
                    self.content.append(["Null" for a in range(len(self.categories)-1)])
                    self.content[i][0] = i
                self.content[i].append(values_categories[i])
    
    def set_data_at_col_and_line(self, name_col, id_line, text):
        self.content[id_line][self.get_id_column(name_col)] = text
    
    def clear_file(self):
        f = open(self.path, "w+")
        f.truncate()
        f.close()
                
    def write_data_to_file(self):
        self.clear_file()
        f=open(self.path, 'a')
        writer = csv.writer(f)
        writer.writerow(self.categories)
        writer.writerows(self.content)
        f.close()

    def save_file(self, path_save_data):
        #TODO may not work on Windows
        #path_new_file =path_saev_file+"/"+name_saved_file
        f= open(path_save_data, 'x')
        f.close()
        shutil.move(self.path, path_save_data)
        self.path = path_save_data
        



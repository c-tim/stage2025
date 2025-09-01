#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:16:45 2025

@author: tim
"""

from codecarbon import track_emissions
from CSVReader import CSVfile
import matplotlib.pyplot as plt
import DataTools
import DataValidation

class EnergyAnalyzer():
    """
    An object that track the emissions of chosen functions (using codecarbon)
    and deals with its data stored in a csv file.

    Parameters
    ----------
    name_project : str
        The name displayed in the column "project_name". Used to retrieve datas in the file.
    name_output_file : str, optional
        The csv file sotring the emissions datas. The default is "emissions.csv".
    -------
    """
    def __init__(self, name_project : str, name_output_file :str = "emissions.csv"):

        DataValidation.addSuffixIfNecessary(name_output_file, ".csv")
        self.current_project_name = ""
        self.set_new_project(name_project)
        self.csvResult = None
        self.name_file = name_output_file
        #self.recup_file_result()
        
    def set_new_project(self, name_project : str):
        """
        Change the name of the project (value of the column 'project_name').
        """
        self.current_project_name = name_project
    
    def track_function(self,f, additional_infos = ""):
        """
        track the emissions of a function using codecarbon module
        
        Usage :
            test_analyse = EnergyAnalyzer("test_project")
            
            def f(a):
                return a*2
            tracked_function = test_analyse.track_function(f)
            print(tracked_function(5))
            --> return 10

        Parameters
        ----------
        f : function to track no argument
            DESCRIPTION.
        additional_infos : TYPE, optional
            DESCRIPTION. The default is "".
        """
        label =self.current_project_name+ ":" + f.__name__
        if additional_infos != "":
            label += ":"+additional_infos
        @track_emissions(project_name=label, output_file=self.name_file)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped  
    
    def recup_file_result(self, path_file = ""):
        if path_file == "":
            path_file= self.name_file

        DataValidation.addSuffixIfNecessary(path_file, ".csv")
        self.csvResult = CSVfile.create_file(path_file)
    
    def convertData(self, col):
        l=[]
        for element in col:
            l.append(DataTools.scientificNotation_to_double(element))
        return l
    
    def display_data_axis(self, col_names, name_file="" ,x_axis = "", x_col = [],condFilter = None):
        """
        Warning : do not put ',' in the project_name, it is processed as another case for the CSV reader

        Parameters
        ----------
        col_names : TYPE
            DESCRIPTION.
        name_file : TYPE, optional
            DESCRIPTION. The default is "".
        x_axis : TYPE, optional
            DESCRIPTION. The default is "".
        condFilter : TYPE, optional
            DESCRIPTION. The default is None.

        """
        #self.blank_function()
        self.recup_file_result(name_file)
        #one col is given
        if type(col_names)==str:
            col_names = [col_names]
        
        
        cols_result = self.csvResult.extract_data(col_names, condFilter)

        # if the resukt and the x axis are given, take them
        if len(x_col) > 0 :
            col_label_result = x_col
            if x_axis == "":
                print("WARNING : there is no name for x_axis although data for the x column has been given.")
                x_axis = "Datas"
        #else if a name for x col has been set but not data, it must be a col from the csv file
        elif x_axis != "":
            col_label_result = self.csvResult.extract_data(x_axis, condFilter)
        
        #else the x axis will be ids of tests
        else :
            # we get the last part of the label "project:current_test"-> "current_test"
            col_label_result = self.csvResult.extract_data("project_name", condFilter)[0]
            for i in range(len(col_label_result)):
                #we add i at the end in cases several tests are done with the same name
                col_label_result[i]=col_label_result[i].split(":")[-1]+str(i+1)             
            x_axis = "label test"
        l=[]
        for col_res in cols_result:
            l.append(self.convertData(col_res))
        cols_result = l
        return self.display_graph_col_on_2_axis(x_axis, col_names, col_label_result, cols_result)    
    
    def display_graph_col_on_2_axis(self, label_x, label_y,col_data_x, cols_data_y):
        all_label = ""
        args = []
        #loop pour afficher plusieurs courbes en memes temps (plusieurs y et meme x)
        for i in range(len(cols_data_y)):
            if i > 0:
                all_label+=", "
            all_label += label_y[i]
            args.append(col_data_x)
            args.append(cols_data_y[i])
        
        plt.plot(*args)
        plt.xlabel(label_x)
        plt.ylabel(all_label)
        plt.suptitle(self.current_project_name)
        plt.legend()
        plt.show()
    
    def analyse_model(self, model, database,output_file = ""):
        if output_file == "":
            output_file = self.csvResult.path
        
        f = self.track_function(model.train, model.name)
        f(database, "analysis_model.pth")   
        lastline = self.csvResult.last_line()
        print("Model : ", model.name)
        cats = ["emissions","emissions_rate","cpu_energy","ram_energy","energy_consumed"]
        for cat in cats:
            print(cat, " : ", lastline[self.csvResult.get_id_column(cat)])
        
                

        
        
        
        

            
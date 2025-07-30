#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:16:45 2025

@author: tim
"""

from codecarbon import track_emissions
from CSVReader import CSVfile

class EnergyAnalyzer():
    
    def __init__(self, name_project : str):
        self.current_project_name = ""
        self.set_new_project(name_project)
    
    def set_new_project(self, name_project : str):
        self.current_project_name = name_project

    def track_function(self,f, additional_infos = ""):
        label =self.current_project_name+ ":" + f.__name__
        if additional_infos != "":
            label += "("+additional_infos+")"
        @track_emissions(project_name=label)
        def wrapped(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapped  
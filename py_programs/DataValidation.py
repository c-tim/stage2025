#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:20:08 2025

@author: tim

Module utilisé pour détecter les erreurs pendant le lancement d'un programme couteux'

"""



def isCorrectType(var, t : type):
    return type(var)==t

def isCorrectRange(var, threshold_min = None, threshold_max = None):
    error = False
    if threshold_min != None:
        error = error or var < threshold_min
        
    if threshold_max != None:
        error = error or var > threshold_max
        
    return not error

def addSuffixIfNecessary(path : str, suffixe : str)->str:
    
    if not suffixe.startswith("."):
        suffixe='.'+suffixe
    
    if not path.endswith(suffixe):
        path+=suffixe
    return path

#TODO may be usefull later to verify the arguments of a function before launching it
def launch_delayed():
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapped

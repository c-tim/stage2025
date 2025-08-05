#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:00:20 2025

@author: tim
"""

from classEnergyAnalyzer import EnergyAnalyzer

analysor  = EnergyAnalyzer("CompExampleModel")
analysor.display_data_axis(["ram_energy", "energy_consumed"], name_file= "./emissions_CompareExamples.csv")
analysor.csvResult.print_columns(["project_name"])
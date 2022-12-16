# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:20:23 

"""

def import_csv_to_DF_transpose(file_name):
    """
    imports csv file and makes DF then transposes dataframe
    and returns df and df_transposed to be renamed by user
    """
    import pandas as pd
    df = pd.read_excel(file_name) 
    df_transposed = df.transpose()
    return df, df_transposed
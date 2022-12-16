# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:33:59 2022

@author: ag11afr
"""

#import required modules

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from import_csv_to_DF_transposeimport import import_csv_to_DF_transpose


df_all_data, df_all_data_transposed = import_csv_to_DF_transpose("Climate_Change_Data_Set.xlsx")

# Make datafrane (df_all_data) for climate change large data set from xlsx file named appropriatly

#df_all_data = pd.read_excel("Climate_Change_Data_Set.xlsx")  

#import_csv_to_DF_transpose("Climate_Change_Data_Set.xlsx")


# removes ALL rows containing nan cells

df_all_data = df_all_data.dropna()

# data frame containing first row to make the index values
first_row = df_all_data.iloc[0]

# Use the `rename` method to rename the columns to make the df more clear
df_all_data.rename(columns=first_row, inplace=True)


df_pop_total = df_all_data.loc[df_all_data['Indicator Code'].str.contains('SP.POP.TOTL')]




# Sort the dataframe by the '2021.0' column in descending order
df_pop_total = df_pop_total.sort_values(2021, ascending=False)

# Select the top 10 rows
top_10 = df_pop_total.head(10)

# Get the value in column '2021' for the row containing 'China'
china_value = df_pop_total.loc[df_pop_total['Country Name'] == 'China', 2021]

# extract value on its own for china population

china_value = china_value.iloc[0]

# Select rows from china and below
df_pop_total = df_pop_total.loc[df_pop_total[2021] <= china_value]

# extract top 10 countries data from clearer dataframe

df_pop_total_top_10 = df_pop_total.iloc[[0, 1, 30, 31, 32, 33, 34, 35, 36, 37]]


# Extract the 'Country Name' values for the top 10 rows
top_10_country_names = df_pop_total_top_10['Country Name']


# Extract the columns from the DataFrame to plot
columns = df_pop_total_top_10.columns[5:]



# Iterate through the rows of the DataFrame and plot each by country
# divide values by 1,000,000,000 to simplify units for report

for i, row in df_pop_total_top_10.iterrows():
    country_name = row['Country Name']
    population_data = row[columns]
    population_data = population_data/1000000000
    plt.plot(columns, population_data, label=country_name)

# Add a legend and show the plot
plt.legend()

# Label the y-axis
plt.ylabel('Population (Billions)')

plt.show()




print(df_all_data)

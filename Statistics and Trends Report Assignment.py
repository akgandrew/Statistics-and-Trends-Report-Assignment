# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:33:59 2022

@author: ag11afr
"""

#import required modules

import matplotlib.pyplot as plt

import pandas as pd

import import_csv

from scipy.stats import shapiro

from scipy.stats import pearsonr

from scipy.stats import ttest_ind


# using function import_csv_to_DF_transpose from module import_csv

df_all_data, df_all_data_transposed = import_csv.import_csv_to_DF_transpose("Climate_Change_Data_Set.xlsx")

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

# Add a legend and place it to the right of the graph
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Label the y-axis
plt.ylabel('Population (Billions)')
plt.xlabel('Time (Years)')
plt.show()

print(df_all_data)


# shifting to using transposed dataframe where 
# countries and environmental factors arranged as columns

# remove unwanted first 3 columns
df_all_data_transposed = df_all_data_transposed.iloc[:, 3:]

#make years and row subjects the index for row.
df_all_data_transposed = df_all_data_transposed.set_index(3)

# Convert top_10_country_names to a DataFrame
df_top_10_country_names = top_10_country_names.to_frame()


#identifies top 10 country data by columns
columns_to_keep = df_all_data_transposed.columns[df_all_data_transposed.isin(top_10_country_names.values.flatten()).any()]

##just keeps the data for the top 10 coutries population wise

df_all_data_transposed = df_all_data_transposed.loc[:, columns_to_keep]

# We are going to look at correlations so need to get rid of data with less 
# than 50% (self selected cut off) of data for each variable

# Create boolean df that indicates which values are NaN
df_nan = df_all_data_transposed.isnull()

# Count the number of NaN values in each column of data
nan_count = df_nan.sum()

# Select only the columns that have mostly full data sets first 4 rows will be nan 
# so will allow 11 other missing values so we have 75% or data minimum

df_filtered = df_all_data_transposed.loc[:, nan_count < 15]


#change to df notation for dataframe for upcoming function

df = df_filtered


# Loop through the columns of the dataframe to find shapiro wilk and ttest 
# and non parametric equivalents for analysis later


for col, data in df.iteritems():
    
    
    # Slice the data from row 4 to the last row
    data_sliced = data.iloc[4:]

    # Calculate the Shapiro-Wilk test for normality
    stat, p = shapiro(data_sliced)
    mean_stat = data_sliced.mean()
    stdev_stat = data_sliced.std()

    # If the p-value is less than 0.05, the data is not normal
    if p < 0.05:
        result = 'Data is not normal'
    else:
        result = 'Data is normal'

    # Add a new row at the bottom of the column with the results of the Shapiro-Wilk test
    df.at['Normality Result', col] = result
    df.at['Mean Result', col] = mean_stat
    df.at['Stdev Result', col] = stdev_stat

# creat new dataframe
data_filtered = pd.DataFrame()

 # Identify the column with data that includes both 'country code' and 'variable code'
column_label1 = df.columns[df.apply(lambda x: x.str.contains('CHN').any() & x.str.contains('EN.ATM.CO2E.LF.KT').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label1]

# add column to dataframe column 1
data_filtered = data_filtered.assign(col1=df_filtered[column_label1])

column_label2 = df.columns[df.apply(lambda x: x.str.contains('CHN').any() & x.str.contains('SP.POP.TOTL').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label2]

# add column to dataframe column 2
data_filtered = data_filtered.assign(col2=df_filtered[column_label2])


 # Identify the column with data that includes both 'country code' and 'variable code'
column_label3 = df.columns[df.apply(lambda x: x.str.contains('IND').any() & x.str.contains('EN.ATM.CO2E.LF.KT').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label3]

# add column to dataframe column 3
data_filtered = data_filtered.assign(col3=df_filtered[column_label3])

column_label4 = df.columns[df.apply(lambda x: x.str.contains('IND').any() & x.str.contains('SP.POP.TOTL').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label4]

# add column to dataframe column 4
data_filtered = data_filtered.assign(col4=df_filtered[column_label4])

# dat removed for last 5 years as data missing for C02.

data_corr_ready = data_filtered[5:-8]

#data adjusted to billions for clearer graphs
data_corr_ready["col2"] = data_corr_ready["col2"]/1000000000
data_corr_ready["col4"] = data_corr_ready["col4"]/1000000000

# Select the data for the x and y axes
x1 = data_corr_ready["col1"]
y1 = data_corr_ready["col2"]
x2 = data_corr_ready["col3"]
y2 = data_corr_ready["col4"]

# Create the scatter plot
plt.scatter(x1, y1, c='b', label='China')
plt.scatter(x2, y2, c='r', label='India')
plt.ylabel("Population total (Billions)")
plt.xlabel("CO2 emissions from liquid fuel consumption (kt)")

# Add a legend
plt.legend()

# Show the plot
plt.show()

##data was reported as normal so a persons correlation undertook.

# Extract the two columns of interest
x = x1
y = y1

# Compute the Pearson correlation coefficient
corr1, pval1 = pearsonr(x, y)

## for China P value less than 0.0001 and r value 0.904
# Extract the two columns of interest
x = x2
y = y2

# Compute the Pearson correlation coefficient
corr2, pval2 = pearsonr(x, y)

## for India P value less than 0.0001 and r value 0.954


# creat new dataframe
data_filtered = pd.DataFrame()

# Identify the column with data that includes both 'country code' and 'variable code'
column_label1 = df.columns[df.apply(lambda x: x.str.contains('CHN').any() & x.str.contains('AG.LND.ARBL.ZS').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label1]

# add column to dataframe column 1
data_filtered = data_filtered.assign(col1=df_filtered[column_label1])

column_label2 = df.columns[df.apply(lambda x: x.str.contains('CHN').any() & x.str.contains('SP.URB.TOTL.IN.ZS').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label2]

# add column to dataframe column 2
data_filtered = data_filtered.assign(col2=df_filtered[column_label2])


 # Identify the column with data that includes both 'country code' and 'variable code'
column_label3 = df.columns[df.apply(lambda x: x.str.contains('IND').any() & x.str.contains('AG.LND.ARBL.ZS').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label3]

# add column to dataframe column 3
data_filtered = data_filtered.assign(col3=df_filtered[column_label3])

column_label4 = df.columns[df.apply(lambda x: x.str.contains('IND').any() & x.str.contains('SP.URB.TOTL.IN.ZS').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label4]

# add column to dataframe column 4
data_filtered = data_filtered.assign(col4=df_filtered[column_label4])

# prepare data for statistical analysis
data_corr_ready = data_filtered[5:-4]


# Select the data for the x and y axes
x1 = data_corr_ready["col1"]
y1 = data_corr_ready["col2"]
x2 = data_corr_ready["col3"]
y2 = data_corr_ready["col4"]

# Create the scatter plot
plt.scatter(x1, y1, c='b', label='China')
plt.scatter(x2, y2, c='r', label='India')
plt.ylabel("Urban population (% of total population)")
plt.xlabel("Arable land (% of land area)")

# Add a legend
plt.legend()

# Show the plot
plt.show()


# Extract the two columns of interest
x = x1
y = y1

# Compute the Pearson correlation coefficient
corr3, pval3 = pearsonr(x, y)

## for China P value less than 0.0001 and r value 0.702
# Extract the two columns of interest
x = x2
y = y2

# Compute the Pearson correlation coefficient
corr4, pval4 = pearsonr(x, y)

## for India P value less than 0.0001 and r value -0.42


#### comparing mean values between china and india

### Comparing Urban population growth
# creat new dataframe
data_filtered = pd.DataFrame()

# Identify the column with data that includes both 'country code' and 'variable code'
column_label1 = df.columns[df.apply(lambda x: x.str.contains('IND').any() & x.str.contains('SP.URB.GROW').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label1]

# add column to dataframe column 1
data_filtered = data_filtered.assign(col1=df_filtered[column_label1])

column_label2 = df.columns[df.apply(lambda x: x.str.contains('CHN').any() & x.str.contains('SP.URB.GROW').any())]

# Filter the original DataFrame to include only the identified column
df_filtered = df[column_label2]

# add column to dataframe column 2
data_filtered = data_filtered.assign(col2=df_filtered[column_label2])

# Prepare data in suitable format for plot and analysis
mean_datarow = data_filtered.loc['Mean Result']
stdev_datarow = data_filtered.loc['Stdev Result']
mean_data = mean_datarow.to_frame()
stdev_data = stdev_datarow.to_frame()


# Data to be plotted
mean_data = pd.DataFrame({'Country': ['India', 'China'], 'Mean': [mean_data.iloc[0,0], mean_data.iloc[1,0]]})
stdev_data = pd.DataFrame({'Country': ['India', 'China'], 'Std Dev': [stdev_data.iloc[0,0], stdev_data.iloc[1,0]]})


# Merge the data into a single DataFrame
data = mean_data.merge(stdev_data, on='Country')

# Create the bar chart
plt.bar(data['Country'], data['Mean'], yerr=data['Std Dev'], capsize=5)
plt.ylabel("Urban population growth (annual %)")
plt.plot(1, 5, marker='*', markersize=10, color='red')
# Show the plot
plt.show()

# Both data Were normally normally distributed (P>0.05) 
# according to Shapiro Wilk so a ttest was performed 

#prepare data for ttest(removing nans)

India_china_data = data_filtered[5:-3]
India_data = India_china_data['col1']
China_data = India_china_data['col2']

#perform ttest

t_stat, p_value = ttest_ind(India_data, China_data)

# t value = 3.02 p value =0.003


# Extract the two columns of interest
x = India_data 
y = China_data

# Compute the Pearson correlation coefficient
corr, pval = pearsonr(x, y)


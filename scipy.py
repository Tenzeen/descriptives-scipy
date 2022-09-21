#Import packages
import pandas as pd
from pandas import plotting
import numpy as np
import scipy
from scipy import stats
import os
import urllib.request
from statsmodels.formula.api import ols
import seaborn as se
import matplotlib.pyplot as plt

#load in dataset
df = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
df

#use numpy to create arrays and lists
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)

#use pandas to build the dataframe
pd.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})  

#visualize dataset
df.shape    # 40 rows and 8 columns
df.columns  # It has columns   
print(df['Gender'])  # Columns can be addressed by name   

# Simpler selector
df[df['Gender'] == 'Female']['VIQ'].mean()

#splitting a dataframe on values of categorical variables:
groupby_gender = df.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))

groupby_gender.mean()


#Exercise 1
#find mean
df['VIQ'].mean() #mean = 112.35

#male and female count
groupby_gender = df.groupby('Gender')
groupby_gender.count() #male = 20, female = 20

#average mri for males and females in log units
df['MRI_base10'] = np.log10(df['MRI_Count'])
groupby_gender = df.groupby('Gender')
for gender, value in groupby_gender['MRI_base10']:
    print((gender, value.mean())) #male 5.934, female 5.979

#Plotting
plotting.scatter_matrix(df[['Weight', 'Height', 'MRI_Count']])   
plotting.scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ']])  

#Exercise 2
#scatter plot matrix for males and females (seperate)
plotting.scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ',]], c=(df['Gender'] == 'Female'))
plotting.scatter_matrix(df[['PIQ', 'VIQ', 'FSIQ',]], c=(df['Gender'] == 'Male'))

#Hypothesis Testing
#1-sample test
stats.ttest_1samp(df['VIQ'], 0)

#2-sample t-test
female_viq = df[df['Gender'] == 'Female']['VIQ']
male_viq = df[df['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)   

#paired test
stats.ttest_ind(df['FSIQ'], df['PIQ'])   
stats.ttest_rel(df['FSIQ'], df['PIQ'])   
stats.ttest_1samp(df['FSIQ'] - df['PIQ'], 0)   
stats.wilcoxon(df['FSIQ'], df['PIQ'])

#Exercise 3
#what is the differece between weights in males and females
stats.ttest_1samp(df.dropna()['Weight'], 0)
female_weight = df.dropna()[df['Gender'] == 'Female']['Weight']
male_weight = df.dropna()[df['Gender'] == 'Male']['Weight']
stats.ttest_ind(female_weight, male_weight)

female_viq = df.dropna()[df['Gender'] == 'Female']['VIQ']
male_viq = df.dropna()[df['Gender'] == 'Male']['VIQ']
scipy.stats.mannwhitneyu(female_viq, male_viq)

#linear regression
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
df = pd.DataFrame({'x': x, 'y': y})
model = ols("y ~ x", df).fit()
print(model.summary())  

#categorical variables
#compare IQ of males and females.
df = pd.read_csv('data/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", df).fit()
print(model.summary())  
model = ols('VIQ ~ C(Gender)', df).fit()

#create table
data_fisq = pd.DataFrame({'iq': df['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': df['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)  

model = ols("iq ~ type", data_long).fit()
print(model.summary()) 
stats.ttest_ind(df['FSIQ'], df['PIQ'])   


#multiple regression
flower = pd.read_csv('data/iris.csv')
flower_model = ols('sepal_width ~ name + petal_length', df).fit()
print(model.summary())

#test
print(model.f_test([0, 1, -1, 0])) 

#Exercise 5
model = ols('VIQ ~ Gender + MRI_Count + Height', df).fit()
print(model.summary())
print(model.f_test([0, 1, -1, 0]))

#visualization
if not os.path.exists('wages.txt'):
    # Download the file if it is not present
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                       'wages.txt')

# Give names to the columns
names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]

short_names = [n.split(':')[0] for n in names]

wages = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None,
                       header=None)
wages.columns = short_names

# Log-transform the wages, because they typically are increased with
# multiplicative factors
wages['WAGE'] = np.log10(wages['WAGE'])

se.pairplot(wages, vars=['WAGE', 'AGE', 'EDUCATION'],
                      kind='reg')

se.pairplot(wages, vars=['WAGE', 'AGE', 'EDUCATION'],
                      kind='reg', hue='SEX')

#simple regression
se.lmplot(y='WAGE', x='EDUCATION', data=wages)

plt.show()

#test for interactions
result = sm.OLS(formula='wage ~ education + gender + education * gender',
                data=wages).fit()    
print(result.summary())  
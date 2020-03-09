import pandas as pd
import numpy as np

filepath = './datasets/athletes/'

data_medals = pd.read_csv(f'{filepath}Medals.csv', encoding='ISO-8859-1')
data_country = pd.read_csv(f'{filepath}Athelete_Country_Map.csv', encoding='ISO-8859-1')
data_country = data_country.drop_duplicates(subset='Athlete')
data_sports = pd.read_csv(f'{filepath}Athelete_Sports_Map.csv', encoding='ISO-8859-1')
data_sports = data_sports.drop_duplicates(subset='Athlete')

# Getting randomly 6 atheletes to remove their records from the datasets
out_athletes = np.random.choice(data_medals['Athlete'], size=6, replace=False)
print('\nAtheletes to remove...')
print(f'Michael Phelps and these athletes to remove from datasets =>{out_athletes}')


# Setting a 'copy' from the datasets, but without the random atheletes (Michael Phelps included)
data_medals_dlt = data_medals[(~data_medals['Athlete'].isin(out_athletes)) &
                               (data_medals['Athlete'] != 'Michael Phelps')]

data_country_dlt = data_country[(~data_country['Athlete'].isin(out_athletes)) &
                                (data_country['Athlete'] != 'Michael Phelps')]

data_sports_dlt = data_sports[(~data_sports['Athlete'].isin(out_athletes)) &
                              (data_sports['Athlete'] != 'Michael Phelps')]

print('\nData remove (length)')
print(f'len(data_medals) - len(data_medals_dlt) => {len(data_medals) - len(data_medals_dlt)}')
print(f'len(data_country) - len(data_country_dlt) => {len(data_country) - len(data_country_dlt)}')
print(f'len(data_sports) - len(data_sports_dlt) => {len(data_sports) - len(data_sports_dlt)}')

print('\nOriginal length datasets (without repeated atheles in countries and sports)')
print(f'len(data_medals) => {len(data_medals)}')
print(f'len(data_country) => {len(data_country)}')
print(f'len(data_sports) => {len(data_sports)}')

# Merging with inner join (intersection between 2 datasets in 'Athlete' values)
merged_inner = pd.merge(left=data_medals, right=data_country_dlt, how='inner', left_on='Athlete', right_on='Athlete')
print('\ninner join data_medals and data_country_dlt')
print(f'len(merged_inner) => {len(merged_inner)}')

# Merging with left join (first dataset)
merged_left = pd.merge(left=data_medals, right=data_country_dlt, how='left', left_on='Athlete', right_on='Athlete')
print('\nleft join data_medals and data_country_dlt')
print(f'len(merged_left) => {len(merged_left)}')
# print(merged_left.head()) #look that Michael Phelps country is NaN

# Merging with right join (second dataset)
merged_right = pd.merge(left=data_medals_dlt, right=data_country, how='right', left_on='Athlete', right_on='Athlete')
print('\nright join data_medals_dlt and data_country')
print(f'len(merged_right) => {len(merged_right)}')
# print(merged_right.tail(10)) #look NaNs in the lasts rows

# Merging with outer join (union)
data_country_custom = data_country_dlt.append(
    {
        "Athlete": "Alejandro Rivera",
        "Country": "Mexico"
    }, ignore_index=True
)
merged_outer = pd.merge(left=data_medals, right=data_country_custom, how='outer', left_on='Athlete', right_on='Athlete')
print('\nouter join data_medals and data_country_custom')
print(f'len(merged_outer) => {len(merged_outer)}')
print(merged_outer.tail(10)) #look last one
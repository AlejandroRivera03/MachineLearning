import pandas as pd

filepath = './datasets/athletes/'

# Getting the datasets to join by athelete name
data_medals = pd.read_csv(f'{filepath}Medals.csv', encoding='ISO-8859-1')
data_country = pd.read_csv(f'{filepath}Athelete_Country_Map.csv', encoding='ISO-8859-1')
data_sports = pd.read_csv(f'{filepath}Athelete_Sports_Map.csv', encoding='ISO-8859-1')

print(f'\ndata_medals.shape => {data_medals.shape}')
print('unique atheletes => {}'.format(len(data_medals['Athlete'].unique().tolist())))
print(f'\ndata_country.shape => {data_country.shape}')
print(f'\ndata_sports.shape => {data_sports.shape}')


# In data country there are duplicated atheletes because they
# represented different countries in different olimpic games
# print(data_country[data_country.duplicated(['Athlete'])]['Athlete'])
# repeated = data_country[data_country.duplicated(['Athlete'])]['Athlete'].tolist()
# for index, value in enumerate(repeated):
#     print(data_country[data_country['Athlete'] == value])


# Drop duplicated athletes (who represent more than 1 country, in this case)
# this avoid when merge medals and country datasets, duplicate some records
data_country_dp = data_country.drop_duplicates(subset='Athlete')
print(f'\ndata_country_dp.shape => {data_country_dp.shape}')


# Merge medals dataset with country dataset by athelete name in common
medals_country = pd.merge(left=data_medals, right=data_country_dp, left_on='Athlete', right_on='Athlete')
print(f'\nmedals_country.shape => {medals_country.shape}')


# In data sports there are duplicated atheletes because some of them
# in more that one sport in the olimpic games
# repeated = data_sports[data_sports.duplicated(['Athlete'])]['Athlete'].tolist()
# for index, value in enumerate(repeated):
#     print(data_sports[data_sports['Athlete'] == value])

# Drop duplicated athleted (who plays more than one disciplines in the olimpic games)
data_sports_dp = data_sports.drop_duplicates(subset='Athlete')
print(f'\ndata_sports_dp.shape => {data_sports_dp.shape}')


data_final = pd.merge(left=medals_country, right=data_sports_dp, left_on='Athlete', right_on='Athlete')
print(f'\ndata_final.shape => {data_final.shape}')

print(data_final.head(20))
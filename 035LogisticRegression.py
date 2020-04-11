import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('./datasets/Bank/bank.csv', sep=';')

print(data.head())
print(data.columns.values)

# changing the 'yes/no' y column values to zeros and ones
data["y"] = (data["y"]=="yes").astype(int)

print(data['education'].unique())

# Simplifying 'education' column
# Basic education generalized
data['education'] = np.where(data['education']=='basic.4y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'Basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y', 'Basic', data['education'])
# Better presentation for education values
data['education'] = np.where(data['education']=='high.school', 'High School', data['education'])
data['education'] = np.where(data['education']=='professional.course', 'Professional Course', data['education'])
data['education'] = np.where(data['education']=='university.degree', 'University Degree', data['education'])
data['education'] = np.where(data['education']=='illiterate', 'Illiterate', data['education'])
data['education'] = np.where(data['education']=='unknown', 'Unknown', data['education'])

# mean values respect y column
print(data.groupby('y').mean())

#mean values respect education column
print(data.groupby('education').mean())


## Personas con grado universitario tienden un poco mas a invertir
# Porcentaje respecto a cada categoria (nivel de educacion), el porcentaje es similar
pd.crosstab(data.education, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del nivel de educacion')
plt.xlabel('Nivel de educacion')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.education, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de educacion contra inversion')
plt.xlabel('Dia')
plt.ylabel('Proporcion de clientes')
plt.show()


## Porcentaje de cada categoria (estado civil) es similar
table = pd.crosstab(data.marital, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de estado civil contra inversion')
plt.xlabel('Estado civil')
plt.ylabel('Proporcion de clientes')
plt.show()


## Porcetaje de cada categoria (dia de la semana) es similar
pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del dia de la semana')
plt.xlabel('Dia de la semana')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.day_of_week, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado de dia contra inversion')
plt.xlabel('Dia')
plt.ylabel('Proporcion de clientes')
plt.show()


## En marzo y diciembre se ofrece menos invetir, pero en porcentaje hay mas caso de exito
pd.crosstab(data.month, data.y).plot(kind='bar')
plt.title('Frecuencia de compra en funcion del mes')
plt.xlabel('Mes')
plt.ylabel('Frecuencia de compra del producto')
plt.show()

table = pd.crosstab(data.month, data.y)
table.div( table.sum(1).astype(float), axis=0 ).plot(kind='bar', stacked=True)
plt.title('Diagrama apilado del mes contra inversion')
plt.xlabel('Mes')
plt.ylabel('Proporcion de clientes')
plt.show()


## Personas entre 25 y 60 años invierten en la bolsa de valores
data.age.hist()
plt.title('Histograma de la edad')
plt.xlabel('Edad')
plt.ylabel('Cliente')
plt.show()

pd.crosstab(data.age, data.y).plot(kind='bar')
plt.show()


## Quien pierde en invercion, raramente vuelve a invertir
# Quien no tiene conocimiento de la bolsa, raramente invierte
# Quien gana en inversion, normalmente vuelve a invertir
pd.crosstab(data.poutcome, data.y).plot(kind='bar')
plt.show()

# Changing categorical variables to dummy
categories = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

for category in categories:
    cat_list = f'cat_{category}'
    cat_dummies = pd.get_dummies(data[category], prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new

# Removing categorical variables and keeping dummies
data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in to_keep if v not in ['default']]

bank_data = data[to_keep]


# Building model
bank_data_vars = bank_data.columns.values.tolist()
# print(bank_data_vars)
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]

n = 12
lr = LogisticRegression()
rfe = RFE(lr, n)
rfe = rfe.fit(bank_data[X], bank_data[Y].values.ravel())

# print(rfe.support_)

# print(rfe.ranking_)

z = list(zip(bank_data_vars, rfe.support_, rfe.ranking_))
print(z) # variables that the method recommends to use to build the model (Trues and lower numbers)



# columns that the last print recommends to use in the logistic model
cols = [recommendation[0] for recommendation in z if recommendation[1]]

X = bank_data[cols]
Y = bank_data['y']

logit_model = sm.Logit(Y, X)
result = logit_model.fit()

# P-Valor => the lower value, the more important is (ex. job_student, job_technician, day_of_week_wed aren't good option)
print(result.summary2())

logit_model = LogisticRegression()
logit_model.fit(X,Y)
print(logit_model.score(X,Y))


print(pd.DataFrame(list(zip(X.columns, np.transpose(logit_model.coef_)))))

# LOGISTIC MODEL VALIDATION

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# Making a logistic model with train data
lm = LogisticRegression()
lm.fit(X_train, Y_train)
probs = lm.predict_proba(X_test)
print('\nLeft => security, Right => les tha 0.5 no, more that 0.5 yes')
print(probs)

prediction = lm.predict(X_test)
print(f'\n{prediction}')

prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshol = 0.1
prob_df['prediction'] = np.where(prob_df[0] > threshol, 1, 0)
print(prob_df.head())

print(pd.crosstab(prob_df.prediction, columns='count'))

# 390 / len(prob_df)

print(metrics.accuracy_score(Y_test, prediction))


# CROSS VALIDATION

# K-Iteraciones (10 in this example)
scores = cross_val_score(LogisticRegression(), X, Y, cv=10, scoring='accuracy')

print(f'\nScores => \n{scores}')
print(f'\nScores mean => \n{scores.mean()}')

# Matrices de confucion

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lm = LogisticRegression()
lm.fit(X_train, Y_train)
probs = lm.predict_proba(X_test)
prob = probs[:,1]
prob_df = pd.DataFrame(prob)
threshol = 0.1
prob_df['prediction'] = np.where(prob_df[0]>threshol, 1, 0)
prob_df['actual'] = list(Y_test)
print(f'\n{prob_df.head()}')
confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)
print(f'\n{confusion_matrix}')

TN = confusion_matrix[0][0] # Verdaderos negativos
TP = confusion_matrix[1][1] # Verdaderos positivos
FP = confusion_matrix[0][1] # Falsos negativos
FN = confusion_matrix[1][0] # Falsos positivos

sens = TP / (TP+FN)
print(f'\nSensibilidad => {sens}')

espc_1 = 1-TN/(TN+FP)
print(f'\nEspecifidad => {espc_1}')

# Curvas ROC (Manual)

thresholds = [0.04, 0.05, 0.07, 0.1, 0.12, 0.15, 0.18, 0.20, 0.25, 0.3, 0.4, 0.5]
sensitivities = [1]
especifities_1 = [1]

for t in thresholds:
    prob_df['prediction'] = np.where(prob_df[0]>=t, 1, 0)
    prob_df['actual'] = list(Y_test)

    confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)
    TN = confusion_matrix[0][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]


    sens = TP/(TP+FN)
    sensitivities.append(sens)
    espc_1 = 1-TN/(TN+FP)
    especifities_1.append(espc_1)

sensitivities.append(0)
especifities_1.append(0)
print(f'\nSensibilidades => {sensitivities}')
print(f'\nEspeficidades => {especifities_1}')

plt.plot(especifities_1, sensitivities, marker='o', linestyle='--', color='r')
x=[i*0.01 for i in range(100)]
y=[i*0.01 for i in range(100)]
plt.plot(x, y)
plt.xlabel('1-Especifidad')
plt.ylabel('Sensibilidad')
plt.title('Curva ROC')
plt.show()

# Curvas ROC (Library)

espc_1, sensit, _ = metrics.roc_curve(Y_test, prob)

df = pd.DataFrame({
    'esp': espc_1,
    'sens': sensit,
})

plt.plot(df.esp, df.sens)
plt.show()

auc = metrics.auc(espc_1, sensit)
print(f'\nArea under curve => {auc}')
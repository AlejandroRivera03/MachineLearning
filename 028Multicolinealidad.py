import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# VIF => Factor Inflacion de la Varianza
# VIF == 1 : Las variables no estan correlacionadas
# VIF < 5 : Las variables tienen una correlacion moderada y se pueden quedar en el modelo
# VIF > 5 : Las variables estan altamente correlacionadas y deben desaparecer del modelo

data = pd.read_csv('./datasets/ads/Advertising.csv')

# Newspaper ~ TV + Radio
lm_n = smf.ols(formula='Newspaper~TV+Radio', data=data).fit()
rsquared_n = lm_n.rsquared
VIF_n = 1/(1-rsquared_n)
print(f'VIF Newspaper ~ TV + Radio => {VIF_n}')

# TV ~ Newspaper + Radio
lm_t = smf.ols(formula='TV~Newspaper+Radio', data=data).fit()
rsquared_t = lm_t.rsquared
VIF_t = 1/(1-rsquared_t)
print(f'VIF TV ~ Newspaper + Radio => {VIF_t}')

# Radio ~ TV + Newspaper
lm_r = smf.ols(formula='Radio~TV+Newspaper', data=data).fit()
rsquared_r = lm_r.rsquared
VIF_r = 1/(1-rsquared_r)
print(f'VIF Radio ~ TV + Newspaper => {VIF_r}')
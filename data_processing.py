# -*- coding: utf-8 -*-
#AaltoEE:n data-analytiikan osaaja -rekrytointikurssin projektityön osa, datan prosessointi.
#Huom: tässä vain murto-osa lopullisesta datasta
"""
Created on Sat Mar 14 12:42:05 2020

@author: Pirjo Valjanen
"""


import pandas as pd
from pandas import set_option
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression, Lars, lars_path
import statsmodels.api as sm
from sklearn.decomposition import PCA


#luetaan csv-tiedosto
#data_raw = pd.read_csv("bowling_db280120_all.csv", header=None, encoding = "latin1")
data_raw = pd.read_csv("bowling_top350.csv", header=None, encoding = "latin1")

data_raw.head()

#määritellään pandas näyttämään kaikki rivit ja sarakkeet
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

#lisätään sarakkeiden nimet
column_names = ["CenterName","Country","State","City","Lane","Time","Id","RPM","Loft","Complete","IsRightHanded","LaunchAxis","HookStart","HookEnd","PindeckExitBoard","PindeckSpeed","AverageSpeed","HeadsRPM","MidRPM","BackRPM","ReadStart","Guid","MaxHookDistance","PindeckDeflection","BreakPointDistance","BreakPointPosition","LaunchPosition","LaunchAngle","LaunchSpeed","EntryPosition","EntryAngle","EntrySpeed","BreakPointLength","BoardsCrossed","EffectiveRPM","IsFullRack","HookPower","TimeUploaded","PositionsAsString","PindeckPositionsAsString","SpeedsAsString","AnglesAsString","HookSpotsAsString","MaxAngleDistance","HookBoard","RollBoard","SpeedLossTotal","SpeedLossHeads","SpeedLossMid","SpeedLossBack","SpeedLossPindeck","SpeedLossPercent","MaxSpeedLossDistance","BreakPointSpeed","BreakPointAngle","TotalHook","TrueBreakPointDistance","ImpactAngle","LastReadDistance","MeanSquaredError","MaxHookAngle","HookShape","ShotState","MaxAngle","DistanceTraveled","RPS","ShotTimeMS","RPMToSpeedRatio","BowlerName","Frame","ShotNumber","PinCount"]
data_raw.columns = column_names
data_raw.columns

#perustiedot datasta
data_raw.describe() 

#muuttujien tyypit
datatypes=data_raw.dtypes
print(datatypes)

#muuttujien yleiskuvaus
print(data_raw.describe())

#Näistä nähdään, että muuttujat CenterName, Country, State, City, Lane, Time, ID
#Guid, TimeUpLoaded, AsStringit, BowlerName, ShotNumber, Frame, ShotState, ovat kategorisia ja IsFullRack, IsRightHanded, ja Complete ovat boolean
#Myös PinCount on kategorinen, ja alustavasti y

#Poistetaan osa kategorisista, jotka jo tässä vaiheessa tiedetään turhiksi
dataset=data_raw
dataset.groupby("IsRightHanded").size()
#on vino, joten -> otetaan mukaan vain oikeakätiset
dataset = dataset[dataset.IsRightHanded == True]
dataset=dataset.drop("IsRightHanded", axis=1)

dataset=dataset.drop("PositionsAsString", axis=1)
dataset=dataset.drop("PindeckPositionsAsString", axis=1)
dataset=dataset.drop("SpeedsAsString", axis=1)
dataset=dataset.drop("AnglesAsString", axis=1)
dataset=dataset.drop("HookSpotsAsString", axis=1)
dataset=dataset.drop("BowlerName", axis=1)
dataset=dataset.drop("State", axis=1)
dataset=dataset.drop("City", axis=1)
dataset=dataset.drop("SpeedLossHeads", axis=1)
dataset=dataset.drop("SpeedLossMid", axis=1)
dataset=dataset.drop("SpeedLossPindeck", axis=1)
dataset=dataset.drop("SpeedLossBack", axis=1)
dataset=dataset.drop("Time", axis=1)
dataset=dataset.drop("Id", axis=1)
dataset=dataset.drop("Guid", axis=1)
dataset=dataset.drop("TimeUploaded", axis=1)
dataset=dataset.drop("Complete", axis=1)
dataset=dataset.drop("ShotState", axis=1)
dataset=dataset.drop("IsFullRack", axis=1)
dataset=dataset.drop("CenterName", axis=1)
dataset=dataset.drop("Frame", axis=1)
dataset=dataset.drop("Lane", axis=1)
dataset=dataset.drop("Country", axis=1)

#DistanceTraveled ja launchAxis on nolla-arvoja, poistetaan
dataset=dataset.drop("DistanceTraveled", axis=1)
dataset=dataset.drop("PindeckDeflection", axis=1)


dataset=dataset.drop("LaunchAxis", axis=1)
dataset=dataset.drop("PindeckExitBoard", axis=1)
dataset=dataset.drop("PindeckSpeed", axis=1)
dataset=dataset.drop("HeadsRPM", axis=1)
dataset=dataset.drop("MidRPM", axis=1)
dataset=dataset.drop("BackRPM", axis=1)

dataset=dataset.drop("MeanSquaredError", axis=1)
dataset=dataset.drop("HookShape", axis=1)
dataset=dataset.drop("RPS", axis=1)

dataset=dataset.drop("RPMToSpeedRatio", axis=1)
dataset=dataset.drop("MaxHookAngle", axis=1)
print(dataset.describe())

#tsekataan y:n/PinCountin jakautuminen eri kategorioihin
dataset.groupby("PinCount").size()
#nähdään, että jakauma aika tasaisesti eri määrille keiloja. Pidetään tämä, koska,
# jos tarkasteltaisiin vain kaatoja, jakauma olisi aika vino.


#tarkistetaan muuttujien välisiä korrelaatioita
set_option("display.width",100)
set_option("precision", 3)

correlations = dataset.corr(method="pearson")
print(correlations)

#tsekataan muuttujien jakauman vinous (skew) -> positiivinen luku kuvaa oikealle vinoutta
#-> negatiivinen vasemmalle ja lähellä nollaa oleva luku ei ole vinoutta
skewness= dataset.skew()
print(skewness)

#otetaan muuttujien visuaalinen kuvaus
#ensin univariate plots eli yhden muuttujan kuvia

#histogrammi
#dataset.hist(figsize=(20,20))
#plt.show()

#poistetaan vielä ShotTimeMS, EffectiveRPM, LastReadDistance, SpeedLossTotal, ReadStart
dataset=dataset.drop("ShotTimeMS", axis=1)
dataset=dataset.drop("EffectiveRPM", axis=1)
dataset=dataset.drop("LastReadDistance", axis=1)
dataset=dataset.drop("SpeedLossTotal", axis=1)
dataset=dataset.drop("ReadStart", axis=1)
#poistetaan ShotNumber 2 eli paikot -> poistetaan ShotNumber-muuttuja
dataset = dataset[dataset.ShotNumber !=2]
dataset=dataset.drop("ShotNumber", axis=1)

#tsekataan, montako SpeedLossPercentiä on >100
#dataset[dataset["SpeedLossPercent"]>100].count() 
speedlosspercent=dataset["SpeedLossPercent"]

over_100 = 0
for row in speedlosspercent:
    if row>100:
        over_100+=1
        print(row)
print(over_100)

#poistetaan ne rivit, joissa tuo prosentti on yli 100
dataset=dataset.drop(dataset[dataset.SpeedLossPercent >100].index)

#feature-selection ja correlaatiot on tiedostossa bowling_prosessointi_140320.py
X=dataset.drop("PinCount", axis=1)
y=dataset["PinCount"]

dataset_cols = dataset.columns.tolist()
X_cols = X.columns.tolist()

#tehdään muuttujien skaalaus, standardointi
scaler=StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_pd = pd.DataFrame(X_scaled)
X_scaled_pd.columns = X_cols

#tehdään minmaxscaling, jotta saadaan kaikista positiivinen 
minmax=MinMaxScaler(feature_range=(0,1))
X_minmax=minmax.fit_transform(X_scaled)

#seuraavaksi valitaan lopullisen datasetin muuttujat (4-5) ja tehdään multikategorinen classification
#luodaan ensin yleinen dataset, jossa neljä eri metodien tärkeäksi osoittamaa muuttujaa (maxhookdistance, entryangle, breakpointlength, boardscrossed)

#nimeä sarakkeet
X_gen4 = X_scaled_pd[["MaxHookDistance", "EntryAngle", "BreakPointLength", "BoardsCrossed"]]
X_gen4.shape
y.shape
#jaetaan train-test setteihin

X_train_g4, X_test_g4, y_train, y_test = train_test_split(X_gen4, y, test_size=0.2, random_state=0)

log_regr_model = LogisticRegression(solver="liblinear")
log_regr_model.fit(X_train_g4, y_train)
result_lr_g4 = log_regr_model.score(X_test_g4,y_test)
print("Accuracy:", result_lr_g4)
predicted_lr_g4 = log_regr_model.predict(X_test_g4)
matrix_lr_g4 = confusion_matrix(y_test, predicted_lr_g4)
print(matrix_lr_g4)
#Huom: päätöspuut pystyvät käsittelemään multikollineaarisuutta datasetissä (eli jos joku predictive variable voidaan ennustaa
#muiden pohjalta hyvin tarkasti)

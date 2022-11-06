import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor
def RMSE(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
from sklearn import metrics
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns;
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
#from utils import *
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from sklearn import linear_model
import xgboost 

# Import other files

from graph import*
from model import*
from measrmnt_indices import* 
from Feature_selection import*

dataFr = pd.read_csv("H:/Glucose/Preprocessed File/preprocessed_PPG-34.csv")
dataFr.drop(dataFr.columns[[0, 1, 39, 40]], axis=1, inplace=True) 
# dataFr.drop(dataFr.columns[[0, 1, 38, 40]], axis=1, inplace=True) #GL
dataFr.head()
dataFr.shape
df = dataFr


dataFr = dataFr.drop(labels=[26, 67], axis=0)


# print(dataFr[dataFr['Sex(M/F)']==1]['Sex(M/F)'].count())

Xorg = dataFr.to_numpy()
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
Xmeans = scaler.mean_
Xstds = scaler.scale_
y = Xscaled[:, 36]
X = Xscaled[:, 0:36]



#RFE selection
columns_names = dataFr.columns.values
rfe = RFE_Selection(X, y, RFR()) 
print("Optimal number of features : %d" % rfe.n_features_)
selected_columns_names_rfe, get_best_ind_rfe = selected_index_RFE(dataFr, rfe.ranking_, 1, 36)
X_rfe= X[:, get_best_ind_rfe]
print(X_rfe.shape)
# df.drop(dataFr.columns[36], axis=1, inplace=True)
# rfe.support_rfecv_df = pd.DataFrame(rfe.ranking_,index=df.columns,columns=['Rank']).sort_values(by='Rank',ascending=True)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(6,3))
# fig.set_facecolor("white")
# ax.set_xlabel("Number of features selected")
# ax.set_ylabel("Cross validation score($R^2$)")
# plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
# plt.show()


#RReliefF Feature selection
relief = ReliefF_Selection(X, y) 
print(relief.feature_importances_)
# print(relief.top_features_) #index of the features sorted according to feature importance
# for feature_name, feature_score in zip(df.columns, relief.feature_importances_):
#     print(feature_name, '\t', feature_score)

selected_columns_names_relief, get_best_ind_relief = selected_feature_and_index(dataFr, relief.feature_importances_, 0.02, 36)
X_relief = X[:, get_best_ind_relief]
print(X_relief.shape)





#Random Forest Feature selection
rfr = RFR()
rfr.fit(X, y)
print(rfr.feature_importances_)

selected_columns_names_rfr, get_best_ind_rfr = selected_feature_and_index(dataFr, rfr.feature_importances_, 0.01, 36)
X_rfr = X[:, get_best_ind_rfr]
print(X_rfr.shape)


# #Forward Base Feature Selection 

# from sklearn.feature_selection import SequentialFeatureSelector
# sfs = SequentialFeatureSelector( RFR(), cv=10)
# sfs.fit(X, y)
# print(sfs.get_support())
# print(sfs.ranking_)



#Pearson Correlation Base
# Correlation with output variable
cor_target =Corr(dataFr)
selected_columns_names_cor, get_best_ind_cor = selected_feature_and_index(dataFr, cor_target, 0.1, 36)
X_cor = X[:, get_best_ind_rfr]
print(X_cor.shape)


#Select the features that selected by four FSM
selected_columns_names_final, get_best_ind_final = select_three_or_four_FSM(dataFr, 36, get_best_ind_rfr,\
                                        get_best_ind_relief, get_best_ind_rfr, get_best_ind_cor)

X_final = X[:, get_best_ind_final]


# For MIC Feature Selection 
get_best_ind_final_Gl = [0, 1, 2, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22, 26, 27, 30, 32, 35] 
len(get_best_ind_final_Gl)
X_final_Gl = X[:, get_best_ind_final_Gl]

'''
21   GL
['Time of Max. Slope(t_ms)' 'Prev. point a_ip(d)' 'Time of a_ip(t_ip)'
 'Diastolic_peak(y)' 'Systolic_peak_time(t1)' 'Diastolic_peak_time(t2)'
 'Dicrotic_notch_time(t3)' 'w' 'Inflection_point_area_ratio(A2/A1)' 'b1'
 'e1' 'l1' 'b2' 'e2' 'ta1' 'ta2' 'tb2'
 'Fundamental_component_magnitude(|sbase|)'
 '2nd_harmonic_magnitude(|s2nd|)' '3rd_harmonic_magnitude(|s3rd|)'
 'Stress-induced_vascular_response_index(sVRI)']
[0.5584697798865652, 0.601999167664786, 0.5278523171601588,
 0.5251085233884004, 0.5584697798865652, 0.5197479592518415,
 0.5278523171601588, 0.536681817899924, 0.5075238316744818,
 0.5537667809087452, 0.5039738557461904, 0.6129789044207969,
 0.5915665501246591, 0.5781910107362448, 0.5153639087397159, 
 0.5000941976759652, 0.5087691156484294, 0.6336010836061294,
 0.5218812664541789, 0.5025694792396849, 0.5430029162361002]
'''

get_best_ind_final_Hb = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 18, 19, 20, 21, 22, 24, 26, 27, 30, 32, 35] 
X_final_Hb = X[:, get_best_ind_final_Hb]


'''
22 Hb
['Age' 'Sex(M/F)' 'Systolic_peak(x)' 'Max. Slope(c)'
 'Time of Max. Slope(t_ms)' 'Prev. point a_ip(d)' 'Diastolic_peak(y)'
 'Dicrotic_notch(z)' 'Systolic_peak_time(t1)' 'Diastolic_peak_time(t2)'
 'Inflection_point_area_ratio(A2/A1)' 'l1' 'a2' 'b2' 'e2' 'ta1' 'te1'
 'ta2' 'tb2' 'Fundamental_component_magnitude(|sbase|)'
 '2nd_harmonic_magnitude(|s2nd|)'
 'Stress-induced_vascular_response_index(sVRI)']
[0.5782913312984014, 0.6479885787538197, 0.565205471092822, 
 0.565205471092822, 0.5270508232478199, 0.5864529681846651, 
 0.5063551064073172, 0.517021263400945, 0.5270508232478199,
 0.5340094872292461, 0.5248759215895156, 0.528910657725647,
 0.5538176161962306, 0.5479143197080275, 0.5723241815522768,
 0.5301059419932976, 0.5573336834803955, 0.5994718506688926, 
 0.5023750659226986, 0.6022088376734687, 0.5633940320776447, 0.5232032464024526]

'''

import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers 
import random
import pandas as pd
import numpy as np



X = X[:, get_best_ind_final_Hb]

model = DNN(X)
model.summary()

keras_model_path = 'H:/Glucose/Preprocessed File/GL Model'

n_splits = 10
kf = 0
score = []
std_r2 = []
std_mae = []
cv_set = np.repeat(-1.,X.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)
for train_index,test_index in skf.split(X, y):
    x_train,x_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
    # model.fit(x_train,y_train)

    # --------------------------------------
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5) 

    model.fit(x_train,y_train, epochs=100, batch_size=32,
                        shuffle=True,  callbacks=[callback],  validation_split=0, verbose=1)
    # model.save("H:/Glucose/Preprocessed File/GL Model", 'DNN_model' + str(kf) + '.h5')  
    # model.save(keras_model_path) #To Save in Saved_model format
    # model.save('DNN_model_Gl' + str(kf) + '.h5') #To save model in H5 or HDF5 format   
    predicted_y =  model.predict(x_test)
    LOG_INFO(f"Individual R = {pearsonr(y_test, predicted_y)}", mcolor="green")
    print("R: " + str(pearsonr(y_test, predicted_y)))
    cv_set[test_index] = predicted_y[:, 0]
    kf+=1




# import tensorflow as tf
# h5_model = tf.keras.models.load_model("DNN_model_Gl0.h5") # loading model in h5 format
# h5_model.summary()
# saved_m = tf.keras.models.load_model("saved_model/my_model") #loading model in saved_model format
# saved_m.summary()


# print("R^2 (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(R_2),np.std(R_2)))    
# print("MAE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mae),np.std(mae)))    
# print("MSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mse),np.std(mse)))   
# print("RMSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.sqrt(np.mean(mse)),np.sqrt(np.std(mse))))
# print("MAPE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mape),np.std(mape)))

## For Get real values of target label
yy = (y * Xstds[36]) + Xmeans[36]
cv_sety = (cv_set * Xstds[36]) + Xmeans[36]

   ### =============== Measure all indices ================================
LOG_INFO(f"====> Overall R   = {pearsonr(yy,cv_sety)}", mcolor="red")
LOG_INFO(f"====> R^2 Score   = {metrics.r2_score(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MAE         = {metrics.mean_absolute_error(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MSE         = {metrics.mean_squared_error(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> RMSE        = {RMSE(yy, cv_sety)}", mcolor="red")
LOG_INFO(f"====> MAPE        = {mean_absolute_percentage_error(yy, cv_sety)}", mcolor="red")


R_2 = metrics.r2_score(yy, cv_sety)
mae = metrics.mean_absolute_error(yy, cv_sety)

bland_altman_plot_paper(yy, cv_sety, "S_F_Hb_B")
act_pred_plot_paper(yy, cv_sety,R_2,mae,"S_F_Hb_R2")


# saved_m = tf.keras.models.load_model(keras_model_path) #loading model in saved_model format
# saved_m.summary()

# X_in_select = X[:1,]
# pred = saved_m.predict(X_in_select) ### Prediction of Hb Level
# pred = (pred * Xstds[36]) + Xmeans[36]
# print("Predicted Hb (mmol/L): " + str(pred))




# import tensorflow as tf
# h5_model = tf.keras.models.load_model("DNN_model_Hb.h5") # loading model in h5 format
# h5_model.summary()
# X_in_select = X[:1,]
# pred = h5_model.predict(X_in_select) ### Prediction of Hb Level
# pred = (pred * Xstds[36]) + Xmeans[36]
# print("Predicted Hb (mmol/L): " + str(pred))


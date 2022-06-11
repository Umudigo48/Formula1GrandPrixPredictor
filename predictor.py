from pickle import load
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

driver_dict = pickle.load(open('driver_dict','rb'))
constructor_dict = pickle.load(open('consructor_dict','rb'))
clf = pickle.load(open('RandomForestClassifier.pkl','rb'))
data = pd.read_csv('cleaned_data.csv')
y_dict = {1:'Podyumda bitirecektir (1-3) ',
          2:'Puan alacaktır. (4-10)',
          3:'Puan alamayacaktır. (11-20 veya DNF)'
               
        }

le_d = LabelEncoder()
le_d.fit(data[['driver']])
le_c = LabelEncoder()
le_c.fit(data[['constructor']])
le_gp = LabelEncoder()
le_gp.fit(data[['GP_name']])

def pred(driver,constructor,quali,circuit):
    gp = le_gp.fit_transform([circuit]).max()
    quali_pos = quali
    constructor_enc = le_c.transform([constructor]).max()
    driver_enc = le_d.transform([driver]).max()
    driver_confidence = driver_dict[driver].max()
    constructor_relaiablity = constructor_dict[constructor].max()
    prediction = clf.predict([[gp,quali_pos,constructor_enc,driver_enc,driver_confidence,constructor_relaiablity]]).max()

    return y_dict[prediction]


print(pred('Charles Leclerc', 'Haas F1 Team', '14', 'Silverstone Circuit'))
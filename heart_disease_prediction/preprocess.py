import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(df):
    # Encode categorical columns manually
    le_dict = {
        'Sex': {'M':1, 'F':0},
        'ChestPainType': {'TA':0, 'ATA':1, 'NAP':2, 'ASY':3},
        'RestingECG': {'Normal':0, 'ST':1, 'LVH':2},
        'ExerciseAngina': {'Y':1, 'N':0},
        'ST_Slope': {'Up':0, 'Flat':1, 'Down':2}
    }
    for col, mapping in le_dict.items():
        df[col] = df[col].map(mapping)
    return df

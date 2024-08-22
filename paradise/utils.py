import pandas as pd
import os

def drop_nan_df(input_dir = 'data_1/labels/paradise_csi', output_dir = 'data_1/labels/paradise_csi_drop_nan'):

    df = pd.read_csv(f'{input_dir}.csv')
    df = pd.DataFrame(data=df, columns=['number','id_number', 'csi_total','csi', 'right_sup', 
                                                            'left_sup','right_mid',
                                                            'left_mid','right_mid',])
    
    missing_values = df.isnull().sum()
    # print(missing_values[missing_values > 0])  # Columns with missing values
    df_cleaned = df.dropna()
    
    df_cleaned.to_csv(f'{output_dir}.csv')
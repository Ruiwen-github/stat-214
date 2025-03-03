import numpy as np
import pandas as pd


def clean_data(data=pd.read_csv('TBI PUD 10-08-2013.csv'), 
               row_drop=0.8, col_drop=0.8, 
               indicator_upper=0.8, indicator_lower=0.1, 
               feature_drop=[],
               indicator_feature=[],
               impute_feature=[]):
    """
    Data cleaning function. This function and removes columns and rows with missing values above the certain threshold.
    Remove features we deem irrelevant or redundant. Generate missing indicator for certain columns with missing values.
    Impute missing values with the mean value (for numeric features) or mode category (for categorical features).

    Parameters
    ----------
    data : pandas DataFrame
        The dataset to be cleaned.
    row_drop : float
        The lower bound for the missing indicator. If the percentage of missing values in a column is higher than this threshold,
        the feature will be droped.
    col_drop : float
        The lower bound for the missing indicator. If the percentage of missing values in a column is higher than this threshold,
        the feature will be droped.
    indicator_upper : float
        The upper bound for the missing indicator. If the percentage of missing values in a row is lower than this threshold,
        the row will be deleted.
    indicator_lower : float
        The lower bound for the missing indicator. If the percentage of missing values in a row is higher than this threshold,
        the row will be deleted.
    feature_drop : list
        The given list of features to be deleted.
    feature_indicator: list
        The given list of features to generate missing indicator.
    feature_impute : list
        The given list of features to be imputed.

    Returns
    -------
    pandas DataFrame
        The cleaned data.
    """

    ### Step 0: Assert ###############################################################
    
    assert col_drop >= indicator_upper, "col_drop must be higher than indicator_upper"
    assert indicator_upper >= indicator_lower, "indicator_upper must be higher than indicator_lower"

    ## if the feature have been dropped, indicator_feature and impute_feature should not include them
    assert set(feature_drop).isdisjoint(set(indicator_feature)), "feature_drop and indicator_feature could not exist same features"
    assert set(feature_drop).isdisjoint(set(impute_feature)), "feature_drop and impute_feature could not exist same features"

    ## if the given feature have genertated indicator feature, no need to impute missing values, vise versa
    assert set(indicator_feature).isdisjoint(set(impute_feature)), "indicator_feature and impute_feature could not exist same features"



    ### Step 1: Filter rows with 14<=GCS<=15 #######################################

    # According to the domain knowledge, we only keep the rows with 14<=GCS<=15
    data=data.loc[(data['GCSTotal'] >= 14) & (data['GCSTotal'] <= 15)]



    ### Step 2: process inconsistent/missing variable value, 92 = 'Not applicable' ############

    # 92 = 'Not applicable' --depend on situation, we can convert to NaN or remain 92
    # If the corresponding main variable is 0, keep 92, otherwise replace with NaN

    # Main Variable, 'LOCSeparate'
    mask = (data['LocLen'] == 92) & (data['LOCSeparate'] != 0)
    data.loc[mask, 'LocLen'] = np.nan

    # Main Variable, 'Seiz'
    for col in ['SeizOccur', 'SeizLen']:
        mask = (data[col] == 92) & (data['Seiz'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'HA_verb'
    for col in ['HASeverity', 'HAStart']:
        mask = (data[col] == 92) & (data['HA_verb'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable, 'Vomit'
    for col in ['VomitNbr', 'VomitStart', 'VomitLast']:
        mask = (data[col] == 92) & (data['Vomit'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable, 'AMS'
    for col in ['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']:
        mask = (data[col] == 92) & (data['AMS'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'SFxPalp'==0 or 2
    mask = (data['SFxPalpDepress'] == 92) & (~data['SFxPalp'].isin([0, 2]))
    data.loc[mask, 'SFxPalpDepress'] = np.nan

    # Main Variable 'SFxBaws'
    for col in ['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi']:
        mask = (data[col] == 92) & (data['SFxBas'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'Hema'
    for col in ['HemaLoc', 'HemaSize']:
        mask = (data[col] == 92) & (data['Hema'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'Clav'
    for col in ['ClavFace', 'ClavNeck', 'ClavFro', 'ClavOcc', 'ClavPar', 'ClavTem']:
        mask = (data[col] == 92) & (data['Clav'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'NeuroD'
    for col in ['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']:
        mask = (data[col] == 92) & (data['NeuroD'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'OSI'
    for col in ['OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank', 'OSIAbdomen', 'OSIPelvis', 'OSIOth']:
        mask = (data[col] == 92) & (data['OSI'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'CTForm1'
    columns_ctform1 = [
        'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC',
        'IndMech', 'IndNeuroD', 'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma',
        'IndSeiz', 'IndVomit', 'IndXraySFx', 'IndOth',
        'CTSed', 'CTSedAgitate', 'CTSedAge', 'CTSedRqst', 'CTSedOth'
    ]
    for col in columns_ctform1:
        mask = (data[col] == 92) & (data['CTForm1'] != 0)
        data.loc[mask, col] = np.nan

    # Main Variable 'CTDone', for 'EDCT', 'PosCT', Finding1-14, Finding20-23
    columns_ctdone = ['EDCT', 'PosCT'] + [f'Finding{i}' for i in list(range(1, 15)) + list(range(20, 24))]
    for col in columns_ctdone:
        mask = (data[col] == 92) & (data['CTDone'] != 0)
        data.loc[mask, col] = np.nan



    ### Step 3: process inconsistent/missing rows, Main variable = 0 ############

    # When the main variable is 0, the ideal value of the corresponding subvariables should all be 92.
    # For each row with this case, we convert all the NA subvariables in this group to 92.
    # If there exists a subvariable other than 92 and NA, we will delete this row.


    # For groups, main variable is the key, subvariables is the list
    groups = {
        'LOCSeparate': ['LocLen'],
        'Seiz': ['SeizOccur', 'SeizLen'],
        'HA_verb': ['HASeverity', 'HAStart'],
        'Vomit': ['VomitNbr', 'VomitStart', 'VomitLast'],
        'AMS': ['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth'],
        'SFxPalp': ['SFxPalpDepress'],
        'SFxBaws': ['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi'],
        'Hema': ['HemaLoc', 'HemaSize'],
        'Clav': ['ClavFace', 'ClavNeck', 'ClavFro', 'ClavOcc', 'ClavPar', 'ClavTem'],
        'NeuroD': ['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth'],
        'OSI': ['OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank', 'OSIAbdomen', 'OSIPelvis', 'OSIOth'],
        'CTForm1': [
            'IndAge', 'IndAmnesia', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC',
            'IndMech', 'IndNeuroD', 'IndRqstMD', 'IndRqstParent', 'IndRqstTrauma',
            'IndSeiz', 'IndVomit', 'IndXraySFx', 'IndOth',
            'CTSed', 'CTSedAgitate', 'CTSedAge', 'CTSedRqst', 'CTSedOth'
        ],
        'CTDone': ['EDCT', 'PosCT'] + [f'Finding{i}' for i in list(range(1, 15)) + list(range(20, 24))]
    }

    rows_to_drop = set()
    modified_rows = set()

    for main_var, sub_vars in groups.items():
        if main_var not in data.columns:
            continue

        # Select key values is 0
        mask_main0 = data[main_var] == 0
        if mask_main0.sum() == 0:
            continue

        # select the inconsistent rows (variables with sub_vars rather than 92 and NaN): delete
        sub_data = data.loc[mask_main0, sub_vars]
        invalid_mask = sub_data.notna() & (sub_data != 92)
        inconsistent = invalid_mask.any(axis=1)

        drop_idxs = inconsistent.index[inconsistent].tolist()
        rows_to_drop.update(drop_idxs)

        # Consistent rows: fillna with 92
        consistent_idxs = inconsistent.index[~inconsistent]
        if not consistent_idxs.empty:
            data.loc[consistent_idxs, sub_vars] = sub_data.loc[~inconsistent].fillna(92)
            modified_rows.update(consistent_idxs.tolist())

    data.drop(index=list(rows_to_drop), inplace=True)



    ### Step 4: process special data, 90 and 91 #################################

    # 90 = 'Other' --We can convert the all the NaN to 90 'Other'
    # 91 = 'Pre-verbal/Non-verbal' -- not clear, so they can be considered as NA
    
    for col in ['Certification', 'InjuryMech', 'Race', 'EDDisposition']:
        data[col].fillna(90, inplace=True)

    for col in ['Amnesia_verb', 'HA_verb']:
        data.loc[data[col] == 91, col] = np.nan



    ### Step 5: Remove features with high missing values percentage ###############################

    miss_percentage = data.isnull().mean()
    col_high_missing = miss_percentage[miss_percentage > col_drop].index
    data.drop(columns= col_high_missing,  inplace=True)



    ### Step 6: Remove given list of features  ####################################################

    data.drop(columns=feature_drop, inplace=True)

    # Delete the features that contain repulicate information (to avoid multicollinearity)
    # GCSTotal and GCSGroup
    if 'GCSTotal' in data.columns and 'GCSGroup' in data.columns:
        data.drop(columns=['GCSGroup'], inplace=True)

    # AgeInMonth, AgeinYears and AgeTwoPlus
    if all(col in data.columns for col in ['AgeInMonth', 'AgeinYears', 'AgeTwoPlus']):
        data.drop(columns=['AgeinYears', 'AgeTwoPlus'], inplace=True)


    ### Step 7: Remove duplicate rows and rows with high missing percentage ########################

    data.drop_duplicates()

    miss_per_row = data.isnull().mean(axis=1)
    row_high_missing = miss_per_row[miss_per_row > row_drop].index
    data.drop(index=row_high_missing, inplace=True)



    ### Step 8: Generate missing indicator for certain columns ####################################

    columns_in_range = miss_percentage[(miss_percentage >= indicator_lower) & (miss_percentage <= indicator_upper)].index

    # generate the missing indicator (mi) for each column in the range
    for col in columns_in_range:
        data[f"{col}_mi"] = data[col].isna().astype(int)



    ### Step 9: Generate missing indicator for specified columns ####################################

    for col in indicator_feature:
        if col in data.columns:
            data[f"{col}_mi"] = data[col].isna().astype(int)



    ### Step 10: Impute missing values for specified columns ########################################

    for col in impute_feature:
        if col in data.columns: 
            if data[col].dtype == 'O': ## mode for categorical columns
                mode_value = data[col].mode()[0]  
                data[col].fillna(mode_value, inplace=True)
            else:  ## mean for numeric columns
                mean_value = data[col].mean() 
                data[col].fillna(mean_value, inplace=True)


    print(data.info())
    data.to_csv('cleaned_data.csv', index=False)

    return data
    

clean=clean_data() 
             
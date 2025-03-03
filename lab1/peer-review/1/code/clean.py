import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Function to combine multiple data(Yes/No and detailed data) based on domain knowledge
def combine_category(origin, detail, max_value):
    """
    Combines categorical data based on domain knowledge.

    Args:
        origin (int): origin data
        detail (int): Detailed data for the condition
        max_value (int): Maximum value in the detail data

    Returns:
            - np.nan if origin is missing
            - 0 if origin is 0
            - detail value if origin is 1 and detail exists
            - max_value + 1 if origin is 1 but detail is missing(Considering worst case)
    """
    if pd.isna(origin):
        return np.nan
    elif origin == 0:
        return 0
    elif origin == 1:
        if pd.isna(detail):
            return max_value + 1
        elif detail <= max_value:
            return detail


def apply_combine_category(df, Col_origin, Col_detail_list):
    """
    Applies the combine_category function to multiple detail columns.

    Args:
        df (pd.DataFrame): data
        Col_origin (str): origin column name
        Col_detail_list (list): detail column names

    Returns:
        pd.DataFrame: Processed dataframe without origin data
    """
    for Col_detail in Col_detail_list:
        max_value = df[Col_detail].max()
        df[Col_detail] = df.apply(
            lambda row: combine_category(row[Col_origin], row[Col_detail], max_value),
            axis=1,
        )
    df = df.drop(Col_origin, axis=1)
    return df


def clean_data(file_path):
    """
    Cleans the ciTBI dataset with domain knowledge.

    Args:
        file_path (str): Path

    Returns:
            - preprocessed dataset
            - Training dataset
            - Test dataset
    """
    df = pd.read_csv(file_path, na_values=[""], index_col=0)
    df = df.astype("Int64")
    df = df.replace({pd.NA: np.nan})  # Deal with missing values

    # "EmplType", "Certification" do not affect ciTBI, so I excludes the data.
    df = df.drop(["EmplType", "Certification"], axis=1)

    # Dealing with "InjuryMech"
    # Others include missing values
    df.loc[df["InjuryMech"].isna() | (df["InjuryMech"] == 90), "InjuryMech"] = 13

    # Dealing with "Amnesia_verb"
    # change 91 to np.nan
    df.loc[(df["Amnesia_verb"] == 91), "Amnesia_verb"] = np.nan

    # Dealing with "LOCSeparate"
    df["LOCSeparate"] = df["LOCSeparate"].replace({2: 1})  # When suspected, deal as Yes
    df["LocLen"] = df["LocLen"].replace({92: np.nan})  # When missing, deal as Na
    df = apply_combine_category(df, "LOCSeparate", ["LocLen"])

    # Dealing with "Seiz"
    df["SeizOccur"] = df["SeizOccur"].replace({92: np.nan})  # When missing, deal as Na
    df["SeizLen"] = df["SeizLen"].replace({92: np.nan})  # When missing, deal as Na
    df = apply_combine_category(df, "Seiz", ["SeizOccur", "SeizLen"])

    # Dealing with "HA_verb"
    # If the child cannot speak, the data is treated as missing.
    # Because whether a child can or cannot speak is a matter of growth and has nothing to do with ciTBI.
    df["HA_verb"] = df["HA_verb"].replace({91: np.nan})
    df["HASeverity"] = df["HASeverity"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["HAStart"] = df["HAStart"].replace({92: np.nan})  # When missing, deal as Na
    df = apply_combine_category(df, "HA_verb", ["HASeverity", "HAStart"])

    # Dealing with "Vomit"
    df["VomitNbr"] = df["VomitNbr"].replace({92: np.nan})  # When missing, deal as Na
    df["VomitStart"] = df["VomitStart"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["VomitLast"] = df["VomitLast"].replace({92: np.nan})  # When missing, deal as Na
    df = apply_combine_category(df, "Vomit", ["VomitNbr", "VomitStart", "VomitLast"])

    # GCSTotal does not have any missing values, so the categorised GCSGroup is not used.
    df = df.drop("GCSGroup", axis=1)

    # Dealing with "AMS"
    df["AMSAgitated"] = df["AMSAgitated"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["AMSSleep"] = df["AMSSleep"].replace({92: np.nan})  # When missing, deal as Na
    df["AMSSlow"] = df["AMSSlow"].replace({92: np.nan})  # When missing, deal as Na
    df["AMSRepeat"] = df["AMSRepeat"].replace({92: np.nan})  # When missing, deal as Na
    df["AMSOth"] = df["AMSOth"].replace({92: np.nan})  # When missing, deal as Na

    # Dealing with "SFxPalp"
    # many of detailed data are 0/1, so we can just add 1 to them
    df["SFxPalp"] = df["SFxPalp"].replace({2: 1})  # When unclear, deal as Yes
    df["SFxPalpDepress"] = df["SFxPalpDepress"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["SFxPalpDepress"] = df["SFxPalpDepress"] + 1
    df = apply_combine_category(df, "SFxPalp", ["SFxPalpDepress"])

    # Dealing with "SFxBas"
    # many of detailed data are 0/1, so we can just add 1 to them
    df["SFxBasHem"] = df["SFxBasHem"].replace({92: np.nan})  # When missing, deal as Na
    df["SFxBasOto"] = df["SFxBasOto"].replace({92: np.nan})  # When missing, deal as Na
    df["SFxBasPer"] = df["SFxBasPer"].replace({92: np.nan})  # When missing, deal as Na
    df["SFxBasRet"] = df["SFxBasRet"].replace({92: np.nan})  # When missing, deal as Na
    df["SFxBasRhi"] = df["SFxBasRhi"].replace({92: np.nan})  # When missing, deal as Na
    for col in ["SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi"]:
        df[col] = df[col] + 1
    df = apply_combine_category(
        df, "SFxBas", ["SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi"]
    )

    # Dealing with "Hema"
    df["HemaLoc"] = df["HemaLoc"].replace({92: np.nan})  # When missing, deal as Na
    df["HemaSize"] = df["HemaSize"].replace({92: np.nan})  # When missing, deal as Na
    df = apply_combine_category(df, "Hema", ["HemaLoc", "HemaSize"])

    # Dealing with "Clav"
    # Even if Clav is Yes, there are cases where all the details are No, so I create ClavOth.
    # Also, because the information is the same, I exclude Clav.
    df["ClavFace"] = df["ClavFace"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavNeck"] = df["ClavNeck"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavFro"] = df["ClavFro"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavOcc"] = df["ClavOcc"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavPar"] = df["ClavPar"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavTem"] = df["ClavTem"].replace({92: np.nan})  # When missing, deal as Na
    df["ClavOth"] = df["Clav"] - df[
        ["ClavFace", "ClavNeck", "ClavFro", "ClavOcc", "ClavPar", "ClavTem"]
    ].fillna(0).sum(axis=1)
    df["ClavOth"] = df["ClavOth"].apply(
        lambda x: 1 if x > 0 else (np.nan if pd.isna(x) else 0)
    )
    df = df.drop("Clav", axis=1)

    # Dealing with "NeuroD"
    # Even if NeuroD is Yes, there are cases where all the details are No, so I include this case as NeuroDOth.
    # Also, because the information is the same, I exclude NeuroD.
    df["NeuroDMotor"] = df["NeuroDMotor"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["NeuroDSensory"] = df["NeuroDSensory"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["NeuroDCranial"] = df["NeuroDCranial"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["NeuroDReflex"] = df["NeuroDReflex"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    temp_neurod_new_oth = df["NeuroD"].fillna(0) - df[
        ["NeuroDMotor", "NeuroDSensory", "NeuroDCranial", "NeuroDReflex"]
    ].fillna(0).sum(axis=1)
    temp_neurod_oth = pd.concat(
        [temp_neurod_new_oth, df["NeuroDOth"].fillna(0)], axis=1
    ).max(axis=1)
    df["NeuroDOth"] = df.apply(
        lambda row: (
            1
            if temp_neurod_oth[row.name] > 0
            else (0 if pd.notna(row["NeuroD"]) else pd.NA)
        ),
        axis=1,
    )
    df = df.drop("NeuroD", axis=1)

    # Dealing with "OSI"
    # Even if OSI is Yes, there are cases where all the details are No, so I include this case as OSIOth.
    # Also, because the information is the same, I exclude OSI.
    df["OSIExtremity"] = df["OSIExtremity"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["OSICut"] = df["OSICut"].replace({92: np.nan})  # When missing, deal as Na
    df["OSICspine"] = df["OSICspine"].replace({92: np.nan})  # When missing, deal as Na
    df["OSIFlank"] = df["OSIFlank"].replace({92: np.nan})  # When missing, deal as Na
    df["OSIAbdomen"] = df["OSIAbdomen"].replace(
        {92: np.nan}
    )  # When missing, deal as Na
    df["OSIPelvis"] = df["OSIPelvis"].replace({92: np.nan})  # When missing, deal as Na
    temp_neurod_new_oth = df["OSI"].fillna(0) - df[
        ["OSIExtremity", "OSICut", "OSICspine", "OSIFlank", "OSIAbdomen", "OSIPelvis"]
    ].fillna(0).sum(axis=1)
    temp_neurod_oth = pd.concat(
        [temp_neurod_new_oth, df["OSIOth"].fillna(0)], axis=1
    ).max(axis=1)
    df["OSIOth"] = df.apply(
        lambda row: (
            1
            if temp_neurod_oth[row.name] > 0
            else (0 if pd.notna(row["OSI"]) else pd.NA)
        ),
        axis=1,
    )
    df = df.drop("OSI", axis=1)

    # The aim is to prevent the use of CT, so CT-related data cannot be used.
    df = df.drop(
        [
            "CTForm1",
            "IndAge",
            "IndAmnesia",
            "IndAMS",
            "IndClinSFx",
            "IndHA",
            "IndHema",
            "IndLOC",
            "IndMech",
            "IndNeuroD",
            "IndRqstMD",
            "IndRqstParent",
            "IndRqstTrauma",
            "IndSeiz",
            "IndVomit",
            "IndXraySFx",
            "IndOth",
            "CTSed",
            "CTSedAgitate",
            "CTSedAge",
            "CTSedRqst",
            "CTSedOth",
        ],
        axis=1,
    )

    # Delete duplicate age data (as there are no missing values)
    df = df.drop(
        [
            "AgeinYears",
            "AgeTwoPlus",
        ],
        axis=1,
    )

    # Dealing with "Race"
    df["Race"] = df["Race"].replace({90: 6})  # Convert 90 to 6

    # The aim is to prevent the use of CT, so CT-related data cannot be used.
    df = df.drop(
        [
            "Observed",
            "EDDisposition",
            "CTDone",
            "EDCT",
            "PosCT",
            "Finding1",
            "Finding2",
            "Finding3",
            "Finding4",
            "Finding5",
            "Finding6",
            "Finding7",
            "Finding8",
            "Finding9",
            "Finding10",
            "Finding11",
            "Finding12",
            "Finding13",
            "Finding14",
            "Finding20",
            "Finding21",
            "Finding22",
            "Finding23",
        ],
        axis=1,
    )

    # Not related to ciTBI, so exclude.
    df = df.drop("HospHead", axis=1)

    # Fullfill PosIntFinal when PosIntFinal is NaN
    # When PosIntFinal is NaN and any other target col is NaN, set PosIntFinal to NaN
    target_cols = [
        "DeathTBI",
        "HospHeadPosCT",
        "Intub24Head",
        "Neurosurgery",
    ]
    df["PosIntFinal"] = df[target_cols].eq(1).any(axis=1).astype("Int64")
    all_zero = df[target_cols].eq(0).all(axis=1)
    df.loc[all_zero, "PosIntFinal"] = 0
    df = df.drop(target_cols, axis=1)

    # Convert all columns to Int64
    df = df.astype("Int64")

    # Save the processed data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=777)

    return df, train_df, test_df

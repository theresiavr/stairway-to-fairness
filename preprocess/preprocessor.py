import pandas as pd
import numpy as np

import os
import csv

import re


def load_data(path, data_name):
    df_inter = pd.read_table(f"{path}/{data_name}.inter")
    print("Finish loading interaction")

    df_user = pd.read_table(f"{path}/{data_name}.user")
    print("Finish loading user data")
    df_item = pd.read_table(f"{path}/{data_name}.item")
    print("Finish loading item data")
    
    return df_inter, df_user, df_item

def preprocess_column(df):
    #ML-1M and LFM
    df.columns = df.columns\
                            .str.replace(":token_seq","")\
                            .str.replace(":token","")\
                            .str.replace(":float","")
    
    #LFM
    df = df.rename(columns={"artists_id":"item_id"})

    #JobRec
    df = df.rename(columns={"UserID":"user_id", "JobID":"item_id", "ApplicationDate":"timestamp"})

    return df

def core_filtering(df, core_filter):
    if core_filter > 0:
        user_val_count = df.user_id.value_counts()
        item_val_count = df.item_id.value_counts()

        user_min_count = user_val_count.min()
        item_min_count = item_val_count.min()

        user_id_gt_core_filter = user_val_count[user_val_count >= core_filter].index
        item_id_gt_core_filter = item_val_count[item_val_count >= core_filter].index

        while user_min_count < core_filter or item_min_count < core_filter:
            df = df[df.user_id.isin(user_id_gt_core_filter)]
            df = df[df.item_id.isin(item_id_gt_core_filter)]
            
            user_val_count = df.user_id.value_counts()
            item_val_count = df.item_id.value_counts()

            user_min_count = user_val_count.min()
            item_min_count = item_val_count.min()
            
            user_id_gt_core_filter = user_val_count[user_val_count >= core_filter].index
            item_id_gt_core_filter = item_val_count[item_val_count >= core_filter].index
    
        assert user_min_count >= core_filter
        assert item_min_count >= core_filter 

    return df

def filter_data(data, df_inter, df_user, df_item, train_prop=0.6, test_prop=0.2, min_inter=5, rating_thresh=3, core_filter=5):
    # train/val/test: 6:2:2
    if "ml-1m" in data:
        df_user = df_user.drop(columns="zip_code")
        df_user = df_user.dropna()
        df_user = df_user.query("age!=1") #get rid of users whose age are under 18
        df_user = df_user.query("occupation!=0") #get rid of users w/ unspecified occupations
        
        # Only keep ratings with at least rating_thresh
        df_inter = df_inter.query("rating>=@rating_thresh")

        # Convert ratings to 1
        df_inter["mapped_rating"] = 1

    if "lfm1b-artists" in data:
        #clean missing country/age/gender info, exclude minors (<18 years) and erroneous ages (>100)
        df_user = df_user\
                        .dropna()\
                        .query("gender!='n' & age!='-1'")\
                        .query("18<=age<=100")
        
    if "jobs" in data:
        df_user = df_user\
                        .dropna()\
                        .query("TotalYearsExperience<=60")
        
        df_item = df_item.query("Title!='.'")

    #drop duplicates
    df_inter = df_inter.drop_duplicates(subset=["user_id", "item_id"])

    #filter df_inter accordingly
    df_inter = df_inter[df_inter.user_id.isin(df_user.user_id)]

    # Ensure items have metadata, otherwise remove because we won't be able to compare the title during eval
    df_inter = df_inter[df_inter.item_id.isin(df_item.item_id)]

    # Apply k-core filtering
    df_inter = core_filtering(df_inter, core_filter)

    # Sort by timestamp
    df_inter = df_inter.sort_values("timestamp")

    # Split into train test
    train_n = round(df_inter.shape[0] * train_prop) 
    df_train = df_inter.head(train_n) 

    test_n = round(df_inter.shape[0] * test_prop)
    df_test = df_inter.tail(test_n)

    df_val = df_inter.loc[~df_inter.index.isin(df_train.index)]
    df_val = df_val.loc[~df_val.index.isin(df_test.index)]

    # Keep items with >= min_inter in train
    df_train_item_val_count = df_train.item_id.value_counts()
    filtered_item_id = df_train_item_val_count[df_train_item_val_count >= min_inter].index

    # Filter train/val/test accordingly
    df_train = df_train.loc[df_train.item_id.isin(filtered_item_id)]
    df_val = df_val.loc[df_val.item_id.isin(filtered_item_id)]
    df_test = df_test.loc[df_test.item_id.isin(filtered_item_id)]

    # Keep users with >= min_inter in train
    df_train_user_val_count = df_train.user_id.value_counts()
    filtered_user_id = df_train_user_val_count[df_train_user_val_count >= min_inter].index

    # Filter train/val/test accordingly
    df_train = df_train.loc[df_train.user_id.isin(filtered_user_id)]
    df_val = df_val.loc[df_val.user_id.isin(filtered_user_id)]
    df_test = df_test.loc[df_test.user_id.isin(filtered_user_id)]

    # Remove users from val/test that don't exist in df_train
    train_user = df_train.user_id.unique()
    df_val = df_val.loc[df_val.user_id.isin(train_user)]
    df_test = df_test.loc[df_test.user_id.isin(train_user)]

    # Remove items from val/test that don't exist in df_train
    train_item = df_train.item_id.unique()
    df_val = df_val.loc[df_val.item_id.isin(train_item)]
    df_test = df_test.loc[df_test.item_id.isin(train_item)]

    # Filter user metadata accordingly
    train_user = df_train.user_id.unique()
    val_user = df_val.user_id.unique()
    test_user = df_test.user_id.unique()
    all_users = np.concatenate([train_user, val_user, test_user])
    
    df_user = df_user.loc[df_user.user_id.isin(all_users)]

    # Filter item metadata accordingly
    train_item = df_train.item_id.unique() 
    val_item = df_val.item_id.unique() 
    test_item = df_test.item_id.unique() 
    all_items = np.concatenate([train_item, val_item, test_item])

    df_item = df_item.dropna()
    df_item = df_item.loc[df_item.item_id.isin(all_items)]

    return df_train, df_val, df_test, df_user, df_item


def _keep_column_and_valid_user(df, df_user, df_item):
    #for LFM
    if "num_repeat" in df.columns:
        df = df[["user_id","item_id", "timestamp", "num_repeat"]]

    else:
        df = df[["user_id","item_id", "timestamp"]]

    #needed for jobrec, as we filter user again outside of the filter_data process
    df = df[df.user_id.isin(df_user.user_id)]

    #needed for LFM
    df = df[df.item_id.isin(df_item.item_id)]
    return df

def prepare_to_save(df_train, df_val, df_test, df_user, df_item):
    df_train = _keep_column_and_valid_user(df_train, df_user, df_item)
    df_val = _keep_column_and_valid_user(df_val, df_user, df_item)
    df_test = _keep_column_and_valid_user(df_test, df_user, df_item)

    #filter so that we keep only metadata of items that will be useful
    df_item = df_item[(df_item.item_id.isin(df_train.item_id))|(df_item.item_id.isin(df_val.item_id))|(df_item.item_id.isin(df_test.item_id))]

    return df_train, df_val, df_test, df_item


def general_save(df_train, df_val, df_test, df_user, df_item, data_name):
    df_train.to_pickle(f"{data_name}_train.pkl")
    df_val.to_pickle(f"{data_name}_val.pkl")
    df_test.to_pickle(f"{data_name}_test.pkl")
    df_user.to_pickle(f"{data_name}_user.pkl")
    df_item.to_pickle(f"{data_name}_item.pkl")


# === RECAI ===

def load_cleaned_data(data_name):
    df_train = pd.read_pickle(f"{data_name}_train.pkl")
    df_val = pd.read_pickle(f"{data_name}_val.pkl")
    df_test = pd.read_pickle(f"{data_name}_test.pkl")
    df_user = pd.read_pickle(f"{data_name}_user.pkl")
    df_item = pd.read_pickle(f"{data_name}_item.pkl")

    return df_train, df_val, df_test, df_user, df_item

def select_and_map_column_for_recai(df_user, data):
    map_gender = {"M": "male", "F": "female"}

    map_occupation = {
                1:  "academic/educator",
                2:  "artist",
                3:  "clerical/admin",
                4:  "college/grad student",
                5:  "customer service",
                6:  "doctor/health care",
                7:  "executive/managerial",
                8:  "farmer",
                9:  "homemaker",
                10:  "K-12 student",
                11:  "lawyer",
                12:  "programmer",
                13:  "retired",
                14:  "sales/marketing",
                15:  "scientist",
                16:  "self-employed",
                17:  "technician/engineer",
                18:  "tradesman/craftsman",
                19:  "unemployed",
                20:  "writer",
            }

    #ml-1m
    map_age = {
        18:  "18-24",
        25:  "25-34",
        35:  "35-44",
        45:  "45-49",
        50:  "50-55",
        56:  "56+"
    }


    if data == "ml-1m":
        df_user = df_user[["user_id", "gender", "age", "occupation"]]

        df_user["gender"] = df_user["gender"].map(map_gender)
        df_user["age"] = df_user["age"].map(map_age)
        df_user["occupation"] = df_user["occupation"].map(map_occupation)
      

    elif data == "jobrec":
        df_user = df_user[["user_id","DegreeType", "Major","TotalYearsExperience"]]
    
    
    elif data =="lfm-1b":
        df_user = df_user[["user_id", "gender", "age", "country_name"]]
        
        df_user["gender"] = df_user["gender"].map(map_gender)

    return df_user


def create_inter_file(df, split, save_path):
    df_filtered = None
    cols = ["user_id", "item_id"]

    if split in ["test"]:
        df = df[cols]\
                    .groupby("user_id")\
                    .agg(lambda x: " ".join([str(x) for x in x]))\
                    .reset_index()
    elif split in ["train", "val"]:
        if "lfm-1b" in save_path and split =="train":
            cols += ["num_repeat"]

        # take only last 10 for train/val
        df = df[cols]\
                    .groupby("user_id")\
                    .agg(lambda x: [str(x) for x in x][-10:])
        
        df_filtered = df.copy()

        assert all(df.item_id.apply(len) <= 10)
        
        df["item_id"] = df["item_id"].apply(lambda x: " ".join(x))


        df = df.reset_index()
    
        if "lfm-1b" in save_path and split =="train":
            df["num_repeat"] = df["num_repeat"].apply(lambda x: " ".join(x))
            df_num_repeat = df["user_id"].astype(str) + " " + df["num_repeat"]

            df_num_repeat.to_csv(f"{save_path}/repeat_temp.txt",
                    sep=" ",
                    header=None,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
            )

            #read and remove the quotation marks
            new = []

            with open(f"{save_path}/repeat_temp.txt", "rb") as f:
                line = f.readlines()
                for l in line:
                    l = l.replace(b'"',b"")
                    new.append(l)
            
            #remove the temporary file
            os.remove(f"{save_path}/repeat_temp.txt")

            #write lines without quotation marks
            with open(f"{save_path}/repeat.txt", "wb") as f:
                f.writelines(new)



    df = df["user_id"].astype(str) + " " + df["item_id"]

    df.to_csv(f"{save_path}/{split}_temp.txt",
            sep=" ",
            header=None,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
    )

    #read and remove the quotation marks
    new = []

    with open(f"{save_path}/{split}_temp.txt", "rb") as f:
        line = f.readlines()
        for l in line:
            l = l.replace(b'"',b"")
            new.append(l)
    
    #remove the temporary file
    os.remove(f"{save_path}/{split}_temp.txt")

    #write lines without quotation marks
    with open(f"{save_path}/{split}.txt", "wb") as f:
        f.writelines(new)

    return df_filtered

# === Dataset-specific cleaning === 

def clean_movie_name(name):
    # comma in bracket
    pat = "\((.*), (The|A|An|La|Les|El)\)"
    res = re.search(pat, name)

    if res:
        all_result = res.group()
        before_comma = res.group(1)
        after_comma = res.group(2)

        name = name.replace(all_result, f"({after_comma} {before_comma})")

    # comma before bracket
    pat = "(.*), (The|A|An|La|Les|El) \("
    res = re.search(pat, name)

    if res:
        all_result = res.group()
        before_comma = res.group(1)
        after_comma = res.group(2)

        name = name.replace(all_result, f"{after_comma} {before_comma} (")

    # comma (no bracket)
    pat = "(.*), (The|A|An|La|Les|El)"
    res = re.search(pat, name)

    if res: 
        all_result = res.group()
        before_comma = res.group(1)
        after_comma = res.group(2)

        name = name.replace(all_result, f"{after_comma} {before_comma}")

    return name
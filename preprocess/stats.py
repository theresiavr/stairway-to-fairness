import pandas as pd
import numpy as np

def print_stats(val_count: pd.Series):
    to_print = ""
    for idx, val in zip(val_count.index, val_count.values):
        to_print += f"{idx} ({val}); "
    to_print = to_print[:-2]
    print(to_print)


def get_df_stats(df_train, df_val, df_test, df_user, df_item):
    #num inter in total
    num_inter = pd.concat([
                        df_train, 
                        df_val, 
                        df_test]).shape[0]
    
    print(f"Num inter: {num_inter}")


    #num_items in total
    total_num_item = len(set(np.concatenate([
                            df_train.item_id.unique(), 
                            df_val.item_id.unique(), 
                            df_test.item_id.unique()])))
    
    num_item_df_item = df_item.item_id.nunique()

    assert total_num_item == num_item_df_item


    print(f"Num items: {total_num_item}")
    
    #num_users in total
    test_users =  df_test.user_id
    num_test_users = test_users.nunique()


    total_num_user = len(set(np.concatenate([
                            df_train.user_id.unique(), 
                            df_val.user_id.unique(), 
                            test_users.unique()
                           ])))
    
    num_user_df_user = df_user.user_id.nunique()

    assert total_num_user == num_user_df_user


    print(f"Num users (test): {total_num_user} ({num_test_users})")

    sparsity = 1 - num_inter / (total_num_user * total_num_item)

    print(f"Sparsity: {round(sparsity, 5)}")

    
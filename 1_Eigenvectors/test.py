import pandas as pd

bank_df = pd.read_csv("bank-full.csv")
bank_df_copy = bank_df.copy()
print(bank_df_copy)

bank_df_copy = bank_df_copy.drop("contact", axis = 1) #inplace is False by default
print(bank_df_copy)

bank_df_copy.drop(3, axis = 0, inplace = True)
print(bank_df_copy)

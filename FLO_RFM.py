import pandas as pd
import datetime as dt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


def data_inspection(inp_dataframe, today_date, segment_names):
    print("############################################# First 10 Rows #############################################")
    print(inp_dataframe.head(10))
    print("############################################# Description of the Dataframe #############################################")
    print(inp_dataframe.describe().T)
    print("############################################# Check if missing values exist #############################################")
    print(inp_dataframe.isnull().sum())
    # Dropping missing values from the dataframe
    inp_dataframe.dropna(inplace=True)
    # Adding new columns
    inp_dataframe["order_num_ever"] = inp_dataframe["order_num_total_ever_online"] + inp_dataframe["order_num_total_ever_offline"]
    inp_dataframe["order_value_ever"] = inp_dataframe["customer_value_total_ever_offline"] + inp_dataframe["customer_value_total_ever_online"]
    # Change the data type of dates to datetime
    date_cols = [col for col in inp_dataframe.columns if "date" in col]
    inp_dataframe[date_cols] = inp_dataframe[date_cols].apply(pd.to_datetime)
    # Calculating recency, frequency, monetary values and scores, calculating RF/RFM scores
    inp_dataframe["Recency"] = inp_dataframe["last_order_date"].apply(lambda x: (today_date - x).days)
    inp_dataframe["Frequency"] = inp_dataframe["order_num_ever"]
    inp_dataframe["Monetary"] = inp_dataframe["order_value_ever"]
    inp_dataframe["Recency Score"] = pd.qcut(inp_dataframe["Recency"], q=5, labels=[5, 4, 3, 2, 1])
    inp_dataframe["Frequency Score"] = pd.qcut(inp_dataframe["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
    inp_dataframe["Monetary Score"] = pd.qcut(inp_dataframe["Monetary"], q=5, labels=[1, 2, 3, 4, 5])
    inp_dataframe["RF Score"] = inp_dataframe["Recency Score"].astype("str") + inp_dataframe["Frequency Score"].astype("str")
    inp_dataframe["RFM Score"] = inp_dataframe["Recency Score"].astype("str") + \
                                 inp_dataframe["Frequency Score"].astype("str") + \
                                 inp_dataframe["Monetary Score"].astype("str")

    # segment_means = flo_data.groupby("Segments").agg({"Recency": "mean",
    #                                                 "Frequency": "mean",
    #                                                  "Monetary": "mean"})

    inp_dataframe["Segments"] = inp_dataframe["RF Score"].replace(segment_names, regex=True)

    print("############################################# First 10 Rows After Manupilations #############################################")
    print(inp_dataframe.head(10))
    return inp_dataframe


# Read the csv
flo_data = pd.read_csv("flo_data_20k.csv")
today = dt.datetime(2021, 6, 1)
segment_mapping = {r"[1-2][1-2]": "hibernating",
                   r"[1-2][3-4]": "at_risk",
                   r"[1-2]5": "cant_lose",
                   r"3[1-2]": "about_to_sleep",
                   r"33": "need_attention",
                   r"[3-4][4-5]": "loyal_customers",
                   r"41": "promising",
                   r"51": "new_customers",
                   r"[4-5][2-3]": "potential_loyalists",
                   r"5[4-5]": "champions"}
flo_data = data_inspection(flo_data, today, segment_mapping)

# Finding customers that are in segments "champions, loyal customers" and minimum average spending of 250
# and are interested in women shoes
new_brand_target__customer_id = flo_data.loc[(flo_data["Segments"].str.contains("|".join(["champions", "loyal_customers"]))) &
                                             (flo_data["order_value_ever"] / flo_data["order_num_ever"] > 250.0) &
                                             (flo_data["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
# Finding customers that are in segments "about to sleep, new customers" and interested in men/children shoes
new_brand_target__customer_id.to_csv("new_brand_target__customer_id.csv")
men_children_discount_id = flo_data.loc[(flo_data["Segments"].str.contains("|".join(["cant_lose", "about_to_sleep", "new_customers"]))) &
                                        (flo_data["interested_in_categories_12"].str.contains("|".join(["ERKEK", "COCUK"])))]["master_id"]
men_children_discount_id.to_csv("men_children_discount_id.csv")

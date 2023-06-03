import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import matplotlib.pyplot as plt

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.float_format", lambda x: '%.4f' % x)

flo_data = pd.read_csv("flo_data_20k.csv")
today = dt.datetime(2021, 6, 1)
flo_data["order_num_ever"] = flo_data["order_num_total_ever_online"] + flo_data["order_num_total_ever_offline"]
flo_data["customer_value_ever"] = flo_data["customer_value_total_ever_online"] + flo_data["customer_value_total_ever_offline"]
flo_data.head()

def check(df):
    summary = []
    cols = df.columns
    for col in cols:
        data_types = df[col].dtypes
        num_unique = df[col].nunique()
        sum_null = df[col].isnull().sum()
        summary.append([col, data_types, num_unique, sum_null])
    df_check = pd.DataFrame(summary, columns=['columns', 'dtypes', 'nunique', 'sum_null'])
    print(df_check)
def outlier_thresholds(df, variables, q_limit):
    upper_quantiles = []
    lower_quantiles = []
    interquantile_ranges = []
    upper_limits = []
    lower_limits = []
    for i, variable in enumerate(variables):
        upper_quantiles.append(df[variable].quantile(1 - q_limit / 100))
        lower_quantiles.append(df[variable].quantile(q_limit / 100))
        interquantile_ranges.append(upper_quantiles[i] - lower_quantiles[i])
        upper_limits.append(upper_quantiles[i] + interquantile_ranges[i] * 1.5)
        lower_limits.append(lower_quantiles[i] - interquantile_ranges[i] * 1.5)
    return upper_limits, lower_limits
def replace_with_tresholds(df, variables, q_limit):
    upper, lower = outlier_thresholds(df, variables, q_limit)
    for i, variable in enumerate(variables):
        df = df.loc[(df[variable] < upper[i]) & (df[variable] >= lower[i])]
    return df
def cltv_calculation(df, today_date, supressed_variables):
    df = replace_with_tresholds(df, supressed_variables, 1)

    date_variables = [x for x in df.columns if "date" in x]
    df[date_variables] = df[date_variables].apply(pd.to_datetime)

    cltv_df = pd.DataFrame()
    cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7
    cltv_df["t_weekly"] = df["first_order_date"].apply(lambda x: (today_date - x).days) / 7
    cltv_df["frequency"] = df["order_num_ever"]
    cltv_df["monetary_cltv_avg"] = df["customer_value_ever"] / df["order_num_ever"]
    cltv_df.set_index(df["master_id"])
    cltv_df = cltv_df.loc[cltv_df["frequency"] > 1]

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["t_weekly"])
    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

    cltv_df["3_month_expected_sales"] = bgf.conditional_expected_number_of_purchases_up_to_time(12, cltv_df["frequency"],
                                                                                                cltv_df["recency_cltv_weekly"],
                                                                                                cltv_df["t_weekly"])
    cltv_df["6_month_expected_sales"] = bgf.conditional_expected_number_of_purchases_up_to_time(24, cltv_df["frequency"],
                                                                                                cltv_df["recency_cltv_weekly"],
                                                                                                cltv_df["t_weekly"])

    cltv_df["expected_avg_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

    cltv_df["cltv"] = ggf.customer_lifetime_value(bgf, cltv_df["frequency"],
                                                  cltv_df["recency_cltv_weekly"],
                                                  cltv_df["t_weekly"],
                                                  cltv_df["monetary_cltv_avg"], time=6, freq="W", discount_rate=0.0)

    cltv_df["segments"] = pd.qcut(cltv_df["cltv"], q=4,
                                  labels=["penny_pinchers", "rare_shoppers", "frequent_shoppers", "crazy_shoppers"])

    return cltv_df


supressed_variables = ["customer_value_total_ever_offline", "customer_value_total_ever_online",
                       "order_num_total_ever_online", "order_num_total_ever_offline"]

cltv_dataframe = cltv_calculation(flo_data, today, supressed_variables)
plt.scatter(cltv_dataframe["cltv"], cltv_dataframe["frequency"], marker="o", color="red")
plt.xlabel("cltv", fontsize=18)
plt.ylabel("frequency", fontsize=18)
plt.show()
plt.scatter(cltv_dataframe["cltv"], cltv_dataframe["recency_cltv_weekly"], marker="o", color="green", alpha=0.75)
plt.xlabel("cltv", fontsize=18)
plt.ylabel("recency_cltv_weekly", fontsize=18)
plt.show()


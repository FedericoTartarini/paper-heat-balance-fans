import requests
from pprint import pprint
import json
import codecs
import pandas as pd
import os
import time
import sqlite3
import numpy as np
import json


def get_population_data_from_un_dataset():

    df = pd.read_csv(os.path.join(os.getcwd(), "code", "UNdata_cities.csv"))
    df = df.dropna(subset=["Country or Area"])
    df["City"] = df["City"].str.lower()
    df["Country or Area"] = df["Country or Area"].str.lower()

    # some big countries are missing from the ASHRAE database

    df_population = pd.DataFrame()

    for country in df["Country or Area"].unique():
        # # print countries not in the list
        # if country.lower() in df_ashrae["country"].unique():
        #     # df_ashrae["country"].sort_values().unique()
        #     # print(country, " is there")
        #     pass
        # else:
        #     print(country)
        df_country = df[df["Country or Area"] == country]
        for city in df_country["City"].unique():
            df_city = df_country[df_country["City"] == city]
            df_city = df_city[df_city["Sex"] == "Both Sexes"]
            df_city = df_city.sort_values(["City type", "Year"], ascending=False)
            if df_city.shape[0] > 0:
                df_population = df_population.append(df_city.iloc[0])

    df_population["city+country"] = (
        df_population["City"]
        .str.split(" ", expand=True)[0]
        .str.split(",", expand=True)[0]
        + "+"
        + df_population["Country or Area"]
    )

    return df_population


def import_ashrae_data():
    """Source http://data.un.org/Data.aspx?d=POP&f=tableCode:240"""

    df_ashrae = pd.read_sql(
        "SELECT wmo, lat, long, place, "
        '"n-year_return_period_values_of_extreme_DB_10_max" as db_max, '
        '"n-year_return_period_values_of_extreme_WB_10_max" as wb_max '
        "FROM data",
        con=sqlite3.connect(os.path.join(os.getcwd(), "code", "weather_ashrae.db")),
    )
    df_ashrae[["city", "country"]] = (
        df_ashrae["place"].str.lower().str.rsplit(", ", n=1, expand=True)
    )

    df_ashrae["city+country"] = (
        df_ashrae["city"].str.split(" ", expand=True)[0].str.split(",", expand=True)[0]
        + "+"
        + df_ashrae["country"]
    )

    df_ashrae = df_ashrae.replace("N/A", np.nan)
    df_ashrae = df_ashrae.astype({"db_max": "float"})
    df_ashrae = df_ashrae.dropna(subset=["db_max"])
    df_ashrae = df_ashrae.sort_values("db_max", ascending=False)
    df_ashrae["city"] = df_ashrae["city"].str.split(" intl", expand=True)[0]
    df_ashrae["city"] = df_ashrae["city"].str.split(",", expand=True)[0]

    return df_ashrae


def request_data(query):
    # https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a
    # -population-1000/information/?disjunctive.country&q=iran+abadan

    r = requests.get(
        f"https://public.opendatasoft.com/api/records/1.0/search/?dataset=geonames-all-cities-with-a-population-1000&q={query}&facet=timezone&facet=country",
    )

    try:
        decoded_data = codecs.decode(r.text.encode(), "utf-8-sig")
        decoded_data = json.loads(decoded_data)
    except json.decoder.JSONDecodeError:
        decoded_data = False

    return decoded_data


def query_api_population(df_to_query):

    for ix, val in enumerate(df_to_query["query_col"].unique()):

        print(
            f"quering: {val} percentage queried: {int(ix/df_to_query.shape[0]*100)} %"
        )

        with open("./code/queries_population.json") as f:
            previous_queries = json.load(f)

        data = request_data(query=val)
        # pprint(data)
        previous_queries.append(data)
        # time.sleep(0.2)

        with open("./code/queries_population.json", "w") as f:
            json.dump(previous_queries, f)


df_population = get_population_data_from_un_dataset()

df_ashrae = import_ashrae_data()

df_merged = pd.merge(
    df_ashrae, df_population[["city+country", "Value"]], on="city+country", how="left",
)

df_merged = df_merged[~df_merged["city"].str.contains("buoy")]
df_merged = df_merged[~df_merged["city"].str.contains("intercontinental")]
df_merged = df_merged[~df_merged["city"].str.contains("air terminal")]
df_merged = df_merged[~df_merged["city"].str.contains("memorial")]
df_merged = df_merged[~df_merged["city"].str.contains("campus")]
df_merged["country"] = df_merged["country"].replace(
    {"usa": "United States"}, regex=True
)
df_merged["country"] = df_merged["country"].replace(
    {"u.s.": "United States"}, regex=True
)
df_merged["country"] = df_merged["country"].replace(
    {"islamic republic of": "Iran"}, regex=True
)
df_merged["country"] = df_merged["country"].replace(
    {"republic of": "Korea"}, regex=True
)
df_merged["city"] = df_merged["city"].replace({" ap": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" downtown": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" airfield": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" county": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" regional": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" lighthouse": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" city": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" aws": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" afb": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" field": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace(
    {" meteorological office": ""}, regex=True
)
df_merged["city"] = df_merged["city"].replace({" municipal": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" raaf": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" executive": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" post office": ""}, regex=True)
df_merged["city"] = df_merged["city"].replace({" fertilia": ""}, regex=True)

df_merged["query_col"] = df_merged["city"] + " " + df_merged["country"]
df_merged["ascii_name"] = None
df_merged["country_code"] = None

df_merged = df_merged.drop_duplicates(subset="query_col")

df_to_query = df_merged[df_merged.db_max > 30]
df_to_query = df_to_query[df_to_query["Value"].isna()]

# query_api_population(df_to_query)

with open("./code/queries_population.json") as f:
    previous_queries = json.load(f)

for result in previous_queries:
    if "error" in result.keys():
        pass

    elif result["nhits"] != 0:
        data = result["records"][0]["fields"]
        df_merged.loc[
            df_merged["query_col"] == result["parameters"]["q"], "Value"
        ] = data["population"]
        df_merged.loc[
            df_merged["query_col"] == result["parameters"]["q"], "ascii_name"
        ] = data["ascii_name"]
        df_merged.loc[
            df_merged["query_col"] == result["parameters"]["q"], "country_code"
        ] = data["country_code"]

# for city in df_merged[(df_merged["country"] == "usa") & (df_merged["Value"].isna())][
#     "city"
# ]:
#     print(city)
#
# df_merged[(df_merged["city"] == "san francisco")]["Value"]
df_merged["Value"].isna().sum()

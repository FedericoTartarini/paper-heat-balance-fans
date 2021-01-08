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
            # Sorting on city type since urban agglomerate is what I want
            df_city = df_city.sort_values(
                ["City type", "Year", "Value"], ascending=False
            )
            if df_city.shape[0] > 0:
                df_population = df_population.append(df_city.iloc[0])

    df_population["Country or Area"] = df_population["Country or Area"].replace(
        {"democratic people's republic of korea": "north korea"}, regex=True
    )
    df_population["Country or Area"] = df_population["Country or Area"].replace(
        {"republic of korea": "south korea"}, regex=True
    )
    df_population["Country or Area"] = df_population["Country or Area"].replace(
        {"islamic republic of": "iran"}, regex=True
    )

    df_population["City"] = df_population["City"].str.split("(", expand=True)[0]

    df_population["City"] = df_population["City"].replace(
        {"montréal": "montreal"}, regex=True
    )
    df_population["City"] = df_population["City"].replace(
        {"guadalupe, nuevo león": "monterrey"}, regex=True
    )
    df_population["City"] = df_population["City"].replace(
        {"bogotá, d.c.": "bogota"}, regex=True
    )
    df_population["City"] = df_population["City"].replace({"-tlaxcala": ""}, regex=True)

    df_population["city+country"] = (
        df_population["City"].str.strip()
        # .str.split(" ", expand=True)[0]
        # .str.split(",", expand=True)[0]
        + "+"
        + df_population["Country or Area"]
    )

    df_population = df_population[
        df_population["city+country"] != "tlalnepantla+mexico"
    ]
    df_population = df_population[
        df_population["city+country"] != "new delhi municipal council+india"
    ]
    df_population = df_population[
        df_population["city+country"] != "bruhat bengaluru mahanagara palike+india"
    ]
    df_population = df_population[
        df_population["city+country"] != "greater hyderabad municipal corporation+india"
    ]
    df_population = df_population[
        df_population["city+country"] != "greater mumbai+india"
    ]
    df_population = df_population[
        df_population["city+country"] != "greater sydney+australia"
    ]
    df_population = df_population[
        df_population["city+country"] != "greater melbourne+australia"
    ]
    df_population = df_population[
        df_population["city+country"] != "greater brisbane+australia"
    ]
    df_population = df_population[
        df_population["city+country"] != "west midlands+united kingdom"
    ]
    df_population = df_population[df_population["city+country"] != "zapopan+mexico"]
    df_population = df_population[df_population["city+country"] != "tlaquepaque+mexico"]
    df_population = df_population[df_population["city+country"] != "tonala+mexico"]
    df_population = df_population[df_population["city+country"] != "dongguan+china"]

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

    df_ashrae["city"] = (
        df_ashrae["city"].str.lower().str.split(", ", n=1, expand=True)[0]
    )

    df_ashrae = df_ashrae[~df_ashrae["city"].str.contains("buoy")]
    df_ashrae = df_ashrae[~df_ashrae["city"].str.contains("intercontinental")]
    df_ashrae = df_ashrae[~df_ashrae["city"].str.contains("air terminal")]
    df_ashrae = df_ashrae[~df_ashrae["city"].str.contains("memorial")]
    df_ashrae = df_ashrae[~df_ashrae["city"].str.contains("campus")]

    df_ashrae["city"] = df_ashrae["city"].replace({" ap": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" downtown": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" airfield": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" county": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" regional": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" lighthouse": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" city": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" aws": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" afb": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" field": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" parafield": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {" meteorological office": ""}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({" municipal": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" raaf": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" executive": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" post office": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" fertilia": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" aaf": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" natl": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" intl": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" dam": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" capital": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" observatory": ""}, regex=True)

    # change name cities
    df_ashrae["city"] = df_ashrae["city"].replace({" orly": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" hongqiao": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" shivaji": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" heathrow": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" congonhas": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" jiangbeiintl": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" callao": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" baiyun": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" bao'an": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" begumpet": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" pudahuel": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" king khaled": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" pulkovo": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" bose": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" jomo kenyatta": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" shuangliu": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" st-hubert": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" logan": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" barajas": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" coleman young": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" tegel": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" central park": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" tianhe": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" ellington": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" lukou": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" king abdulaziz": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" longjia": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" wujiaba": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" tsinan": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" longdongbao": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" sky harbor": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" ciampino": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"-tacoma": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" pestszentlorinc": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" suna": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" wusu": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" xiaoshan": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" ambedkar": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({" archerfield": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"-st paul": ""}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"hong kong": "hong kong sar"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"juanda": "surabaya"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"xinzheng": "zhengzhou"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"eldorado": "bogota"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"kiev boryspil": "kyiv"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"pyongyangn": "pyongyang"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"science garden": "quezon city"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"harbin": "haerbin"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"moscow sheremetyevo": "moskva"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"jeddah": "jiddah"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"jorge newbery": "buenos aires"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"jinnah": "karachi"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"mexico": "mexico, ciudad de"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"new delhi indira gandhi": "delhi"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"jose joaquin de olmedo": "guayaquil"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"heydar aliyev": "baku"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"singapore changiintl": "singapore"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"rio santos dumont": "rio de janeiro"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"bangaluru": "bangalore"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"ahmedabad": "ahmadabad"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"st petersburg": "st. petersburg"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace(
        {"dar el beida": "algiers"}, regex=True
    )
    df_ashrae["city"] = df_ashrae["city"].replace({"jinghe": "xi'an"}, regex=True)
    df_ashrae["city"] = df_ashrae["city"].replace({"mehrabad": "tehran"}, regex=True)
    df_ashrae = df_ashrae.append(
        df_ashrae[df_ashrae["city"] == "lucknow"].replace({"lucknow": "kanpur"})
    )

    df_ashrae["country"] = df_ashrae["country"].replace(
        {"democratic people's republic of": "north korea"}, regex=True,
    )
    df_ashrae["country"] = df_ashrae["country"].replace(
        {"islamic republic of": "iran"}, regex=True
    )
    df_ashrae["country"] = df_ashrae["country"].replace(
        {"republic of": "south korea"}, regex=True
    )
    df_ashrae["country"] = df_ashrae["country"].replace({"usa": "u.s."}, regex=True)

    df_ashrae["city+country"] = (
        # df_ashrae["city"].str.split(" ", expand=True)[0].str.split(",", expand=True)[0]
        df_ashrae["city"]
        + "+"
        + df_ashrae["country"]
    )

    df_ashrae = df_ashrae.replace("N/A", np.nan)
    df_ashrae = df_ashrae.astype({"db_max": "float"})
    df_ashrae = df_ashrae.dropna(subset=["db_max"])
    df_ashrae = df_ashrae.drop_duplicates()
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


if __name__ == "__main__":

    df_ashrae = import_ashrae_data()

    df_population = get_population_data_from_un_dataset()

    # print which countries are missing in the ashrae database
    country_pop = df_population["Country or Area"].unique()
    country_ash = df_ashrae["country"].unique()

    [x for x in country_pop if x not in country_ash]

    # big countries missing
    big_countries_missing = [
        "bangladesh",
        "ethiopia",
        "iraq",
        "mozambique",
        "nepal",
        "nigeria",
        "republic of south sudan",
        "rwanda",
        "uganda",
        "yemen",
        "zambia",
    ]

    df_population = df_population.drop_duplicates(subset="city+country")

    df_population.Value.sum() / 10 ** 9

    df_merged = pd.merge(
        df_population[["city+country", "Value"]],
        df_ashrae,
        on="city+country",
        how="left",
    )

    df_merged.Value.sum() / 10 ** 9

    df_merged = df_merged.sort_values("Value", ascending=False).drop_duplicates(
        subset="city+country"
    )

    df_merged.sort_values("Value", ascending=False).to_csv(
        os.path.join(os.getcwd(), "code", "population_weather.csv"), index=False
    )

    # country = "china"
    # _df = df_merged[df_merged.country == country]
    # df_population[df_population["Country or Area"] == country]["Value"].sum() / 10 ** 6
    # df_merged[(df_merged.country == country) & (~df_merged["db_max"].isna())][
    #     "Value"
    # ].sum() / 10 ** 6
    #
    # # df_merged["country"] = df_merged["country"].replace(
    # #     {"usa": "United States"}, regex=True
    # # )
    #
    # df_merged["query_col"] = df_merged["city"] + " " + df_merged["country"]
    # df_merged["ascii_name"] = None
    # df_merged["country_code"] = None
    #
    # df_merged = df_merged.drop_duplicates(subset="query_col")
    #
    # df_to_query = df_merged[df_merged.db_max > 30]
    # df_to_query = df_to_query[df_to_query["Value"].isna()]
    #
    # # query_api_population(df_to_query)
    #
    # with open("./code/queries_population.json") as f:
    #     previous_queries = json.load(f)
    #
    # for result in previous_queries:
    #     if "error" in result.keys():
    #         pass
    #
    #     elif result["nhits"] != 0:
    #         data = result["records"][0]["fields"]
    #         df_merged.loc[
    #             df_merged["query_col"] == result["parameters"]["q"], "Value"
    #         ] = data["population"]
    #         df_merged.loc[
    #             df_merged["query_col"] == result["parameters"]["q"], "ascii_name"
    #         ] = data["ascii_name"]
    #         df_merged.loc[
    #             df_merged["query_col"] == result["parameters"]["q"], "country_code"
    #         ] = data["country_code"]
    #
    # # for city in df_merged[(df_merged["country"] == "usa") & (df_merged["Value"].isna())][
    # #     "city"
    # # ]:
    # #     print(city)
    # #
    # # df_merged[(df_merged["city"] == "san francisco")]["Value"]
    # df_merged["Value"].isna().sum()
    #
    # df_merged = df_merged.sort_values("Value", ascending=False)

import psychrolib
import requests
from pprint import pprint
import json
import codecs
import pandas as pd
import os
import time
import sqlite3
import sys

psychrolib.SetUnitSystem(psychrolib.SI)

# create a db to hold the data
conn = sqlite3.connect(os.path.join(os.getcwd(), "code", "weather_ashrae.db"))

# read file which contains list of wmo
df_ashrae = pd.read_csv(
    os.path.join(os.getcwd(), "code", "ASHRAE_2013_Yearly_DesignConditions.csv"),
    encoding="ISO-8859-1",
    header=6,
)

# get the list of stations that I need to query
stations = df_ashrae["WMO#"]
stations = [x if len(x) != 5 else f"0{x}" for x in stations]
df_ashrae["WMO#"] = stations

# remove from the list the stations that were queried already
stations_in_db = pd.read_sql("SELECT wmo FROM data", con=conn).values

stations = [x for x in stations if x not in stations_in_db]

# skip the first station since no data are available online
stations_data_not_available = [
    "020800",
    "020960",
    "021120",
    "021200",
    "021280",
    "021420",
    "022060",
    "023240",
    "023760",
    "024100",
    "025760",
    "026300",
    "026720",
    "026800",
    "028480",
    "029135",
    "033600",
    "038170",
    "072554",
    "081605",
    "090910",
    "091620",
    "091700",
    "091840",
    "091930",
    "092610",
    "093610",
    "093790",
    "093850",
    "093930",
    "094880",
    "094900",
    "095540",
    "095780",
    "100465",
    "104920",
    "107340",
    "115410",
    "119330",
    "132280",
    "134624",
    "134810",
    "143015",
    "160365",
    "167464",
    "171295",
    "171900",
    "257440",
    "262180",
    "274590",
    "277310",
    "277860",
    "286790",
    "298620",
    "356630",
    "357910",
    "375490",
    "401760",
    "404170",
    "429340",
    "589680",
    "619880",
    "681040",
    "681740",
    "690140",
    "690160",
    "690190",
    "690230",
    "691464",
    "701335",
    "710377",
    "711010",
    "711140",
    "711380",
    "711510",
    "711820",
    "711830",
    "712073",
    "712640",
    "713020",
    "713130",
    "713570",
    "714765",
    "714880",
    "714960",
    "715330",
    "715580",
    "715870",
    "716290",
    "716760",
    "716950",
    "717110",
    "717410",
    "718233",
    "718360",
    "718440",
    "718700",
    "718740",
    "718750",
    "718765",
    "718810",
    "718980",
    "719100",
    "719130",
    "719204",
    "719330",
    "719674",
    "722025",
    "722045",
    "722056",
    "722068",
    "722135",
    "722142",
    "722146",
    "722317",
    "722359",
    "722435",
    "722445",
    "722544",
    "722583",
    "722596",
    "722618",
    "722636",
    "722735",
    "723020",
    "723040",
    "723084",
    "723125",
    "723145",
    "723183",
    "723246",
    "723265",
    "723446",
    "723489",
    "723527",
    "723546",
    "723604",
    "724045",
    "724089",
    "724236",
    "724284",
    "724339",
    "724396",
    "724456",
    "724917",
    "724935",
    "724936",
    "725033",
    "725035",
    "725063",
    "725095",
    "725115",
    "725185",
    "725246",
    "725455",
    "725715",
    "725745",
    "725846",
    "725864",
    "726088",
    "726164",
    "726284",
    "726409",
    "726415",
    "726686",
]
stations = [x for x in stations if x not in stations_data_not_available]


def request_data(station_wmo="726884"):
    r = requests.post(
        "http://ashrae-meteo.info/v2.0/request_meteo_parametres.php",
        data={"wmo": station_wmo, "ashrae_version": 2017, "si_ip": "SI"},
        headers={"Content-type": "application/x-www-form-urlencoded"},
    )

    try:
        decoded_data = codecs.decode(r.text.encode(), "utf-8-sig")
        decoded_data = json.loads(decoded_data)
    except json.decoder.JSONDecodeError:
        decoded_data = False

    return decoded_data


for wmo in stations:

    # print(wmo)

    station_data = request_data(station_wmo=wmo)

    if station_data:

        df = pd.DataFrame(data=station_data["meteo_stations"][0], index=[0])

        df.to_sql("data", con=conn, if_exists="append")

    else:
        print(wmo)

    time.sleep(1)

# df_queried = pd.read_sql(
#     "SELECT wmo, "
#     '"n-year_return_period_values_of_extreme_DB_50_max" as db_max, '
#     '"n-year_return_period_values_of_extreme_WB_50_max" as wb_max '
#     "FROM data",
#     con=conn,
# )
#
# df_queried.merge(df_ashrae[["DBmax50years", "WMO#"]], left_on="wmo", right_on="WMO#")
#
# psychrolib.GetRelHumFromTWetBulb(20.6, 15.1, 101325)
#
# arr_rh = []
# df_queried[["db_max", "wb_max"]] = df_queried[["db_max", "wb_max"]].apply(
#     pd.to_numeric, errors="coerce"
# )
# df_queried.dropna(inplace=True)
# for ix, row in df_queried.iterrows():
#     arr_rh.append(
#         psychrolib.GetRelHumFromTWetBulb(row["db_max"], row["wb_max"], 101325)
#     )
#
# df_queried["rh"] = [x * 100 for x in arr_rh]
#
# import matplotlib.pyplot as plt
#
# plt.scatter(
#     df_queried["rh"], df_queried["db_max"],
# )
# plt.show()

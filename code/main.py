import matplotlib as mpl

mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, interp, maskoceans
import numpy as np
import pandas as pd
from pythermalcomfort.psychrometrics import p_sat, units_converter, p_sat_torr
from pythermalcomfort.models import set_tmp
import math
import seaborn as sns
import os
import scipy
from scipy import optimize
from pprint import pprint
import sqlite3
import psychrolib
from scipy.ndimage.filters import gaussian_filter1d

psychrolib.SetUnitSystem(psychrolib.SI)

plt.style.use("seaborn-paper")


class DataAnalysis:
    def __init__(self):
        self.dir_figures = os.path.join(os.getcwd(), "manuscript", "src", "figures")
        self.dir_tables = os.path.join(os.getcwd(), "manuscript", "src", "tables")

        self.ta_range = np.arange(28, 55, 0.5)
        self.v_range = [0.2, 0.8, 4.5]
        self.rh_range = np.arange(0, 105, 5)

        self.colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
        self.colors_f3 = ["tab:gray", "tab:cyan", "tab:olive"]

        try:
            self.heat_strain = np.load(
                os.path.join("code", "heat_strain.npy"), allow_pickle="TRUE"
            ).item()
        except FileNotFoundError:
            self.heat_strain = {}

        try:
            self.heat_strain_ollie = np.load(
                os.path.join("code", "heat_strain_ollie.npy"), allow_pickle="TRUE"
            ).item()
        except FileNotFoundError:
            self.heat_strain_ollie = {}

        self.conn = sqlite3.connect(
            os.path.join(os.getcwd(), "code", "weather_ashrae.db")
        )

        # define map extent
        self.lllon = -180
        self.lllat = -60
        self.urlon = 180
        self.urlat = 60

    def draw_map_contours(self, draw_par_mer="Yes"):
        # set up plot
        ax = plt.gca()

        # set up Basemap instance
        m = Basemap(
            projection="merc",
            llcrnrlon=self.lllon,
            llcrnrlat=self.lllat,
            urcrnrlon=self.urlon,
            urcrnrlat=self.urlat,
            resolution="l",
        )

        # draw map
        m.drawmapboundary(fill_color="white")
        m.drawcountries(
            linewidth=0.5,
            linestyle="solid",
            color="k",
            antialiased=True,
            ax=ax,
            zorder=3,
        )

        if draw_par_mer == "Yes":
            m.drawparallels(
                np.arange(self.lllat, self.urlat + 10, 10.0),
                color="black",
                linewidth=0.5,
                labels=[True, False, False, False],
            )
            m.drawmeridians(
                np.arange(self.lllon, self.urlon, 30),
                color="0.25",
                linewidth=0.5,
                labels=[False, False, False, True],
            )

        m.drawcoastlines(linewidth=0.5, color="k")
        m.drawstates(
            linewidth=0.5, linestyle="solid", color="gray",
        )

        return ax, m

    def plot_map_world(self, save_fig):

        # draw map contours
        plt.figure(figsize=(7, 3.78))
        [ax, m] = self.draw_map_contours(draw_par_mer="Yes")

        df = pd.read_sql(
            "SELECT wmo, lat, long, place, "
            '"n-year_return_period_values_of_extreme_DB_10_max" as db_max, '
            '"n-year_return_period_values_of_extreme_WB_10_max" as wb_max '
            "FROM data",
            con=self.conn,
        )
        df[["lat", "long", "db_max", "wb_max"]] = df[
            ["lat", "long", "db_max", "wb_max"]
        ].apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)

        df = df[
            (df["lat"] > self.lllat)
            & (df["lat"] < self.urlat)
            & (df["long"] > self.lllon)
            & (df["lat"] < self.urlon)
            & (df["db_max"] > 19.9)
        ]

        df = df.sort_values(["db_max"])

        # # print where the max dry bulb temperatures were recorded
        # df[["place", "db_max"]][-10:]
        # df_ = df.sort_values(["wb_max"])
        # df_[["place", "wb_max"]][-20:]

        # transform lon / lat coordinates to map projection
        proj_lon, proj_lat = m(df.long.values, df.lat.values)

        sc = plt.scatter(
            proj_lon, proj_lat, 10, marker="o", c=df["db_max"], cmap="plasma"
        )
        bounds = np.arange(
            math.floor(df["db_max"].min()), math.ceil(df["db_max"].max()), 4
        )
        plt.colorbar(
            sc,
            fraction=0.1,
            pad=0.1,
            aspect=40,
            label="Extreme dry-bulb air temperature 50 years ($t_{db}$) [°C]",
            ticks=bounds,
            orientation="horizontal",
        )
        sns.despine(left=True, bottom=True, right=True, top=True)
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "world-map.png"), dpi=300)
        else:
            plt.show()

    def model_comparison(self, save_fig=True):

        fig_0, ax_0 = plt.subplots(4, 2, figsize=(7, 8), sharex="all", sharey="row")
        fig_1, ax_1 = plt.subplots(2, 2, figsize=(7, 7), sharex="all")

        index_color = 0

        legend_labels = []

        for v in [0.2, 4.5]:

            for rh in [30, 60]:

                max_skin_wettedness = fan_use_set(
                    50, 50, v, 100, 1.2, 0.5, wme=0, units="SI"
                )["skin_wetness"]

                dry_heat_loss = []
                dry_heat_loss_ollie = []
                sensible_skin_heat_loss = []
                sensible_skin_heat_loss_ollie = []
                sweat_rate = []
                sweat_rate_ollie = []
                max_latent_heat_loss = []
                max_latent_heat_loss_ollie = []
                skin_wettedness = []

                color = self.colors[index_color]
                index_color += 1

                for ta in self.ta_range:

                    r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                    dry_heat_loss.append(r["hl_dry"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    sweat_rate.append(r["sweating_required"])
                    max_latent_heat_loss.append(
                        r["hl_evaporation_max"] * max_skin_wettedness
                    )
                    skin_wettedness.append(r["p_wet"])

                    if v > 1:
                        fan_on = True
                    else:
                        fan_on = False

                    r = ollie(fan_on, ta, rh, is_elderly=False)

                    dry_heat_loss_ollie.append(r["hl_dry"])
                    sensible_skin_heat_loss_ollie.append(r["e_req_w"])

                    sweat_rate_ollie.append(r["s_req"])
                    max_latent_heat_loss_ollie.append(r["e_max_w"])

                sweat_rate_ollie = [x if x > 0 else np.nan for x in sweat_rate_ollie]

                sensible_skin_heat_loss_ollie = [
                    x if x > 0 else np.nan for x in sensible_skin_heat_loss_ollie
                ]

                label = f"v = {v}m/s; RH = {rh}%;"

                ax_0[0][0].plot(self.ta_range, dry_heat_loss, color=color, label=label)
                ax_0[0][1].plot(
                    self.ta_range, dry_heat_loss_ollie, color=color, label=label
                )
                ax_0[1][0].plot(
                    self.ta_range, sensible_skin_heat_loss, color=color, label=label
                )
                ax_0[1][1].plot(
                    self.ta_range,
                    sensible_skin_heat_loss_ollie,
                    color=color,
                    label=label,
                )

                ax_0[2][0].plot(self.ta_range, sweat_rate, color=color, label=label)
                ax_0[2][1].plot(
                    self.ta_range, sweat_rate_ollie, color=color, label=label
                )
                ax_0[3][0].plot(
                    self.ta_range, max_latent_heat_loss, color=color, label=label
                )
                ax_0[3][1].plot(
                    self.ta_range, max_latent_heat_loss_ollie, color=color, label=label
                )

                ax_1[0][0].plot(self.ta_range, dry_heat_loss, color=color, label=label)
                ax_1[0][0].plot(
                    self.ta_range, dry_heat_loss_ollie, color=color, linestyle="-.",
                )
                ax_1[0][1].plot(
                    self.ta_range, skin_wettedness, color=color, label=label
                )

                ax_1[1][1].plot(
                    self.ta_range,
                    gaussian_filter1d(sweat_rate, sigma=2),
                    color=color,
                    label=label,
                )
                ax_1[1][1].plot(
                    self.ta_range, sweat_rate_ollie, color=color, linestyle="-.",
                )
                ax_1[1][0].plot(
                    self.ta_range,
                    gaussian_filter1d(max_latent_heat_loss, sigma=2),
                    color=color,
                    label=label + " Gagge et al. (1986)",
                )
                ax_1[1][0].plot(
                    self.ta_range,
                    max_latent_heat_loss_ollie,
                    color=color,
                    label=label + " Jay et al. (2015)",
                    linestyle="-.",
                )

                legend_labels.append(label)
                legend_labels.append(label)

        ax_0[0][0].set(
            ylim=(-250, 200),
            title="SET",
            ylabel="Sensible heat loss skin (C + R) (W/m$^2$)",
        )
        ax_0[0][1].set(ylim=(-250, 200), title="Ollie")
        ax_0[1][0].set(ylim=(0, 300), ylabel="Required latent heat loss (W/m$^2$)")
        ax_0[1][1].set(ylim=(0, 300))

        ax_0[2][0].set(ylim=(0, 550), ylabel=r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]")
        ax_0[2][1].set(ylim=(0, 550))
        ax_0[3][0].set(
            ylim=(0, 600), xlabel="Temperature", ylabel="Maximum latent heat loss (W)"
        )
        ax_0[3][1].set(ylim=(0, 600), xlabel="Temperature")

        for x in range(0, 4):
            ax_0[x][0].grid()
            ax_0[x][1].grid()

        ax_0[0][0].legend()
        fig_0.tight_layout()
        if save_fig:
            fig_0.savefig(
                os.path.join(self.dir_figures, "comparison_models.png"), dpi=300
            )
        else:
            fig_0.show()

        plt.close(fig_0)

        ax_1[0][0].set(
            ylim=(-201, 100), ylabel="Sensible heat loss skin (C + R) [W/m$^2$]"
        )
        ax_1[0][1].set(ylim=(-0.01, 0.7), ylabel="Skin wettendess (w)")
        ax_1[1][1].set(
            ylim=(-1, 600),
            xlabel=r"Operative temperature ($t_{o}$) [°C]",
            ylabel=r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]",
        )
        ax_1[1][0].set(
            ylim=(-1, 200),
            xlabel=r"Operative temperature ($t_{o}$) [°C]",
            ylabel="Max latent heat loss ($E_{max,w_{max}}$) [W/m$^{2}$]",
        )

        for x in range(0, 2):
            ax_1[x][0].grid(c="lightgray")
            ax_1[x][1].grid(c="lightgray")
            ax_1[x][0].xaxis.set_ticks_position("none")
            ax_1[x][1].xaxis.set_ticks_position("none")
            ax_1[x][0].yaxis.set_ticks_position("none")
            ax_1[x][1].yaxis.set_ticks_position("none")

        ax_1[0][0].text(
            -0.2, 0.97, "A", size=12, ha="center", transform=ax_1[0][0].transAxes
        )
        ax_1[0][1].text(
            -0.2, 0.97, "B", size=12, ha="center", transform=ax_1[0][1].transAxes
        )
        ax_1[1][0].text(
            -0.2, 0.97, "C", size=12, ha="center", transform=ax_1[1][0].transAxes
        )
        ax_1[1][1].text(
            -0.2, 0.97, "D", size=12, ha="center", transform=ax_1[1][1].transAxes
        )

        lines, labels = fig_1.axes[2].get_legend_handles_labels()
        fig_1.legend(
            lines,
            labels,
            loc="upper center",  # Position of legend
            ncol=2,
            frameon=False,
        )
        sns.despine(left=True, bottom=True, right=True)
        fig_1.tight_layout()
        plt.subplots_adjust(top=0.88)
        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "comparison_models_v2.png"), dpi=300
            )
        else:
            plt.show()

    def figure_2(self, save_fig=True):

        fig_1, ax_1 = plt.subplots(2, 2, figsize=(7, 7), sharex="all")

        index_color = 0

        legend_labels = []

        for v in [0.2, 4.5]:

            for rh in [30, 60]:

                energy_balance = []
                sensible_skin_heat_loss = []
                tmp_core = []
                temp_skin = []
                skin_wettedness = []

                color = self.colors[index_color]
                index_color += 1

                for ta in self.ta_range:

                    r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                    energy_balance.append(r["energy_balance"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    tmp_core.append(r["temp_core"])
                    temp_skin.append(r["temp_skin"])
                    skin_wettedness.append(r["p_wet"])

                label = f"v = {v}m/s; RH = {rh}%"

                ax_1[0][0].plot(
                    self.ta_range,
                    gaussian_filter1d(energy_balance, sigma=2),
                    color=color,
                    label=label,
                )
                ax_1[0][1].plot(
                    self.ta_range,
                    gaussian_filter1d(sensible_skin_heat_loss, sigma=2),
                    color=color,
                    label=label,
                )

                ax_1[1][1].plot(
                    self.ta_range,
                    gaussian_filter1d(tmp_core, sigma=2),
                    color=color,
                    label=label,
                )
                ax_1[1][0].plot(
                    self.ta_range,
                    gaussian_filter1d(temp_skin, sigma=3),
                    color=color,
                    label=label,
                )

                legend_labels.append(label)
                legend_labels.append(label)

        ax_1[0][0].set(
            ylim=(-1, 200), ylabel="Excess heat stored ($S_{cr} + S_{sk}$) [W/m$^2$]"
        )
        ax_1[0][1].set(
            ylim=(-1, 160), ylabel="Evaporative heat loss skin ($E_{sk}$) [W/m$^2$]"
        )
        ax_1[1][1].set(
            ylim=(31.9, 42),
            xlabel=r"Operative temperature ($t_{o}$) [°C]",
            ylabel=r"Core mean temperature ($t_{cr}$) [°C]",
        )
        ax_1[1][0].set(
            ylim=(31.9, 42),
            xlabel=r"Operative temperature ($t_{o}$) [°C]",
            ylabel="Skin mean temperature ($t_{sk}$) [°C]",
        )

        for x in range(0, 2):
            ax_1[x][0].grid(c="lightgray")
            ax_1[x][1].grid(c="lightgray")
            ax_1[x][0].xaxis.set_ticks_position("none")
            ax_1[x][1].xaxis.set_ticks_position("none")
            ax_1[x][0].yaxis.set_ticks_position("none")
            ax_1[x][1].yaxis.set_ticks_position("none")

        ax_1[0][0].text(
            -0.2, 0.97, "A", size=12, ha="center", transform=ax_1[0][0].transAxes
        )
        ax_1[0][1].text(
            -0.2, 0.97, "B", size=12, ha="center", transform=ax_1[0][1].transAxes
        )
        ax_1[1][0].text(
            -0.2, 0.97, "C", size=12, ha="center", transform=ax_1[1][0].transAxes
        )
        ax_1[1][1].text(
            -0.2, 0.97, "D", size=12, ha="center", transform=ax_1[1][1].transAxes
        )

        lines, labels = fig_1.axes[2].get_legend_handles_labels()
        fig_1.legend(
            lines,
            labels,
            loc="upper center",  # Position of legend
            ncol=2,
            frameon=False,
        )
        sns.despine(left=True, bottom=True, right=True)
        fig_1.tight_layout()
        plt.subplots_adjust(top=0.92)
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "results_model_2.png"), dpi=300)
        else:
            plt.show()

    def comparison_air_speed(self, save_fig):

        fig, ax = plt.subplots(figsize=(7, 4))

        # plot enthalpy line
        reference_enthalpies = [100805.98, 73007.24]
        for enthalpy in reference_enthalpies:
            rh_const_enthalpy = []
            for tmp in self.ta_range:
                hr = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(enthalpy, tmp)
                rh_const_enthalpy.append(
                    psychrolib.GetRelHumFromHumRatio(tmp, hr, 101325) * 100
                )

            ax.plot(
                rh_const_enthalpy,
                self.ta_range,
                c="k",
                linestyle=":",
                label="v = 2.0m/s; Morris et al. (2019)",
            )

        ax.scatter(
            15,
            47,
            c="tab:red",
            label="fan (2.0m/s) not beneficial; Morris et al. (2019)",
        )
        ax.text(
            27.1, 50, "H = 101 kJ/kg", ha="center", va="center", size=8, rotation=-46
        )

        ax.scatter(
            50, 40, c="tab:green", label="fan (2.0m/s) beneficial; Morris et al. (2019)"
        )
        ax.text(
            60, 31.35, "H = 73 kJ/kg", ha="center", va="center", size=8, rotation=-24
        )

        heat_strain = {}

        for ix, v in enumerate(self.v_range):

            heat_strain[v] = {}

            color = self.colors_f3[ix]

            for rh in np.arange(0, 105, 1):

                for ta in np.arange(28, 66, 0.25):

                    r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                    # determine critical temperature at which heat strain would occur
                    # if r["p_wet"] >= r["w_max"]:
                    if r["exceeded"]:
                        heat_strain[v][rh] = ta
                        break

            # plot Jay's data
            if v in self.heat_strain_ollie.keys():
                ax.plot(
                    self.heat_strain_ollie[v].keys(),
                    self.heat_strain_ollie[v].values(),
                    linestyle="-.",
                    label=f"v = {v}; Jay et al. (2015)",
                    c=color,
                )

            x = list(heat_strain[v].keys())

            y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

            heat_strain[v] = {}
            for x_val, y_val in zip(x, y_smoothed):
                heat_strain[v][x_val] = y_val

            ax.plot(
                x, y_smoothed, label=f"v = {v}m/s; Gagge et al. (1986)", c=color,
            )

        np.save(os.path.join("code", "heat_strain.npy"), heat_strain)

        ax.grid(c="lightgray")

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel="Relative Humidity ($RH$) [%]",
            ylabel="Operative temperature ($t_{o}$) [°C]",
            ylim=(28, 55),
            xlim=(-1, 100),
        )

        sns.despine(left=True, bottom=True, right=True)

        plt.legend(frameon=False)

        plt.tight_layout()
        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "comparison_air_speed.png"), dpi=300
            )
        else:
            plt.show()

    def met_clo(self, save_fig):

        fig, ax = plt.subplots(figsize=(7, 4))

        heat_strain = {}

        combinations = [
            {"clo": 0.36, "met": 1, "ls": "dashed"},
            {"clo": 0.5, "met": 1, "ls": "dotted"},
            {"clo": 0.36, "met": 1.2, "ls": "dashdot"},
            {"clo": 0.5, "met": 1.2, "ls": "solid"},
            # {"clo": 0, "met": 0.8, "ls": (0, (3, 1, 1, 1, 1, 1))},
        ]

        for combination in combinations:

            clo = combination["clo"]
            met = combination["met"]
            ls = combination["ls"]

            for ix, v in enumerate([0.2, 0.8]):

                heat_strain[v] = {}

                color = self.colors_f3[ix]

                for rh in np.arange(0, 105, 1):

                    for ta in np.arange(28, 66, 0.25):

                        r = fan_use_set(ta, ta, v, rh, met, clo, wme=0, units="SI")

                        if r["exceeded"]:
                            heat_strain[v][rh] = ta
                            break

                x = list(heat_strain[v].keys())

                y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

                ax.plot(
                    x,
                    y_smoothed,
                    label=f"v = {v}, clo = {clo}, met = {met}",
                    c=color,
                    linestyle=ls,
                )

        ax.grid(c="lightgray")

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel="Relative Humidity ($RH$) [%]",
            ylabel="Operative temperature ($t_{o}$) [°C]",
            ylim=(28, 55),
            xlim=(-1, 100),
        )

        sns.despine(left=True, bottom=True, right=True)

        plt.legend(frameon=False)

        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "met_clo.png"), dpi=300)
        else:
            plt.show()

    def plot_other_variables(self, variable, levels_cbar):

        v_range = [0.2, 4.5]

        f, ax = plt.subplots(len(v_range), 1, sharex="all", sharey="all")

        df_comparison = pd.DataFrame()

        for ix, v in enumerate(v_range):

            tmp_array = []
            rh_array = []
            variable_arr = []

            for rh in self.rh_range:

                for ta in self.ta_range:

                    tmp_array.append(ta)
                    rh_array.append(rh)

                    r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                    variable_arr.append(r[variable])

            # dataframe used to plot the two contour plots
            x, y = np.meshgrid(self.rh_range, self.ta_range)

            variable_arr = [x if x > 1 else 0 for x in variable_arr]
            df = pd.DataFrame({"tmp": tmp_array, "rh": rh_array, "z": variable_arr})
            df_comparison[f"index_{ix}"] = variable_arr
            df_w = df.pivot("tmp", "rh", "z")
            cf = ax[ix].contourf(x, y, df_w.values, levels_cbar)

            ax[ix].set(
                xlabel="Relative Humidity",
                ylabel="Air Temperature",
                title=f"{variable} - air speed {v}",
            )

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(cf, cax=cbar_ax)
        plt.show()

        # find point above which heat gains are higher with fans
        df_comparison.describe()

        df_comparison["delta"] = df_comparison["index_1"] - df_comparison["index_0"]
        df_comparison["tmp"] = df["tmp"]
        df_comparison["rh"] = df["rh"]
        df_w = df_comparison.pivot("tmp", "rh", "delta")

        f, ax = plt.subplots(constrained_layout=True)
        origin = "lower"
        levels = [-0.5, 0, 0.5]
        cf = ax.contourf(
            x, y, df_w.values, levels, colors=("b", "r"), extend="both", origin=origin
        )
        cf.cmap.set_under("darkblue")
        cf.cmap.set_over("maroon")
        ax.contour(x, y, df_w.values, [-0.5, 0, 0.5], colors=("k",), origin=origin)

        ax.set(
            xlabel="Relative Humidity",
            ylabel="Air Temperature",
            title=f"{variable} difference at v {v_range[1]} - {v_range[0]} [m/s]",
        )

        f.colorbar(cf)
        plt.show()

    def summary_use_fans(self, save_fig):
        rh_arr = np.arange(34, 110, 2)
        tmp_low = []
        tmp_high = []

        for rh in rh_arr:

            def function(x):
                return (
                    fan_use_set(x, x, v, rh, 1.2, 0.5, wme=0, units="SI")["temp_core"]
                    - fan_use_set(x, x, 0.2, rh, 1.2, 0.5, wme=0, units="SI")[
                        "temp_core"
                    ]
                )

            v = 4.5
            try:
                tmp_high.append(optimize.brentq(function, 30, 130))
            except ValueError:
                tmp_high.append(np.nan)

            v = 0.8
            try:
                tmp_low.append(optimize.brentq(function, 30, 130))
            except ValueError:
                tmp_low.append(np.nan)

        fig, ax = plt.subplots()

        # plot heat strain lines
        for key in self.heat_strain.keys():
            if key == 0.2:
                (ln_2,) = ax.plot(
                    list(self.heat_strain[key].keys()),
                    list(self.heat_strain[key].values()),
                    c="k",
                    label="v = 0.2 m/s",
                )
                f_02_critical = np.poly1d(
                    np.polyfit(
                        list(self.heat_strain[key].keys()),
                        list(self.heat_strain[key].values()),
                        2,
                    )
                )
            if key == 0.8:
                ax.plot(
                    list(self.heat_strain[key].keys()),
                    list(self.heat_strain[key].values()),
                    c="k",
                    linestyle="-.",
                )
                f_08_critical = np.poly1d(
                    np.polyfit(
                        list(self.heat_strain[key].keys()),
                        list(self.heat_strain[key].values()),
                        2,
                    )
                )
            if key == 4.5:
                ax.plot(
                    list(self.heat_strain[key].keys()),
                    list(self.heat_strain[key].values()),
                    c="k",
                    linestyle=":",
                )
                f_45_critical = np.poly1d(
                    np.polyfit(
                        list(self.heat_strain[key].keys()),
                        list(self.heat_strain[key].values()),
                        2,
                    )
                )

        x_new, y_new = interpolate(rh_arr, tmp_low)

        (ln_0,) = ax.plot(x_new, y_new, c="k", linestyle="-.", label="v = 0.8 m/s")

        fb_0 = ax.fill_between(
            x_new,
            y_new,
            100,
            color="tab:red",
            alpha=0.2,
            zorder=100,
            label="No fan - v = 4.5 m/s",
        )

        df = pd.DataFrame(self.heat_strain)
        df_fan_above = df[df[4.5] >= df[0.8]]
        df_fan_below = df[df[4.5] <= df[0.8] + 0.073]

        x_new, y_new = interpolate(
            rh_arr, tmp_high, x_new=list(df_fan_above.index.values)
        )
        (ln_1,) = ax.plot(x_new, y_new, c="k", label="v = 4.5 m/s")

        ax.fill_between(
            x_new,
            df_fan_above[4.5].values,
            y_new,
            facecolor="none",
            zorder=100,
            hatch="/",
            edgecolor="silver",
        )

        fb_1 = ax.fill_between(
            x_new,
            y_new,
            100,
            color="tab:orange",
            alpha=0.2,
            label="No fan - v = 4.5 m/s",
        )

        # green part on the right
        fb_2 = ax.fill_between(x_new, 0, y_new, color="tab:green", alpha=0.2)

        # green part below evaporative cooling
        ax.fill_between(
            df_fan_below.index,
            0,
            df_fan_below[4.5].values,
            color="tab:green",
            alpha=0.2,
        )

        # blue part below evaporative cooling
        ax.fill_between(
            df_fan_below.index,
            df_fan_below[4.5].values,
            100,
            color="tab:blue",
            alpha=0.2,
        )

        ax.set(
            ylim=(29, 55),
            xlim=(0, 100),
            xlabel=r"Relative Humidity ($RH$) [%]",
            ylabel=r"Operative temperature ($t_{o}$) [°C]",
        )
        ax.text(
            10, 37.5, "Use fans", size=12, ha="center", va="center",
        )
        ax.text(
            0.85,
            0.75,
            "Do not\nuse fans",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )
        # ax.text(
        #     0.33,
        #     0.8,
        #     "Move to a\ncooler place\nif possible",
        #     size=12,
        #     zorder=200,
        #     ha="center",
        #     transform=ax.transAxes,
        # )
        ax.text(
            9.5,
            53.5,
            "Evaporative\ncooling",
            size=12,
            zorder=200,
            ha="center",
            va="center",
        )
        text_dic = [
            {"txt": "Thermal strain\nv =0.2m/s", "x": 11, "y": 46.5, "r": -48},
            # {"txt": "Thermal strain\nv=0.8m/s", "x": 8.3, "y": 52, "r": -47},
            {"txt": "Thermal strain\nv=4.5m/s", "x": 93, "y": 33.5, "r": -20},
            {"txt": "No fans, v=4.5m/s", "x": 80, "y": 39, "r": -15},
            {"txt": "No fans, v=0.8m/s", "x": 80, "y": 41.5, "r": -24},
        ]

        for obj in text_dic:
            ax.text(
                obj["x"],
                obj["y"],
                obj["txt"],
                size=8,
                ha="center",
                va="center",
                rotation=obj["r"],
                zorder=200,
                # bbox=dict(boxstyle="round", ec=(0, 0, 0, 0), fc=(1, 1, 1, 0.5),),
            )

        # plot extreme weather events
        df_queried = pd.read_sql(
            "SELECT wmo, "
            '"n-year_return_period_values_of_extreme_DB_10_max" as db_max, '
            '"n-year_return_period_values_of_extreme_WB_10_max" as wb_max '
            "FROM data",
            con=self.conn,
        )

        arr_rh = []
        df_queried[["db_max", "wb_max"]] = df_queried[["db_max", "wb_max"]].apply(
            pd.to_numeric, errors="coerce"
        )
        df_queried.dropna(inplace=True)
        for ix, row in df_queried.iterrows():
            arr_rh.append(
                psychrolib.GetRelHumFromTWetBulb(row["db_max"], row["wb_max"], 101325)
                * 100
            )

        # calculate number of stations where db_max exceeds critical temperature
        f_08_no_fan = np.poly1d(np.polyfit(rh_arr, tmp_low, 2,))
        f_45_no_fan = np.poly1d(np.polyfit(rh_arr, tmp_high, 2,))

        df_queried["rh"] = arr_rh
        df_queried["t_crit_02"] = [f_02_critical(x) for x in arr_rh]
        df_queried["t_crit_08"] = [f_08_critical(x) for x in arr_rh]
        df_queried["t_crit_45"] = [f_45_critical(x) for x in arr_rh]
        df_queried["t_no_fan_08"] = [f_08_no_fan(x) for x in arr_rh]
        df_queried["t_no_fan_45"] = [f_45_no_fan(x) for x in arr_rh]
        df_queried["exc_t_crit_02"] = 0
        df_queried["exc_t_crit_08"] = 0
        df_queried["exc_t_crit_45"] = 0
        df_queried["exc_no_fan_08"] = 0
        df_queried["exc_no_fan_45"] = 0
        df_queried.loc[
            df_queried["t_crit_02"] < df_queried["db_max"], "exc_t_crit_02"
        ] = 1
        df_queried.loc[
            df_queried["t_crit_08"] < df_queried["db_max"], "exc_t_crit_08"
        ] = 1
        df_queried.loc[
            df_queried["t_crit_45"] < df_queried["db_max"], "exc_t_crit_45"
        ] = 1
        df_queried.loc[
            df_queried["t_no_fan_08"] < df_queried["db_max"], "exc_no_fan_08"
        ] = 1
        df_queried.loc[
            df_queried["t_no_fan_45"] < df_queried["db_max"], "exc_no_fan_45"
        ] = 1
        print(
            (
                100
                - df_queried[
                    [
                        "exc_t_crit_02",
                        "exc_t_crit_08",
                        "exc_t_crit_45",
                        "exc_no_fan_08",
                        "exc_no_fan_45",
                    ]
                ].sum()
                / df_queried.shape[0]
                * 100
            ).round()
        )
        print(
            df_queried[
                [
                    "exc_t_crit_02",
                    "exc_t_crit_08",
                    "exc_t_crit_45",
                    "exc_no_fan_08",
                    "exc_no_fan_45",
                ]
            ].sum()
        )

        ax.scatter(df_queried["rh"], df_queried["db_max"], s=3, c="tab:gray")

        # add legend
        plt.legend(
            [ln_2, ln_0, ln_1, fb_0, fb_1, fb_2],
            [
                "v = 0.2 m/s",
                "v = 0.8 m/s",
                "v = 4.5 m/s",
                "No fans - v = 0.8 m/s",
                "No fans - v = 4.5 m/s",
                "Use fans",
            ],
            loc="lower left",
            ncol=2,
            # frameon=False,  # Position of legend
            facecolor="w",
        )
        fig.tight_layout()
        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout()

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "use_fans.png"), dpi=300)
        else:
            plt.show()


def ollie(is_fan_on, ta, rh, is_elderly):

    met = 65
    work = 0
    emissivity = 1
    boltzmann_const = 5.67 * 10 ** -8
    a_r_bsa = 0.7
    t_sk = 35.5
    s_w_lat = 2426

    if is_fan_on:
        r_cl_f = 0.0497
        r_cl_r = 0.0844

        # todo this is not correct need to use Kerslake 1972
        r_cl_mean = (r_cl_f + r_cl_r) / 2

        v = 4.5

        if is_elderly:
            w = 0.5
        else:
            w = 0.65

        # todo choose r_cl based on fan on or off
        r_e_cl_f = 0.0112
        r_e_cl_r = 0.0161

        # todo this is not correct
        r_e_cl_mean = (r_e_cl_f + r_e_cl_r) / 2

    else:

        r_cl_mean = 0.1291
        v = 0.2

        if is_elderly:
            w = 0.65
        else:
            w = 0.85

        r_e_cl_mean = 0.0237

    bsa = 1.8
    t_r = ta
    t_o = ta
    p_a = p_sat(ta) / 1000 * rh / 100
    p_sk_s = p_sat(t_sk) / 1000

    h_c = 8.3 * v ** 0.6

    h_r = 4 * emissivity * boltzmann_const * a_r_bsa * (273.2 + (t_sk + t_r) / 2) ** 3

    h = h_c + h_r

    f_cl = 1 + 0.31 * r_cl_mean / 0.155

    c_r = (t_sk - t_o) / (r_cl_mean + 1 / (f_cl * h))

    c_res_e_res = 0.0014 * met * (34 - ta) + 0.0173 * met * (5.87 - p_a)

    # amount of heat that needs to be loss via evaporation
    e_req = met - work - c_r - c_res_e_res

    # evaporative heat transfer coefficient
    h_e = 16.5 * h_c

    e_max = w * (p_sk_s - p_a) / (r_e_cl_mean + 1 / (f_cl * h_e))

    w_req = e_req / e_max

    s_w_eff = 1 - (w_req ** 2) / 2

    s_req = (e_req / s_w_eff * 3600) / s_w_lat

    # e_req_w = e_req * bsa
    # e_max_w = e_max * bsa
    # c_r_w = c_r * bsa

    return {
        "e_req_w": e_req,
        "e_max_w": e_max,
        "hl_dry": c_r,
        "s_req": s_req,
        "w_req": w_req,
    }


def fan_use_set(
    tdb, tr, v, rh, met, clo, wme=0, body_surface_area=1.8258, patm=101325, units="SI",
):
    """
    Calculates the Standard Effective Temperature (SET). The SET is the temperature of
    an imaginary environment at 50% (rh), <0.1 m/s (20 fpm) average air speed (v),
    and tr = tdb ,
    in which the total heat loss from the skin of an imaginary occupant with an
    activity level of 1.0 met and a clothing level of 0.6 clo is the same as that
    from a person in the actual environment with actual clothing and activity level.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float
        air velocity, default in [m/s] in [fps] if `units` = 'IP'
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0
    body_surface_area : float
        body surface area, default value 1.8258 [m2] in [ft2] if `units` = 'IP'
    patm : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    SET : float
        Standard effective temperature, [°C]

    Notes
    -----
    You can use this function to calculate the `SET`_ temperature in accordance with
    the ASHRAE 55 2017 Standard [1]_.

    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import set_tmp
        >>> set_tmp(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        25.3

        >>> # for users who wants to use the IP system
        >>> set_tmp(tdb=77, tr=77, v=0.328, rh=50, met=1.2, clo=.5, units='IP')
        77.6

    """
    if units.lower() == "ip":
        if body_surface_area == 1.8258:
            body_surface_area = 19.65
        if patm == 101325:
            patm = 1
        tdb, tr, v, body_surface_area, patm = units_converter(
            tdb=tdb, tr=tr, v=v, area=body_surface_area, pressure=patm
        )

    # check_standard_compliance(
    #     standard="ashrae", tdb=tdb, tr=tr, v=v, rh=rh, met=met, clo=clo
    # )

    vapor_pressure = rh * p_sat_torr(tdb) / 100

    # check if reached maximum values
    exc_blood_flow = False
    exc_rgsw = False
    exc_pwet = False

    # Initial variables as defined in the ASHRAE 55-2017
    air_velocity = max(v, 0.1)
    k_clo = 0.25
    body_weight = 69.9
    met_factor = 58.2
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 120  # driving coefficient for vasodilation
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    temp_body_neutral = 36.49
    skin_blood_flow_neutral = 6.3

    temp_skin = temp_skin_neutral
    temp_core = temp_core_neutral
    skin_blood_flow = skin_blood_flow_neutral
    alfa = 0.1  # fractional skin mass
    e_sk = 0.1 * met  # total evaporative heat loss, W

    pressure_in_atmospheres = patm / 101325
    length_time_simulation = 60  # length time simulation
    r_clo = 0.155 * clo  # thermal resistance of clothing, °C M^2 /W

    f_a_cl = (
        1.0 + 0.15 * clo
    )  # increase in body surface area due to clothing todo the eq 49 fundamentals is 1 + 0.3 icl
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = met * met_factor  # metabolic rate
    m = met * met_factor

    if clo <= 0:
        w_crit = 0.38 * pow(air_velocity, -0.29)  # evaporative efficiency
        i_cl = 1.0  # thermal resistance of clothing, clo
    else:
        w_crit = 0.59 * pow(air_velocity, -0.08)
        i_cl = 0.45

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 °C)
    h_fc = 8.600001 * pow((air_velocity * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)

    c_hr = 4.7  # linearized radiative heat transfer coefficient
    CTC = c_hr + h_cc
    r_a = 1.0 / (f_a_cl * CTC)  # resistance of air layer to dry heat
    t_op = (c_hr * tr + h_cc * tdb) / CTC  # operative temperature

    # initialize some variables
    dry = 0
    p_wet = 0
    _set = 0

    for i in range(length_time_simulation):

        iteration_limit = 150
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * temp_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:

            c_hr = 4.0 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.72
            CTC = c_hr + h_cc
            r_a = 1.0 / (f_a_cl * CTC)
            t_op = (c_hr * tr + h_cc * tdb) / CTC
            t_cl_new = (r_a * temp_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations exceeded")

        dry = (temp_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W
        # h_fcs rate of energy transport between core and skin, W
        h_fcs = (temp_core - temp_skin) * (5.28 + 1.163 * skin_blood_flow)
        q_res = 0.0023 * m * (44.0 - vapor_pressure)  # heat loss due to respiration
        CRES = 0.0014 * m * (34.0 - tdb)
        s_core = m - h_fcs - q_res - CRES - wme  # rate of energy storage in the core
        s_skin = h_fcs - dry - e_sk  # rate of energy storage in the skin
        TCSK = 0.97 * alfa * body_weight
        TCCR = 0.97 * (1 - alfa) * body_weight
        DTSK = (s_skin * body_surface_area) / (TCSK * 60.0)  # °C per minute
        DTCR = s_core * body_surface_area / (TCCR * 60.0)
        temp_skin = temp_skin + DTSK
        temp_core = temp_core + DTCR
        t_body = alfa * temp_skin + (1 - alfa) * temp_core  # mean body temperature, °C
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = temp_skin - temp_skin_neutral
        warms = (sk_sig > 0) * sk_sig  # vasodialtion signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the skin, °C
        c_reg_sig = temp_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        BDSIG = t_body - temp_body_neutral
        WARMB = (BDSIG > 0) * BDSIG
        skin_blood_flow = (skin_blood_flow_neutral + c_dil * c_warm) / (
            1 + c_str * colds
        )
        if skin_blood_flow > 90.0:
            skin_blood_flow = 90.0
            exc_blood_flow = True
        if skin_blood_flow < 0.5:
            skin_blood_flow = 0.5
        REGSW = c_sw * WARMB * math.exp(warms / 10.7)
        if REGSW > 500.0:
            REGSW = 500.0
            exc_rgsw = True
        e_rsw = 0.68 * REGSW  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        # e_max = maximum evaporative capacity
        e_max = (
            math.exp(18.6686 - 4030.183 / (temp_skin + 235.0)) - vapor_pressure
        ) / (r_ea + r_ecl)
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        p_wet = 0.06 + 0.94 * p_rsw  # skin wetness
        e_diff = p_wet * e_max - e_rsw  # vapor diffusion through skin
        if p_wet > w_crit:
            p_wet = w_crit
            p_rsw = w_crit / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
            exc_pwet = True
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            p_wet = w_crit
        e_sk = (
            e_rsw + e_diff
        )  # total evaporative heat loss sweating and vapor diffusion
        MSHIV = 19.4 * colds * c_cold
        m = rm + MSHIV
        alfa = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417)

    hsk = dry + e_sk  # total heat loss from skin, W
    W = p_wet
    PSSK = math.exp(18.6686 - 4030.183 / (temp_skin + 235.0))
    CHRS = c_hr
    if met < 0.85:
        CHCS = 3.0
    else:
        CHCS = 5.66 * (met - 0.85) ** 0.39
    if CHCS < 3.0:
        CHCS = 3.0
    CTCS = CHCS + CHRS
    RCLOS = 1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835
    RCLS = 0.155 * RCLOS
    FACLS = 1.0 + k_clo * RCLOS
    FCLS = 1.0 / (1.0 + 0.155 * FACLS * CTCS * RCLOS)
    IMS = 0.45
    ICLS = IMS * CHCS / CTCS * (1 - FCLS) / (CHCS / CTCS - FCLS * IMS)
    RAS = 1.0 / (FACLS * CTCS)
    REAS = 1.0 / (lr * FACLS * CHCS)
    RECLS = RCLS / (lr * ICLS)
    HD_S = 1.0 / (RAS + RCLS)
    HE_S = 1.0 / (REAS + RECLS)

    delta = 0.0001
    dx = 100.0
    set_old = round(temp_skin - hsk / HD_S, 2)
    while abs(dx) > 0.01:
        err_1 = (
            hsk
            - HD_S * (temp_skin - set_old)
            - W
            * HE_S
            * (PSSK - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + 235.0))))
        )
        err_2 = (
            hsk
            - HD_S * (temp_skin - (set_old + delta))
            - W
            * HE_S
            * (PSSK - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + delta + 235.0))))
        )
        _set = set_old - delta * err_1 / (err_2 - err_1)
        dx = _set - set_old
        set_old = _set

    return {
        # "set": _set,
        "hl_evaporation_required": e_sk,
        "hl_evaporation_max": e_max,
        "hl_dry": dry,
        # "temp_skin": temp_skin,
        # "t_clothing": t_cl,
        # "heat lost by vaporization sweat": e_rsw,
        "temp_core": temp_core,
        "temp_skin": temp_skin,
        "exceeded": any([exc_blood_flow, exc_pwet, exc_rgsw]),
        "skin_blood_flow": skin_blood_flow,
        "t_body": t_body,
        "warms": warms,
        "skin_blood_flow": skin_blood_flow,
        "sweating_required": REGSW,
        "e_diff": e_diff,
        "skin_wetness": p_wet,
        "energy_storage_core": s_core,
        "energy_balance": m - hsk - q_res,
        "w_max": w_crit,
        "e_rsw": e_rsw,
        "p_wet": p_wet,
        "q_res": q_res,
    }


def interpolate(x, y, x_new=False, order=2):
    f2 = np.poly1d(np.polyfit(x, y, 2))
    if not x_new:
        x_new = np.linspace(0, 100, 100)
    return x_new, f2(x_new)


def analyse_em_data():

    df = pd.read_csv(
        os.path.join(os.getcwd(), "code", "emdat.csv"), header=6, encoding="ISO-8859-1"
    )
    df = df[df["Disaster Subtype"] == "Heat wave"]
    df = df.dropna(subset=["Dis Mag Value"])

    # number of entries
    print("number of entries with max tmp value = ", df.shape[0])

    print("total deaths = ", df["Total Deaths"].sum())

    for t in [40, 45]:
        print(
            "deaths t lower than ",
            t,
            " equal to ",
            df.loc[df["Dis Mag Value"] < t, "Total Deaths"].sum(),
        )


if __name__ == "__main__":

    plt.close("all")

    analyse_em_data()

    #
    self = DataAnalysis()

    # ta = 45
    # rh = 30
    # v = 4.5
    # pprint(fan_use_set(tdb=ta, tr=ta, v=v, rh=rh, met=1.2, clo=0.5, wme=0, units="SI"))
    #
    # for t in np.arange(33, 39, 0.1):
    #     rh = 60
    #     v = 0.1
    #     print(
    #         t,
    #         " ",
    #         fan_use_set(t, t, v, rh, 1.2, 0.5, wme=0, units="SI")["energy_balance"],
    #     )

    # # Figure 1
    # self.model_comparison(save_fig=True)
    #
    # # Figure 2
    # self.figure_2(save_fig=True)
    #
    # # Figure 3
    # self.comparison_air_speed(save_fig=True)
    #
    # Figure 4 - you also need to generate fig 3
    # self.summary_use_fans(save_fig=False)
    #
    # # Figure 3
    # self.met_clo(save_fig=True)
    #
    # # Figure Maps
    # self.plot_map_world(save_fig=True)

    # # benefit of increasing air speed
    # benefit = [
    #     x[0] - x[1]
    #     for x in zip(self.heat_strain[0.8].values(), self.heat_strain[0.2].values())
    # ]
    # pd.DataFrame({"benefit": benefit}).describe().round(1)

    # # Figure 4
    # self.plot_other_variables(
    #     variable="energy_balance", levels_cbar=np.arange(0, 200, 10)
    # )
    # self.plot_other_variables(variable="temp_core", levels_cbar=np.arange(36, 43, 0.5))

    # # # calculations for the introduction section
    # e_dry = np.round(
    #     psychrolib.GetMoistAirEnthalpy(
    #         47, psychrolib.GetHumRatioFromRelHum(47, 0.15, 101325)
    #     )
    #     / 1000
    # )
    #
    # e_humid = np.round(
    #     psychrolib.GetMoistAirEnthalpy(
    #         40, psychrolib.GetHumRatioFromRelHum(40, 0.5, 101325)
    #     )
    #     / 1000
    # )
    #
    # np.round(
    #     psychrolib.GetRelHumFromHumRatio(
    #         40, psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(73007, 40), 101325
    #     )
    #     * 100
    # )

    # plt.close("all")
    # plt.plot()
    # rh_arr = range(0, 120, 20)
    # t_arr = range(30, 55)
    # for ix, v in enumerate([0.2, 4.5]):
    #     linSt = ["-", ":", "-."]
    #     for r in rh_arr:
    #         results = []
    #         for t in t_arr:
    #             results.append(
    #                 fan_use_set(t, t, v=v, rh=r, met=1.2, clo=0.5, wme=0, units="SI")[
    #                     # "p_wet"
    #                     # "skin_blood_flow"
    #                     # "sweating_required"
    #                     "temp_core"
    #                 ]
    #             )
    #         plt.plot(t_arr, results, label=f"{r}, {v}", linestyle=linSt[ix])
    # plt.legend()
    # plt.show()

    # t, rh = 45, 70
    # e = np.round(
    #     psychrolib.GetMoistAirEnthalpy(
    #         t, psychrolib.GetHumRatioFromRelHum(t, rh/100, 101325)
    #     )
    #     / 1000
    # )
    # print(e)

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
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error

from heat_balance_model import use_fans_heatwaves

psychrolib.SetUnitSystem(psychrolib.SI)

plt.style.use("seaborn-paper")


class DataAnalysis:
    def __init__(self):
        self.dir_figures = os.path.join(os.getcwd(), "manuscript", "src", "figures")
        self.dir_tables = os.path.join(os.getcwd(), "manuscript", "src", "tables")

        self.ta_range = np.arange(28, 50, 0.5)
        self.v_range = [0.2, 0.8, 4.5]
        self.rh_range = np.arange(0, 105, 5)

        self.colors = ["#3B7EA1", "#C4820E", "#003262", "#FDB515"]
        self.colors_f3 = ["tab:gray", "#3B7EA1", "#C4820E"]

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

        # labels chart
        self.label_t_op = r"Operative temperature ($t_{o}$) [°C]"

        # define map extent
        self.lllon = -180
        self.lllat = -50
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
            & (df["db_max"] > 26)
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
            label="Extreme dry-bulb air temperature ($t_{db}$) 10 years [°C]",
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

                max_skin_wettedness = use_fans_heatwaves(
                    50, 50, v, 100, 1.1, 0.5, wme=0
                )["skin_wettedness"]

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

                    r = use_fans_heatwaves(ta, ta, v, rh, 1.1, 0.5, wme=0)

                    dry_heat_loss.append(r["hl_dry"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    sweat_rate.append(r["sweating_required"])
                    max_latent_heat_loss.append(
                        r["hl_evaporation_max"] * max_skin_wettedness
                    )
                    skin_wettedness.append(r["skin_wettedness"])

                    if v > 1:
                        fan_on = True
                    else:
                        fan_on = False

                    if rh > 50:
                        lw = 1
                        alpha = 1
                    else:
                        lw = 3
                        alpha = 0.75

                    r = ollie(fan_on, ta, rh, is_elderly=False)

                    dry_heat_loss_ollie.append(r["hl_dry"])
                    sensible_skin_heat_loss_ollie.append(r["e_req_w"])

                    sweat_rate_ollie.append(r["s_req"])
                    max_latent_heat_loss_ollie.append(r["e_max_w"])

                sweat_rate_ollie = [x if x > 0 else np.nan for x in sweat_rate_ollie]

                sensible_skin_heat_loss_ollie = [
                    x if x > 0 else np.nan for x in sensible_skin_heat_loss_ollie
                ]

                label = f"V = {v}m/s; RH = {rh}%;"

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
                    self.ta_range,
                    dry_heat_loss_ollie,
                    color=color,
                    linestyle="--",
                    linewidth=lw,
                    alpha=alpha,
                )
                ax_1[0][1].plot(
                    self.ta_range, skin_wettedness, color=color, label=label
                )
                # ax_1[0][1].plot(
                #     [28, 55],
                #     w_crit_ollie,
                #     color=color,
                #     label=label,
                #     linestyle="--",
                #     linewidth=lw,
                #     alpha=alpha,
                # )

                ax_1[1][1].plot(
                    self.ta_range,
                    gaussian_filter1d(sweat_rate, sigma=2),
                    color=color,
                    label=label,
                )
                ax_1[1][1].plot(
                    self.ta_range, sweat_rate_ollie, color=color, linestyle="--",
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
                    linestyle="--",
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
        ax_1[0][1].set(ylim=(-0.01, 0.9), ylabel="Skin wettendess (w)")
        ax_1[1][1].set(
            ylim=(-1, 600),
            xlabel=self.label_t_op,
            ylabel=r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]",
        )
        ax_1[1][0].set(
            ylim=(-1, 200),
            xlabel=self.label_t_op,
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

                    r = use_fans_heatwaves(ta, ta, v, rh, 1.1, 0.5, wme=0)

                    energy_balance.append(r["energy_balance"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    tmp_core.append(r["temp_core"])
                    temp_skin.append(r["temp_skin"])
                    skin_wettedness.append(r["skin_wettedness"])

                label = f"V = {v}m/s; RH = {rh}%"

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
            ylim=(34.9, 40),
            xlabel=self.label_t_op,
            ylabel=r"Core mean temperature ($t_{cr}$) [°C]",
        )
        ax_1[1][0].set(
            ylim=(34.9, 40),
            xlabel=self.label_t_op,
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

    def sweat_rate_production(self, save_fig=True):

        air_speeds = [0.2, 0.8]

        results = []
        for v in air_speeds:
            for ta in np.arange(30, 50, 2):
                for rh in np.arange(10, 100, 10):
                    r = use_fans_heatwaves(ta, ta, v, rh, 1.1, 0.5, wme=0)
                    r["ta"] = ta
                    r["rh"] = rh
                    r["v"] = v
                    results.append(r)

        df_sweat = pd.DataFrame(results)

        fig, axn = plt.subplots(1, 2, sharey=True)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.75])

        for i, ax in enumerate(axn.flat):
            v = air_speeds[i]
            df = df_sweat[df_sweat["v"] == v]
            sns.heatmap(
                df.pivot("ta", "rh", "sweating_required").astype("int"),
                annot=True,
                cbar=i == 0,
                ax=ax,
                fmt="d",
                cbar_ax=None if i else cbar_ax,
                cbar_kws={"label": r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]",},
                annot_kws={"size": 8},
            )
            ax.set(
                xlabel="Relative humidity ($RH$) [%]",
                ylabel="Operative temperature ($t_{o}$) [°C]" if i == 0 else None,
                title=r"$V$" + f" = {v} m/s",
            )

        # cbar_ax.collections[0].colorbar.set_label("Hello")

        fig.tight_layout(rect=[0, 0, 0.88, 1])

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "sweat_rate.png"), dpi=300)
        else:
            plt.show()

    def comparison_ravanelli(self, save_fig=True):

        f, ax = plt.subplots(figsize=(4, 3), sharex="all")

        ravanelli = [
            [20.834201219379903, -0.0007788904784615802],
            [24.0124948469266, 0.0017048164043869107],
            [27.826447199982645, 0.020332618025750815],
            [31.17424982099851, -0.00947186456843152],
            [34.64918418711624, 0.010397790494356629],
            [37.615591572826496, -0.00947186456843152],
            [40.75150795200591, 0.00046296296296266526],
            [43.84504708281804, 0.04020227308853874],
            [47.10809520709932, 0.04020227308853874],
            [49.862616350973134, 0.10850421236687313],
            [52.786646488316094, 0.2115780480050864],
            [55.795431122393644, 0.3307959783818152],
            [58.88897025320578, 0.48230209823557446],
            [61.728245893814154, 0.6499523128278492],
            [64.69465327952443, 0.8225699411858209],
            [67.23728818156178, 1.0411361468764904],
        ]

        df_ravanelli = pd.DataFrame(ravanelli)

        v = 4.0
        ta = 42
        tmp_core = []

        for rh in df_ravanelli[0]:

            r = use_fans_heatwaves(ta, ta, v, rh, 1, 0.35, wme=0)

            tmp_core.append(r["temp_core"] - 37.204)

        # # the above temperature which I am subtracting was calculated using the below eq
        # np.mean([x for x in tmp_core if x < 37.24])

        ax.plot(df_ravanelli[0], tmp_core, label="Gagge et al.(1986)", c="#3B7EA1")

        ax.scatter(
            df_ravanelli[0],
            df_ravanelli[1],
            label="Ravanelli et al. (2015)",
            c="#3B7EA1",
        )

        print(
            f"diffrerences models: "
            f"{pd.Series(tmp_core - df_ravanelli[1].values).describe().round(2)}"
            f"\n mean absolute error: "
            f"{round(mean_absolute_error(tmp_core, df_ravanelli[1].values), 2)}"
        )

        ax.grid(c="lightgray")

        ax.set(
            ylabel=r"Change in core temperature ($t_{cr}$) [°C]",
            xlabel="Relative humidity (RH) [%]",
        )

        plt.legend(frameon=False,)

        sns.despine(left=True, bottom=True, right=True)
        f.tight_layout()
        plt.subplots_adjust(top=0.92)
        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "comparison_ravanelli.png"), dpi=300
            )
        else:
            plt.show()

    def comparison_air_speed(self, save_fig):

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.grid(c="lightgray")

        heat_strain = {}

        for ix, v in enumerate(self.v_range):

            heat_strain[v] = {}

            color = self.colors_f3[ix]

            for rh in np.arange(0, 105, 1):

                for ta in np.arange(28, 66, 0.25):

                    r = use_fans_heatwaves(ta, ta, v, rh, 1.1, 0.5, wme=0)

                    # determine critical temperature at which heat strain would occur
                    if r["heat_strain"]:
                        heat_strain[v][rh] = ta
                        break

            # plot Jay's data
            if v in self.heat_strain_ollie.keys():
                ax.plot(
                    self.heat_strain_ollie[v].keys(),
                    self.heat_strain_ollie[v].values(),
                    linestyle="-.",
                    label=f"V = {v}; Jay et al. (2015)",
                    c=color,
                )

            x = list(heat_strain[v].keys())

            y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

            heat_strain[v] = {}
            for x_val, y_val in zip(x, y_smoothed):
                heat_strain[v][x_val] = y_val

            ax.plot(
                x, y_smoothed, label=f"V = {v}m/s; Gagge et al. (1986)", c=color,
            )

        np.save(os.path.join("code", "heat_strain.npy"), heat_strain)

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel="Relative humidity ($RH$) [%]",
            ylabel="Operative temperature ($t_{o}$) [°C]",
            ylim=(29, 50),
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

        fig, ax = plt.subplots(figsize=(7, 6))

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

                        r = use_fans_heatwaves(ta, ta, v, rh, met, clo, wme=0)

                        if r["heat_strain"]:
                            heat_strain[v][rh] = ta
                            break

                x = list(heat_strain[v].keys())

                y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

                ax.plot(
                    x,
                    y_smoothed,
                    label=f"V = {v}, clo = {clo}, met = {met}",
                    c=color,
                    linestyle=ls,
                )

        ax.grid(c="lightgray")

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel="Relative humidity ($RH$) [%]",
            ylabel="Operative temperature ($t_{o}$) [°C]",
            ylim=(29, 50),
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

                    r = use_fans_heatwaves(ta, ta, v, rh, 1.1, 0.5, wme=0)

                    variable_arr.append(r[variable])

            # dataframe used to plot the two contour plots
            x, y = np.meshgrid(self.rh_range, self.ta_range)

            variable_arr = [x if x > 1 else 0 for x in variable_arr]
            df = pd.DataFrame({"tmp": tmp_array, "rh": rh_array, "z": variable_arr})
            df_comparison[f"index_{ix}"] = variable_arr
            df_w = df.pivot("tmp", "rh", "z")
            cf = ax[ix].contourf(x, y, df_w.values, levels_cbar)

            ax[ix].set(
                xlabel="Relative humidity",
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
            xlabel="Relative humidity",
            ylabel="Air temperature",
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
                    use_fans_heatwaves(x, x, v, rh, 1.1, 0.5, wme=0)["temp_core"]
                    - use_fans_heatwaves(x, x, 0.2, rh, 1.1, 0.5, wme=0)["temp_core"]
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
                    label="V = 0.2 m/s",
                )
                f_02_critical = np.poly1d(
                    np.polyfit(
                        list(self.heat_strain[key].keys()),
                        list(self.heat_strain[key].values()),
                        2,
                    )
                )
            if key == 0.8:
                (ln_0,) = ax.plot(
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
                (ln_1,) = ax.plot(
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

        # (ln_0,) = ax.plot(x_new, y_new, c="k", linestyle="-.", label="V = 0.8 m/s")

        fb_0 = ax.fill_between(
            x_new,
            y_new,
            100,
            color="tab:red",
            alpha=0.2,
            zorder=100,
            label="No fan - V = 4.5 m/s",
        )

        df = pd.DataFrame(self.heat_strain)
        df_fan_above = df[df[4.5] >= df[0.8]]
        df_fan_below = df[df[4.5] <= df[0.8] + 0.073]

        x_new, y_new = interpolate(
            rh_arr, tmp_high, x_new=list(df_fan_above.index.values)
        )
        # (ln_1,) = ax.plot(x_new, y_new, c="k", label="V = 4.5 m/s", linestyle=":")

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
            label="No fan - V = 4.5 m/s",
        )

        # green part on the right
        fb_2 = ax.fill_between(
            [0, *x_new], 0, [60, *y_new], color="tab:green", alpha=0.2
        )

        # # green part below evaporative cooling
        # ax.fill_between(
        #     df_fan_below.index,
        #     0,
        #     df_fan_below[4.5].values,
        #     color="tab:green",
        #     alpha=0.2,
        # )
        #
        # # blue part below evaporative cooling
        # ax.fill_between(
        #     df_fan_below.index,
        #     df_fan_below[4.5].values,
        #     100,
        #     color="tab:blue",
        #     alpha=0.2,
        # )

        ax.set(
            ylim=(29, 50),
            xlim=(0, 100),
            xlabel=r"Relative humidity ($RH$) [%]",
            ylabel=self.label_t_op,
        )
        # ax.text(
        #     10, 37.5, "Use fans", size=12, ha="center", va="center",
        # )
        # ax.text(
        #     0.85,
        #     0.75,
        #     "Do not\nuse fans",
        #     size=12,
        #     ha="center",
        #     transform=ax.transAxes,
        # )
        # ax.text(
        #     0.33,
        #     0.8,
        #     "Move to a\ncooler place\nif possible",
        #     size=12,
        #     zorder=200,
        #     ha="center",
        #     transform=ax.transAxes,
        # )
        # ax.text(
        #     9.5,
        #     53.5,
        #     "Evaporative\ncooling",
        #     size=12,
        #     zorder=200,
        #     ha="center",
        #     va="center",
        # )
        # text_dic = [
        #     {"txt": "Thermal strain\nv =0.2m/s", "x": 11, "y": 46.5, "r": -48},
        #     # {"txt": "Thermal strain\nv=0.8m/s", "x": 8.3, "y": 52, "r": -47},
        #     {"txt": "Thermal strain\nv=4.5m/s", "x": 93, "y": 33.5, "r": -20},
        #     # {"txt": "No fans, v=4.5m/s", "x": 80, "y": 39, "r": -15},
        #     # {"txt": "No fans, v=0.8m/s", "x": 80, "y": 41.5, "r": -24},
        # ]
        #
        # for obj in text_dic:
        #     ax.text(
        #         obj["x"],
        #         obj["y"],
        #         obj["txt"],
        #         size=8,
        #         ha="center",
        #         va="center",
        #         rotation=obj["r"],
        #         zorder=200,
        #     )

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

        ax.scatter(df_queried["rh"], df_queried["db_max"], s=3, c=self.colors_f3[0])

        # horizontal line showing limit imposed by most of the standards
        ax.plot([0, 100], [35, 35], c="tab:red")

        # add legend
        plt.legend(
            [ln_2, ln_0, ln_1, fb_0, fb_1, fb_2],
            [
                "V = 0.2 m/s",
                "V = 0.8 m/s",
                "V = 4.5 m/s",
                "No fans - V = 0.8 m/s",
                "No fans - V = 4.5 m/s",
                "Use fans",
            ],
            loc="lower left",
            ncol=2,
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

    def summary_use_fans_and_population(self, save_fig):
        rh_arr = np.arange(34, 110, 2)
        tmp_low = []
        # tmp_high = []

        for rh in rh_arr:

            def function(x):
                return (
                    use_fans_heatwaves(x, x, v, rh, 1, 0.35, wme=0)["temp_core"]
                    - use_fans_heatwaves(x, x, 0.2, rh, 1, 0.35, wme=0)["temp_core"]
                )

            # v = 4.5
            # try:
            #     tmp_high.append(optimize.brentq(function, 30, 130))
            # except ValueError:
            #     tmp_high.append(np.nan)

            v = 0.8
            try:
                tmp_low.append(optimize.brentq(function, 30, 130))
            except ValueError:
                tmp_low.append(np.nan)

        fig, ax = plt.subplots()

        # # plot heat strain lines
        heat_strain = {}

        for ix, v in enumerate([0.2, 0.8]):

            heat_strain[v] = {}

            for rh in np.arange(0, 105, 1):

                for ta in np.arange(28, 66, 0.25):

                    r = use_fans_heatwaves(ta, ta, v, rh, 1.0, 0.35, wme=0)

                    # determine critical temperature at which heat strain would occur
                    if r["heat_strain"]:
                        heat_strain[v][rh] = ta
                        break

            x = list(heat_strain[v].keys())

            y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

            heat_strain[v] = {}
            for x_val, y_val in zip(x, y_smoothed):
                heat_strain[v][x_val] = y_val

            if v == 0.2:
                f_02_critical = np.poly1d(
                    np.polyfit(
                        list(heat_strain[v].keys()), list(heat_strain[v].values()), 2,
                    )
                )

                (ln_2,) = ax.plot(
                    x, y_smoothed, label=f"V = {v}m/s", c="k", linestyle="-"
                )
            if v == 0.8:
                f_08_critical = np.poly1d(
                    np.polyfit(
                        list(self.heat_strain[v].keys()),
                        list(self.heat_strain[v].values()),
                        2,
                    )
                )

                (ln_0,) = ax.plot(
                    x, y_smoothed, label=f"V = {v}m/s", c="k", linestyle="-."
                )

        x_new, y_new = interpolate(rh_arr, tmp_low)

        # (ln_0,) = ax.plot(x_new, y_new, c="k", linestyle="-.", label="V = 0.8 m/s")

        fb_0 = ax.fill_between(
            x_new,
            y_new,
            100,
            color="tab:red",
            alpha=0.2,
            zorder=100,
            label="No fan - V = 4.5 m/s",
        )
        fb_0 = ax.fill_between(
            x_new,
            y_new,
            100,
            color="tab:orange",
            alpha=0.2,
            zorder=100,
            label="No fan - V = 4.5 m/s",
        )

        df = pd.DataFrame(heat_strain)
        df_fan_above = df[df[0.8] >= df[0.2] - 0.2]
        df_fan_below = df[df[0.8] <= df[0.2]]

        x_new, y_new = interpolate(
            rh_arr, tmp_low, x_new=list(df_fan_above.index.values)
        )
        # (ln_1,) = ax.plot(x_new, y_new, c="k", label="V = 4.5 m/s")

        ax.fill_between(
            x_new,
            df_fan_above[0.8].values,
            y_new,
            facecolor="none",
            zorder=100,
            hatch="/",
            edgecolor="silver",
        )

        # fb_1 = ax.fill_between(
        #     x_new,
        #     y_new,
        #     100,
        #     color="tab:orange",
        #     alpha=0.2,
        #     label="No fan - v = 4.5 m/s",
        # )

        # green part on the right
        fb_2 = ax.fill_between(
            [0, *x_new], 0, [60, *y_new], color="tab:green", alpha=0.2
        )

        # # green part below evaporative cooling
        # ax.fill_between(
        #     df_fan_below.index,
        #     0,
        #     df_fan_below[0.8].values,
        #     color="tab:green",
        #     alpha=0.2,
        # )

        # blue part below evaporative cooling
        ax.fill_between(
            df_fan_below.index,
            df_fan_below[0.8].values,
            100,
            color="tab:red",
            alpha=0.2,
        )
        ax.fill_between(
            df_fan_below.index,
            df_fan_below[0.8].values,
            100,
            color="tab:orange",
            alpha=0.2,
        )

        ax.set(
            ylim=(29, 50),
            xlim=(0, 100),
            xlabel=r"Relative humidity ($RH$) [%]",
            ylabel=r"Operative temperature ($t_{o}$) [°C]",
        )
        # ax.text(
        #     10, 37.5, "Use fans", size=12, ha="center", va="center",
        # )
        # ax.text(
        #     0.85,
        #     0.75,
        #     "Do not\nuse fans",
        #     size=12,
        #     ha="center",
        #     transform=ax.transAxes,
        # )
        # ax.text(
        #     0.33,
        #     0.8,
        #     "Move to a\ncooler place\nif possible",
        #     size=12,
        #     zorder=200,
        #     ha="center",
        #     transform=ax.transAxes,
        # )
        # ax.text(
        #     9.5,
        #     53.5,
        #     "Evaporative\ncooling",
        #     size=12,
        #     zorder=200,
        #     ha="center",
        #     va="center",
        # )
        text_dic = [
            {"txt": "Thermal strain\nv =0.2m/s", "x": 80, "y": 31.5, "r": -21},
            {"txt": "Thermal strain\nv=0.8m/s", "x": 93, "y": 33, "r": -22},
            # {"txt": "Thermal strain\nv=4.5m/s", "x": 93, "y": 33.5, "r": -20},
            # {"txt": "No fans, v=4.5m/s", "x": 80, "y": 39, "r": -15},
            # {"txt": "No fans, v=0.8m/s", "x": 80, "y": 41.5, "r": -24},
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
            )

        # plot population
        df_queried = pd.read_csv(
            os.path.join(os.getcwd(), "code", "population_weather.csv"),
            encoding="ISO-8859-1",
        )

        df_queried = df_queried.dropna().reset_index()

        # selecting only the most 115 populous cities
        df_queried = df_queried[df_queried.index < 115]

        arr_rh = []
        df_queried.dropna(inplace=True)
        for ix, row in df_queried.iterrows():
            arr_rh.append(
                psychrolib.GetRelHumFromTWetBulb(row["db_max"], row["wb_max"], 101325)
                * 100
            )

        # calculate number of stations where db_max exceeds critical temperature
        f_08_no_fan = np.poly1d(np.polyfit(rh_arr, tmp_low, 2,))
        # f_45_no_fan = np.poly1d(np.polyfit(rh_arr, tmp_high, 2,))

        df_queried["rh"] = arr_rh
        df_queried["t_crit_02"] = [f_02_critical(x) for x in arr_rh]
        df_queried["t_crit_08"] = [f_08_critical(x) for x in arr_rh]
        # df_queried["t_crit_45"] = [f_45_critical(x) for x in arr_rh]
        df_queried["t_no_fan_08"] = [f_08_no_fan(x) for x in arr_rh]
        # df_queried["t_no_fan_45"] = [f_45_no_fan(x) for x in arr_rh]
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
        # df_queried.loc[
        #     df_queried["t_crit_45"] < df_queried["db_max"], "exc_t_crit_45"
        # ] = 1
        df_queried.loc[
            df_queried["t_no_fan_08"] < df_queried["db_max"], "exc_no_fan_08"
        ] = 1
        # df_queried.loc[
        #     df_queried["t_no_fan_45"] < df_queried["db_max"], "exc_no_fan_45"
        # ] = 1
        print(
            (
                df_queried.shape[0]
                - df_queried[
                    [
                        "exc_t_crit_02",
                        "exc_t_crit_08",
                        # "exc_t_crit_45",
                        "exc_no_fan_08",
                        # "exc_no_fan_45",
                    ]
                ].sum()
            ).round()
        )
        # population that will be fine
        print("no strain without fans")
        print(df_queried[df_queried["exc_t_crit_02"] == 0]["Value"].sum() / 10 ** 6)
        print("no strain with fans 0.8")
        print(df_queried[df_queried["exc_t_crit_08"] == 0]["Value"].sum() / 10 ** 6)
        # print("no strain with fans 4.5")
        # print(df_queried[df_queried["exc_t_crit_45"] == 0]["Value"].sum() / 10 ** 6)
        print("marginal benefit with fans 0.8")
        print(df_queried[df_queried["exc_no_fan_08"] == 0]["Value"].sum() / 10 ** 6)
        # print("marginal benefit with fans 4.5")
        # print(df_queried[df_queried["exc_no_fan_45"] == 0]["Value"].sum() / 10 ** 6)

        ax.scatter(
            df_queried["rh"],
            df_queried["db_max"],
            s=df_queried["Value"] / 10 ** 5,
            c=self.colors_f3[0],
        )

        ax.annotate(
            "Jeddah, Saudi Arabia, pop. 3.8 million",
            xy=(39, 48.5),
            xytext=(49, 48.75),
            size=10,
            arrowprops=dict(relpos=(0, 0), arrowstyle="->", connectionstyle="angle",),
        )

        # horizontal line showing limit imposed by most of the standards
        ax.plot([0, 100], [35, 35], c="tab:red")

        # add legend
        plt.legend(
            [ln_2, ln_0, fb_0, fb_2],
            ["V = 0.2 m/s", "V = 0.8 m/s", "No fans - V = 0.8 m/s", "Use fans",],
            loc="lower left",
            ncol=2,
            facecolor="w",
        )
        fig.tight_layout()
        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "use_fans_and_population.png"), dpi=300
            )
        else:
            plt.show()

    def summary_use_fans_comparison_experimental(self, save_fig):
        rh_arr = np.arange(34, 110, 2)
        tmp_low = []

        for rh in rh_arr:

            def function(x):
                return (
                    use_fans_heatwaves(x, x, v, rh, 1.1, 0.5, wme=0)["temp_core"]
                    - use_fans_heatwaves(x, x, 0.2, rh, 1.1, 0.5, wme=0)["temp_core"]
                )

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
                    label="V = 0.2 m/s",
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

        x_new, y_new = interpolate(rh_arr, tmp_low)

        # (ln_0,) = ax.plot(x_new, y_new, c="k", linestyle="-.", label="V = 0.8 m/s")

        ax.fill_between(
            x_new, y_new, 100, color="tab:red", alpha=0.2, zorder=100,
        )
        fb_0 = ax.fill_between(
            x_new, y_new, 100, color="tab:orange", alpha=0.2, zorder=100,
        )

        df = pd.DataFrame(self.heat_strain)
        df_fan_above = df[df[0.8] >= df[0.2] - 0.2]
        df_fan_below = df[df[0.8] <= df[0.2]]

        x_new, y_new = interpolate(
            rh_arr, tmp_low, x_new=list(df_fan_above.index.values)
        )

        ax.fill_between(
            x_new,
            df_fan_above[0.8].values,
            y_new,
            facecolor="none",
            zorder=100,
            hatch="/",
            edgecolor="silver",
        )

        # green part on the right
        fb_2 = ax.fill_between(
            [0, *x_new], 0, [60, *y_new], color="tab:green", alpha=0.2
        )

        # # green part below evaporative cooling
        # ax.fill_between(
        #     df_fan_below.index,
        #     0,
        #     df_fan_below[0.8].values,
        #     color="tab:green",
        #     alpha=0.2,
        # )

        # blue part below evaporative cooling
        ax.fill_between(
            df_fan_below.index,
            df_fan_below[0.8].values,
            100,
            color="tab:red",
            alpha=0.2,
        )
        ax.fill_between(
            df_fan_below.index,
            df_fan_below[0.8].values,
            100,
            color="tab:orange",
            alpha=0.2,
        )

        ax.set(
            ylim=(29, 50),
            xlim=(0, 100),
            xlabel=r"Relative humidity ($RH$) [%]",
            ylabel=r"Operative temperature ($t_{o}$) [°C]",
        )
        ax.text(
            10, 38.75, "Use fans", size=12, ha="center", va="center",
        )
        ax.text(
            0.85,
            0.75,
            "Do not\nuse fans",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )
        text_dic = [
            {"txt": "Thermal strain\nv =0.2m/s", "x": 80, "y": 31.5, "r": -21},
            {"txt": "Thermal strain\nv=0.8m/s", "x": 93, "y": 33, "r": -22},
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
            )

        plt.scatter(
            15, 47, c="tab:red", label="fan not beneficial; Morris et al. (2019)",
        )
        # ax.text(
        #     27.1, 50, "H = 101 kJ/kg", ha="center", va="center", size=8, rotation=-60
        # )

        plt.scatter(50, 40, c="tab:green", label="fan beneficial; Morris et al. (2019)")
        # ax.text(
        #     60, 31.35, "H = 73 kJ/kg", ha="center", va="center", size=8, rotation=-36
        # )

        # ravanelli's results
        plt.scatter(
            80,
            36,
            c="tab:green",
            marker="+",
            label="fan beneficial; Ravanelli et al. (2015)",
        )

        plt.scatter(
            50,
            42,
            c="tab:green",
            marker="+",
            label="fan beneficial; Ravanelli et al. (2015)",
        )

        # # add legend
        plt.legend(
            facecolor="w", loc="lower left",
        )

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
                rh_const_enthalpy, self.ta_range, c="k", linestyle=":",
            )

        fig.tight_layout()
        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join(
                    self.dir_figures, "summary_use_fans_comparison_experimental.png"
                ),
                dpi=300,
            )
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

    return {
        "e_req_w": e_req,
        "e_max_w": e_max,
        "hl_dry": c_r,
        "s_req": s_req,
        "w_req": w_req,
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


def analyse_population_data(save_fig=False):

    df = pd.read_csv(
        os.path.join(os.getcwd(), "code", "population_weather.csv"),
        encoding="ISO-8859-1",
    )

    df = df.dropna().reset_index()

    # selecting only the most 115 populous cities
    df = df[df.index < 115]

    # print number of people living most pop cities
    print(df.Value.sum() / 10 ** 6)

    # cities with max temperature higher than 35°C
    print(df[df.db_max > 35]["city"].count())
    print(df[df.db_max > 35]["Value"].sum() / 10 ** 6)

    # draw map contours
    plt.figure(figsize=(7, 3.78))
    [ax, m] = self.draw_map_contours(draw_par_mer="Yes")

    df = df[
        (df["lat"] > self.lllat)
        & (df["lat"] < self.urlat)
        & (df["long"] > self.lllon)
        & (df["lat"] < self.urlon)
        & (df["db_max"] > 19.9)
    ]

    df = df.sort_values(["db_max"])

    # transform lon / lat coordinates to map projection
    proj_lon, proj_lat = m(df.long.values, df.lat.values)

    cmap = plt.get_cmap("plasma")
    new_cmap = truncate_colormap(cmap, 0.5, 1)

    sc = plt.scatter(
        proj_lon,
        proj_lat,
        df["Value"] / 10 ** 5,
        marker="o",
        c=df["db_max"],
        cmap=new_cmap,
    )

    # produce a legend with a cross section of sizes from the scatter
    handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
    labels = [
        int(x.replace("$\\mathdefault{", "").replace("}$", "")) / 10 for x in labels
    ]
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=9,
        title="Population in millions",
        frameon=False,
    )

    bounds = np.arange(math.floor(df["db_max"].min()), math.ceil(df["db_max"].max()), 4)
    sc.cmap.set_under("dimgray")
    sc.set_clim(35, df.db_max.max())
    plt.colorbar(
        sc,
        fraction=0.1,
        pad=0.1,
        aspect=40,
        label="Extreme dry-bulb air temperature ($t_{db}$) 10 years [°C]",
        ticks=bounds,
        orientation="horizontal",
        extend="min",
    )
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(
            os.path.join(self.dir_figures, "map-population-temperature.png"), dpi=300
        )
    else:
        plt.show()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def table_list_cities():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "code", "population_weather.csv"),
        encoding="ISO-8859-1",
    )

    df = df.dropna().reset_index()

    # selecting only the most 115 populous cities
    df = df[df.index < 115]

    df["city"] = df.city.str.capitalize()
    df["country"] = df.country.str.capitalize()
    df["db_max"] = df.db_max.round(1)
    df["wb_max"] = df.wb_max.round(1)

    df = df[["country", "city", "Value", "db_max", "wb_max"]].reset_index(drop=False)

    df["index"] += 1
    df = df.astype({"Value": "int32"})

    df.columns = [["Rank", "Country", "City", "Population", r"$t_{db}$", r"$t_{wb}$"]]

    df.to_latex(
        os.path.join(os.getcwd(), "manuscript", "src", "tables", "pop_weather.tex"),
        caption="Population data and weather data for the 115 most populous cities",
        label="tab:pop_weather",
        escape=False,
        column_format="ccccccccc",
        multicolumn_format="c",
        index=False,
    )


if __name__ == "__main__":

    plt.close("all")

    # analyse_em_data()

    self = DataAnalysis()

    figures_to_plot = [
        # "heat_loss",
        # "physio_variables",
        # "weather_data",
        # "heat_strain_limits",
        # "ravanelli_comp",
        # "personal_factors",
        # "fan_usage_region_weather",
        # "summary_use_fans_comparison_experimental",
        # "fan_usage_region_cities",
        # "world_map_population_weather",
        # "table_list_cities"
        "sweat_rate"
    ]

    save_figure = True

    for figure_to_plot in figures_to_plot:
        if figure_to_plot == "heat_loss":
            self.model_comparison(save_fig=save_figure)
        if figure_to_plot == "physio_variables":
            self.figure_2(save_fig=save_figure)
        if figure_to_plot == "heat_strain_limits":
            self.comparison_air_speed(save_fig=save_figure)
        if figure_to_plot == "weather_data":
            self.plot_map_world(save_fig=save_figure)
        if figure_to_plot == "ravanelli_comp":
            self.comparison_ravanelli(save_fig=save_figure)
        if figure_to_plot == "personal_factors":
            self.met_clo(save_fig=save_figure)
        if figure_to_plot == "summary_use_fans_comparison_experimental":
            self.summary_use_fans_comparison_experimental(save_fig=save_figure)
        if figure_to_plot == "fan_usage_region_weather":
            self.summary_use_fans(save_fig=save_figure)
        if figure_to_plot == "fan_usage_region_cities":
            self.summary_use_fans_and_population(save_fig=save_figure)
        if figure_to_plot == "world_map_population_weather":
            analyse_population_data(save_fig=save_figure)
        if figure_to_plot == "table_list_cities":
            table_list_cities()
        if figure_to_plot == "sweat_rate":
            self.sweat_rate_production(save_fig=save_figure)

    # ta = 45
    # rh = 30
    # v = 4.5
    # pprint(fan_use_set(tdb=ta, tr=ta, v=v, rh=rh, met=1.1, clo=0.5, wme=0))
    #
    # for t in np.arange(33, 39, 0.1):
    #     rh = 60
    #     v = 0.1
    #     print(
    #         t,
    #         " ",
    #         fan_use_set(t, t, v, rh, 1.1, 0.5, wme=0)["energy_balance"],
    #     )

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
    #                 fan_use_set(t, t, v=v, rh=r, met=1.1, clo=0.5, wme=0)[
    #                     # "skin_wettedness"
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

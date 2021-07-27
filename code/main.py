import matplotlib as mpl

mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from pythermalcomfort.psychrometrics import p_sat
from pythermalcomfort.models import phs, set_tmp
from matplotlib.colors import DivergingNorm
import seaborn as sns
import os
from scipy import optimize
import sqlite3
import psychrolib
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.colors as colors
from sklearn.metrics import mean_absolute_error

from heat_balance_model import use_fans_heatwaves

psychrolib.SetUnitSystem(psychrolib.SI)

plt.style.use("seaborn-paper")

import matplotlib.pyplot as plt

chart_labels = {
    "sweat": r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]",
    "rh": r"Relative humidity ($RH$) [%]",
    "top": r"Operative temperature ($t_{o}$) [°C]",
}

fig_size = {"1c": 3.47, "2c": 7.22}


class DataAnalysis:
    def __init__(self):
        self.dir_figures = os.path.join(os.getcwd(), "manuscript", "src", "figures")
        self.dir_tables = os.path.join(os.getcwd(), "manuscript", "src", "tables")

        self.ta_range = np.arange(30, 60, 0.5)
        self.v_range = [0.2, 0.8, 4.5]
        self.rh_range = np.arange(0, 105, 5)
        self.defaults = {
            "clo": 0.5,
            "met": 1.1,
        }

        self.colors = ["#00B0DA", "#003262", "#FDB515", "#D9661F"]
        self.colors_f3 = ["#3B7EA1", "#ED4E33", "#C4820E"]

        self.heat_strain_ollie = {
            4.5: {
                10.234375: 48.02926829,
                19.90234375: 44.71707317,
                29.1796875: 42.39512195,
                39.3359375: 40.27804878,
                49.8828125: 38.36585366,
                60.33203125: 36.65853659,
                71.07421875: 35.29268293,
                81.328125: 34.06341463,
                89.82421875: 33.03902439,
                99.8828125: 32.08292683,
            },
            0.2: {
                9.901960784: 45.05982906,
                20: 41.70940171,
                29.50980392: 39.21367521,
                39.41176471: 37.12820513,
                53.62745098: 34.73504274,
                66.37254902: 32.85470085,
                78.62745098: 31.24786325,
                90.09803922: 30.11965812,
                100.0980392: 29.12820513,
            },
        }

        self.heat_strain_file = "heat_strain.npy"
        try:
            open(os.path.join("code", self.heat_strain_file))
        except FileNotFoundError:
            self.heat_strain_different_v(save_fig=True)

        self.heat_strain = np.load(
            os.path.join("code", self.heat_strain_file), allow_pickle="TRUE"
        ).item()

        # benefit of increasing air speed on heat strain temperature
        save_var_latex(
            "increase_t_strain_v_08_rh_60",
            round(self.heat_strain[0.8][60] - self.heat_strain[0.2][60], 1),
        )

        results = []
        for rh in self.heat_strain[0.8]:
            delta = round(self.heat_strain[0.8][rh] - self.heat_strain[0.2][rh], 1)
            if delta > 0:
                results.append(delta)

        save_var_latex(
            "avg_increase_t_strain_v_08",
            pd.Series(results).mean().round(1),
        )

        save_var_latex(
            "increase_t_strain_v_45",
            round(self.heat_strain[4.5][60] - self.heat_strain[0.8][60], 1),
        )

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
        m.drawmapboundary(fill_color="white", color="white")
        m.drawcountries(
            linewidth=0.5,
            linestyle="solid",
            color="#aaa",
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

        m.drawcoastlines(linewidth=0.5, color="gray")
        m.drawstates(
            linewidth=0.5,
            linestyle="solid",
            color="#aaa",
        )

        return ax, m

    def weather_data_world_map(self, save_fig):

        # draw map contours
        plt.figure(figsize=(fig_size["2c"], 3.78))
        [ax, m] = self.draw_map_contours(draw_par_mer="No")

        df = pd.read_sql(
            "SELECT wmo, lat, long, place, "
            '"n-year_return_period_values_of_extreme_DB_20_max" as db_max, '
            '"n-year_return_period_values_of_extreme_WB_20_max" as wb_max '
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

        cmap = plt.get_cmap("plasma")
        new_cmap = truncate_colormap(cmap, 0.5, 1)

        sc = plt.scatter(
            proj_lon, proj_lat, 10, marker="o", c=df["db_max"], cmap=new_cmap
        )
        bounds = np.arange(36, 56, 4)
        sc.cmap.set_under("dimgray")
        sc.set_clim(35, 52)
        plt.colorbar(
            sc,
            fraction=0.1,
            pad=0.02,
            aspect=40,
            label="Extreme dry-bulb air temperature ($t_{db}$) [°C]",
            ticks=bounds,
            orientation="horizontal",
            extend="min",
        )
        [
            ax.spines[x].set_color("lightgray")
            for x in ["bottom", "top", "left", "right"]
        ]
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "world-map.png"), dpi=300)
        else:
            plt.show()

    def gagge_results_physio_heat_loss(self, save_fig=True):

        fig_0, ax_0 = plt.subplots(
            4, 2, figsize=(fig_size["2c"], 8), sharex="all", sharey="row"
        )
        fig_1, ax_1 = plt.subplots(
            2, 2, figsize=(fig_size["2c"], fig_size["2c"]), sharex="all"
        )

        index_color = 0

        legend_labels = []

        results = []

        for v in [0.2, 4.5]:

            for rh in [30, 60]:

                max_skin_wettedness = use_fans_heatwaves(
                    tdb=50,
                    tr=50,
                    v=v,
                    rh=100,
                    met=self.defaults["met"],
                    clo=self.defaults["clo"],
                    wme=0,
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

                if v > 1:
                    fan_on = True
                else:
                    fan_on = False

                if rh > 50:
                    lw = 1
                    alpha = 1
                else:
                    lw = 2.5
                    alpha = 1

                for ta in self.ta_range:

                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )

                    r["tdb"] = ta
                    r["rh"] = rh
                    r["v"] = v
                    r["max_sensible"] = r["hl_evaporation_max"] * max_skin_wettedness

                    results.append(r)

                    dry_heat_loss.append(r["hl_dry"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    sweat_rate.append(r["sweating_required"])
                    max_latent_heat_loss.append(
                        r["hl_evaporation_max"] * max_skin_wettedness
                    )
                    skin_wettedness.append(r["skin_wettedness"])

                    r = ollie(fan_on, ta, rh, is_elderly=False)

                    dry_heat_loss_ollie.append(r["hl_dry"])
                    sensible_skin_heat_loss_ollie.append(r["e_req_w"])

                    sweat_rate_ollie.append(r["s_req"])
                    max_latent_heat_loss_ollie.append(r["e_max_w"])

                sweat_rate_ollie = [x if x > 0 else np.nan for x in sweat_rate_ollie]

                sensible_skin_heat_loss_ollie = [
                    x if x > 0 else np.nan for x in sensible_skin_heat_loss_ollie
                ]

                label = f"V = {v} m/s; RH = {rh} %;"

                ax_0[0][0].plot(self.ta_range, dry_heat_loss, color=color, label=label)
                ax_0[0][1].plot(
                    self.ta_range,
                    dry_heat_loss_ollie,
                    color=color,
                    label=label,
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
                    self.ta_range,
                    skin_wettedness,
                    color=color,
                    label=label,
                    linewidth=lw,
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
                    self.ta_range,
                    sweat_rate_ollie,
                    color=color,
                    linestyle="--",
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

        ax_0[2][0].set(ylim=(0, 550), ylabel=chart_labels["sweat"])
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
        ax_1[0][1].set(ylim=(-0.01, 0.7), ylabel="Skin wettedness (w)")
        ax_1[1][1].set(
            ylim=(-1, 600),
            xlabel=chart_labels["top"],
            ylabel=chart_labels["sweat"],
        )
        ax_1[1][0].set(
            ylim=(-1, 200),
            xlabel=chart_labels["top"],
            ylabel="Max latent heat loss ($E_{max,w_{max}}$) [W/m$^{2}$]",
        )

        for x in range(0, 2):
            ax_1[x][0].grid(c="lightgray")
            ax_1[x][1].grid(c="lightgray")
            ax_1[x][0].set(xlim=(30, 50.1))
            ax_1[x][1].set(xlim=(30, 50.1))
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

        # calculate results for paper
        df = pd.DataFrame(results)
        hl_dry_02 = df[(df["v"] == 0.2) & (df["tdb"] == 45) & (df["rh"] == 30)][
            "hl_dry"
        ]
        hl_dry_45 = df[(df["v"] == 4.5) & (df["tdb"] == 45) & (df["rh"] == 30)][
            "hl_dry"
        ]
        save_var_latex(
            "increase_sensible_v_02_45",
            round(abs(hl_dry_45.values - hl_dry_02.values)[0], 1),
        )
        hl_sensible_02 = df[(df["v"] == 0.2) & (df["tdb"] == 45) & (df["rh"] == 30)][
            "max_sensible"
        ]
        hl_sensible_45 = df[(df["v"] == 4.5) & (df["tdb"] == 45) & (df["rh"] == 30)][
            "max_sensible"
        ]
        save_var_latex(
            "increase_latent_v_02_45",
            round(abs(hl_sensible_45.values - hl_sensible_02.values)[0], 1),
        )

    def gagge_results_physiological(self, save_fig=True):

        fig_1, ax_1 = plt.subplots(
            2, 2, figsize=(fig_size["2c"], fig_size["2c"]), sharex="all"
        )

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

                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )

                    energy_balance.append(r["energy_balance"])
                    sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                    tmp_core.append(r["temp_core"])
                    temp_skin.append(r["temp_skin"])
                    skin_wettedness.append(r["skin_wettedness"])

                label = f"V = {v} m/s; RH = {rh} %"

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
            ylim=(34.99, 40),
            xlabel=chart_labels["top"],
            ylabel=r"Core mean temperature ($t_{cr}$) [°C]",
        )
        ax_1[1][0].set(
            ylim=(34.99, 40),
            xlabel=chart_labels["top"],
            ylabel="Skin mean temperature ($t_{sk}$) [°C]",
        )

        for x in range(0, 2):
            ax_1[x][0].grid(c="lightgray")
            ax_1[x][1].grid(c="lightgray")
            ax_1[x][0].set(xlim=(30, 50.1))
            ax_1[x][1].set(xlim=(30, 50.1))
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
            for ta in np.arange(30, 51, 1):
                for rh in np.arange(0, 101, 1):
                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )
                    r["ta"] = ta
                    r["rh"] = rh
                    r["v"] = v
                    results.append(r)

        df_sweat = pd.DataFrame(results)

        fig, axn = plt.subplots(1, 2, sharey=True, figsize=(fig_size["2c"], 4))

        for i, ax in enumerate(axn.flat):
            v = air_speeds[i]
            df = df_sweat[df_sweat["v"] == v]
            df = df.pivot("ta", "rh", "sweating_required").astype("int")
            title = r"$ m_{rsw, V = 0.2 m/s}$"
            levels = range(0, 550, 50)
            if i == 1:
                df_low = df_sweat[df_sweat["v"] == air_speeds[0]]
                df_low = df_low.pivot("ta", "rh", "sweating_required").astype("int")
                df = df - df_low
                title = (
                    r"$\Delta m_{rsw} = m_{rsw, V = 0.8 m/s} - m_{rsw, V = 0.2 m/s}$"
                )
                levels = range(-50, 40, 10)

            df = df.sort_index(ascending=False)

            x, y = np.meshgrid(df.columns, df.index)

            cs = ax.contourf(
                x,
                y,
                df.values,
                norm=DivergingNorm(0),
                levels=levels,
                cmap="RdBu_r",
            )

            fig.colorbar(
                cs,
                ax=ax,
                shrink=0.9,
            )
            ax.set(
                xlabel=chart_labels["rh"],
                xlim=(0, 100),
                ylim=(30, 50),
                ylabel=chart_labels["top"] if i == 0 else None,
                title=title,
            )
            [
                ax.spines[x].set_color("lightgray")
                for x in ["bottom", "top", "left", "right"]
            ]

        plt.suptitle(chart_labels["sweat"])

        axn[0].text(90, 48.75, "A", size=12, ha="center", va="center")
        axn[1].text(90, 48.75, "B", size=12, ha="center", va="center")

        fig.tight_layout()

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "sweat_rate.png"), dpi=300)
        else:
            plt.show()

        # calculate using PHS
        results = []
        for v in air_speeds:
            for ta in np.arange(30, 50, 2):
                for rh in np.arange(10, 100, 10):
                    r = phs(ta, ta, v, rh, 1.1 * 58.2, 0.5, posture=2, duration=480)
                    r["ta"] = ta
                    r["rh"] = rh
                    r["v"] = v
                    r["water_loss"] /= 8  # dividing it by 8 hours
                    r["water_loss"] /= 1.938  # dividing it by duBois area
                    results.append(r)

        df_sweat = pd.DataFrame(results)

        fig, axn = plt.subplots(1, 2, sharey=True)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.75])

        for i, ax in enumerate(axn.flat):
            v = air_speeds[i]
            df = df_sweat[df_sweat["v"] == v]
            sns.heatmap(
                df.pivot("ta", "rh", "water_loss").astype("int"),
                annot=True,
                cbar=i == 0,
                ax=ax,
                fmt="d",
                cbar_ax=None if i else cbar_ax,
                cbar_kws={
                    "label": r"Sweat rate ($m_{rsw}$) [mL/(hm$^2$)]",
                },
                annot_kws={"size": 8},
            )
            ax.set(
                xlabel=chart_labels["rh"],
                ylabel=chart_labels["top"] if i == 0 else None,
                title=r"$V$" + f" = {v} m/s",
            )

        # cbar_ax.collections[0].colorbar.set_label("Hello")

        fig.tight_layout(rect=[0, 0, 0.88, 1])

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "sweat_rate_phs.png"), dpi=300)
        else:
            plt.show()

    def comparison_ravanelli(self, save_fig=True):

        f, ax = plt.subplots(figsize=(fig_size["1c"], 3), sharex="all")

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

            r = use_fans_heatwaves(ta, ta, v, rh, 1, 0.36, wme=0)

            tmp_core.append(r["temp_core"] - 37.204)

        # # the above temperature which I am subtracting was calculated using the below eq
        # np.mean([x for x in tmp_core if x < 37.24])

        ax.plot(df_ravanelli[0], tmp_core, label="Gagge et al.(1986)", c="k")

        ax.scatter(
            df_ravanelli[0],
            df_ravanelli[1],
            label="Ravanelli et al. (2015)",
            c="k",
        )

        mae = round(mean_absolute_error(tmp_core, df_ravanelli[1].values), 2)
        save_var_latex(
            "mean_abs_err_ravanelli",
            mae,
        )

        ax.text(20, 0.7, f"mean absolute error = {mae} °C", fontsize=9)

        ax.set(
            ylabel=r"Change in core temperature ($t_{cr}$) [°C]",
            xlabel="Relative humidity (RH) [%]",
        )

        plt.legend(
            frameon=False,
        )

        [
            ax.spines[x].set_color("lightgray")
            for x in ["bottom", "top", "left", "right"]
        ]
        f.tight_layout()
        plt.subplots_adjust(top=0.92)
        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "comparison_ravanelli.png"), dpi=300
            )
        else:
            plt.show()

    def heat_strain_different_v(self, save_fig=False):

        fig, ax = plt.subplots(figsize=(fig_size["1c"], 4))
        ax.grid(c="lightgray")

        heat_strain = {}

        for ix, v in enumerate(self.v_range):

            heat_strain[v] = {}
            color = self.colors_f3[ix]

            for rh in np.arange(0, 105, 0.5):

                for ta in np.arange(30, 66, 0.1):

                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )

                    # determine critical temperature at which heat strain would occur
                    if (
                        r["heat_strain"]
                        # or r["temp_core"]
                        # > use_fans_heatwaves(
                        #     ta, ta, self.v_range[0], rh,
                        #                         met=self.defaults["met"],
                        #                         clo=self.defaults["clo"], wme=0
                        # )["temp_core"]
                    ):
                        heat_strain[v][rh] = ta
                        break

            # plot Jay's data
            if v in self.heat_strain_ollie.keys():
                ax.plot(
                    self.heat_strain_ollie[v].keys(),
                    self.heat_strain_ollie[v].values(),
                    linestyle="--",
                    label=f"V = {v} m/s; Jay et al. (2015)",
                    c=color,
                )

            x = list(heat_strain[v].keys())

            y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

            heat_strain[v] = {}
            for x_val, y_val in zip(x, y_smoothed):
                heat_strain[v][x_val] = y_val

            if v != 0.8:
                ax.plot(
                    x,
                    y_smoothed,
                    label=f"V = {v} m/s; Gagge et al. (1986)",
                    c=color,
                )

        print([x[1] for x in heat_strain[0.2].items()])

        np.save(os.path.join("code", self.heat_strain_file), heat_strain)

        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel=chart_labels["rh"],
            ylabel=chart_labels["top"],
            ylim=(29.9, 50),
            xlim=(5, 85.5),
            xticks=(np.arange(5, 95, 10)),
        )

        sns.despine(left=True, bottom=True, right=True)
        plt.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()
        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "comparison_air_speed.png"), dpi=300
            )
        else:
            plt.show()

    def met_clo(self, save_fig):

        fig, ax = plt.subplots(figsize=(fig_size["1c"], 4))

        heat_strain = {}

        combinations = [
            {"clo": 0.36, "met": 1, "ls": "dashed"},
            {"clo": 0.5, "met": 1, "ls": "dotted"},
            {"clo": 0.36, "met": 1.2, "ls": "solid"},
        ]

        for combination in combinations:

            clo = combination["clo"]
            met = combination["met"]
            ls = combination["ls"]

            for ix, v in enumerate([0.2, 0.8]):

                heat_strain[v] = {}

                color = self.colors_f3[ix]

                for rh in np.arange(0, 105, 0.5):

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
                    label=f"V = {v} m/s, clo = {clo} clo, met = {met} met",
                    c=color,
                    linestyle=ls,
                )

        ax.grid(c="lightgray")

        ax.xaxis.set_ticks_position("none")

        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel=chart_labels["rh"],
            ylabel=chart_labels["top"],
            ylim=(29.9, 50),
            xlim=(5, 85.5),
            xticks=(np.arange(5, 95, 10)),
        )

        sns.despine(left=True, bottom=True, right=True)

        plt.legend(
            bbox_to_anchor=(-0.15, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "met_clo.png"), dpi=300)
        else:
            plt.show()

    def met_clo_v(self, save_fig, combinations, airspeeds):

        fig, ax = plt.subplots(figsize=(fig_size["2c"], 6))

        heat_strain = {}

        for combination in combinations:

            clo = combination["clo"]
            met = combination["met"]
            ls = combination["ls"]

            for ix, v in enumerate(airspeeds):

                heat_strain[v] = {}

                color = self.colors_f3[ix]

                for rh in np.arange(0, 105, 0.5):

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
                    label=f"V = {v} m/s, clo = {clo} clo, met = {met} met",
                    c=color,
                    linestyle=ls,
                )

        ax.grid(c="lightgray")

        ax.xaxis.set_ticks_position("none")

        ax.yaxis.set_ticks_position("none")

        ax.set(
            xlabel=chart_labels["rh"],
            ylabel=chart_labels["top"],
            ylim=(29.9, 50),
            xlim=(-1, 100),
        )

        sns.despine(left=True, bottom=True, right=True)

        plt.legend(
            bbox_to_anchor=(-0.15, 1.02, 1, 0.2),
            loc="lower left",
            mode="expand",
            borderaxespad=0,
            frameon=False,
        )

        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "met_clo_v.png"), dpi=300)
        else:
            plt.show()

    def plot_other_variables(self, variable, levels_cbar):

        v_range = [0.2, 0.8]

        f, ax = plt.subplots(
            len(v_range), 1, sharex="all", sharey="all", constrained_layout=True
        )

        df_comparison = pd.DataFrame()

        for ix, v in enumerate(v_range):

            tmp_array = []
            rh_array = []
            variable_arr = []

            for rh in self.rh_range:

                for ta in self.ta_range:

                    tmp_array.append(ta)
                    rh_array.append(rh)

                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )

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
        levels = [-0.1, 0, 0.25, 0.5, 1, 2, 3]
        cf = ax.contourf(x, y, df_w.values, levels, extend="both", origin=origin)
        cf.cmap.set_under("black")
        cf.cmap.set_over("red")
        # ax.contour(x, y, df_w.values, levels, colors=("k",), origin=origin)

        ax.set(
            xlabel="Relative humidity",
            ylabel="Air temperature",
            title=f"{variable} difference at v {v_range[1]} - {v_range[0]} [m/s]",
        )

        f.colorbar(cf)
        plt.show()

    def sweat_rate_vs_body_temperature(self):

        results = []

        for v in self.v_range:
            for t in range(0, 50):

                r = use_fans_heatwaves(
                    t,
                    t,
                    v,
                    5,
                    met=self.defaults["met"],
                    clo=self.defaults["clo"],
                    wme=0,
                )

                results.append(
                    [v, t, r["temp_core"], r["sweating_required"], r["temp_skin"]]
                )

        df = pd.DataFrame(
            results, columns=["v", "tdb", "temp_core", "sweating_required", "temp_skin"]
        )

        fig, ax = plt.subplots(3, 2, sharey=True, sharex="col")
        for ix, v in enumerate(self.v_range):
            df_ = df[df.v == v]

            ax[ix][0].scatter(
                df_["temp_core"],
                df_["sweating_required"],
                c=df_["tdb"],
                label=v,
                vmax=36,
            )
            ax[ix][1].scatter(
                df_["temp_skin"],
                df_["sweating_required"],
                c=df_["tdb"],
                label=v,
                vmax=36,
            )
            ax[ix][0].set_title(f"v = {v}, core temperature")
            ax[ix][1].set_title(f"v = {v}, skin temperature")

        plt.legend()
        plt.show()

        fig, ax = plt.subplots(1, 1)
        for ix, v in enumerate([4.5, 0.8, 0.2]):
            df_ = df[df.v == v]

            ax.plot(
                df_["temp_core"],
                df_["sweating_required"],
                label=v,
            )

        plt.legend()
        plt.show()

    def summary_use_fans_two_speeds(
        self,
        air_speeds=[0.2, 0.8],
        fig=False,
        ax=False,
        plot_heat_strain_lines=False,
        legend=False,
    ):

        if not ax:
            fig, ax = plt.subplots()

        alpha = 0.45

        rh_arr = np.arange(32, 105, 1)
        tmp_low = []

        def function(x):
            return (
                use_fans_heatwaves(
                    x,
                    x,
                    air_speeds[1],
                    rh,
                    met=self.defaults["met"],
                    clo=self.defaults["clo"],
                    wme=0,
                )["temp_core"]
                - use_fans_heatwaves(
                    x,
                    x,
                    air_speeds[0],
                    rh,
                    met=self.defaults["met"],
                    clo=self.defaults["clo"],
                    wme=0,
                )["temp_core"]
            )

        for rh in rh_arr:

            t_min = 37
            if rh < 55:
                t_min = 32.5

            try:
                tmp_low.append(optimize.brentq(function, t_min, 55))
            except ValueError:
                tmp_low.append(np.nan)

        # plot heat strain lines
        heat_strain = {}

        for ix, v in enumerate(air_speeds):

            heat_strain[v] = {}

            for rh in np.arange(0, 105, 1):

                for ta in np.arange(28, 66, 0.25):

                    r = use_fans_heatwaves(
                        ta,
                        ta,
                        v,
                        rh,
                        met=self.defaults["met"],
                        clo=self.defaults["clo"],
                        wme=0,
                    )

                    # determine critical temperature at which heat strain would occur
                    if r["heat_strain"]:
                        heat_strain[v][rh] = ta
                        break

            x = list(heat_strain[v].keys())

            y_smoothed = gaussian_filter1d(list(heat_strain[v].values()), sigma=3)

            heat_strain[v] = {}
            for x_val, y_val in zip(x, y_smoothed):
                heat_strain[v][x_val] = y_val

        if plot_heat_strain_lines:
            ax.plot(
                x,
                heat_strain[air_speeds[0]].values(),
                label=f"V = {air_speeds[0]} m/s",
                c="k",
                linestyle="-",
            )

            ax.plot(
                x,
                heat_strain[air_speeds[1]].values(),
                label=f"V = {air_speeds[1]} m/s",
                c="k",
                linestyle="-.",
            )

        t_cutoff = 30
        rh_cutoff = 10
        for rh in heat_strain[air_speeds[0]].keys():
            if heat_strain[air_speeds[0]][rh] + 0.1 >= heat_strain[air_speeds[1]][rh]:
                t_cutoff = heat_strain[air_speeds[0]][rh]
                rh_cutoff = rh

        save_var_latex(
            f"t_cutoff_v_{air_speeds[1]}".replace(".", ""),
            round(t_cutoff, 1),
        )

        x_new, y_new = rh_arr, tmp_low

        y_new = [x if x < t_cutoff else t_cutoff for x in y_new]

        # plotting this twice to ensure consistency colors previous chart
        ax.fill_between(
            [0, *x_new],
            [t_cutoff, *y_new],
            50,
            color="coral",
            alpha=alpha,
            label=f"No fans - V = {air_speeds[1]} m/s",
            edgecolor=None,
        )

        upper_limit_heat_strain = [
            x if x < t_cutoff else t_cutoff for x in heat_strain[air_speeds[1]].values()
        ][-len(x_new) :]

        upper_limit_heat_strain.insert(0, t_cutoff)

        ax.fill_between(
            [rh_cutoff, *x_new],
            upper_limit_heat_strain,
            [t_cutoff, *y_new],
            color="#2dc653",
            alpha=alpha,
            label=f"Heat strain - fans beneficial",
            edgecolor=None,
        )

        ax.fill_between(
            [0, rh_cutoff, *x_new],
            25,
            [t_cutoff, *upper_limit_heat_strain],
            color="#b7efc5",
            alpha=alpha,
            label=f"No heat strain - fans beneficial",
            edgecolor=None,
        )

        ax.set(
            ylim=(29.95, 50),
            xlim=(5, 85.05),
            xticks=(np.arange(5, 95, 10)),
            xlabel=chart_labels["rh"],
            ylabel=chart_labels["top"],
        )

        text_dic = [
            {
                "txt": f"Thermal strain\nv ={air_speeds[0]} m/s",
                "x": 80,
                "y": 31.5,
                "r": -21,
            },
            {
                "txt": f"Thermal strain\nv={air_speeds[1]} m/s",
                "x": 93,
                "y": 33,
                "r": -22,
            },
        ]

        for obj in text_dic:
            if not ax:
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

        # horizontal line showing limit imposed by most of the standards
        ax.plot([0, 100], [35, 35], c="tab:red")

        # add legend
        if legend:
            plt.legend(
                facecolor="w",
                loc="lower left",
            )

        return fig, ax

    def summary_use_fans(self, save_fig):

        fig, ax = self.summary_use_fans_two_speeds(legend=True)

        # plot extreme weather events
        df_queried = pd.read_sql(
            "SELECT wmo, "
            '"n-year_return_period_values_of_extreme_DB_20_max" as db_max, '
            '"n-year_return_period_values_of_extreme_WB_20_max" as wb_max, '
            '"cooling_DB_MCWB_0.4_DB" as cool_db, '
            '"cooling_DB_MCWB_0.4_MCWB" as cool_wb '
            "FROM data",
            con=self.conn,
        )

        arr_rh = []
        df_queried[["wmo", "db_max", "wb_max", "cool_db", "cool_wb"]] = df_queried[
            ["wmo", "db_max", "wb_max", "cool_db", "cool_wb"]
        ].apply(pd.to_numeric, errors="coerce")
        df_queried.dropna(inplace=True)
        for ix, row in df_queried.iterrows():
            hr = psychrolib.GetHumRatioFromTWetBulb(
                row["cool_db"], row["cool_wb"], 101325
            )
            arr_rh.append(
                psychrolib.GetRelHumFromHumRatio(row["db_max"], hr, 101325) * 100
            )
        df_queried["rh"] = [round(x * 2) / 2 for x in arr_rh]

        t_cut_off = {}

        for ix, v in enumerate([0.2, 0.8, 4.5]):

            heat_strain_v = self.heat_strain[v].copy()

            df_queried[f"exc_t_crit_{str(v).replace('.', '')}"] = [
                1 if x[1] > heat_strain_v[x[0]] else 0
                for x in df_queried[["rh", "db_max"]].values
            ]

            if v != 0.2:
                for rh in self.heat_strain[v].keys():
                    if self.heat_strain[0.2][rh] >= self.heat_strain[v][rh]:
                        t_cut_off[v] = self.heat_strain[0.2][rh]

                df_queried.loc[
                    df_queried["db_max"] > t_cut_off[v],
                    f"exc_t_crit_{str(v).replace('.', '')}",
                ] = 1

        per_locations_fans_beneficial = (
            100
            - df_queried[
                [
                    "exc_t_crit_02",
                    "exc_t_crit_08",
                    "exc_t_crit_45",
                ]
            ].sum()
            / df_queried.shape[0]
            * 100
        ).round()

        print("percentage locations no heat stress")
        print(per_locations_fans_beneficial)
        print("location exceed heat stress")
        print(
            df_queried[
                [
                    "exc_t_crit_02",
                    "exc_t_crit_08",
                    "exc_t_crit_45",
                ]
            ].sum()
        )

        save_var_latex(
            "per_location_fans_beneficial_08",
            per_locations_fans_beneficial["exc_t_crit_08"],
        )
        save_var_latex(
            "per_location_evaporative_cooling",
            100 - per_locations_fans_beneficial["exc_t_crit_08"],
        )
        save_var_latex(
            "per_location_fans_beneficial_45",
            per_locations_fans_beneficial["exc_t_crit_45"],
        )

        ax.scatter(df_queried["rh"], df_queried["db_max"], s=3, c="tab:gray")

        ax.text(
            0.85,
            0.75,
            "Do not\nuse fans\n$V=0.8$m/s",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )

        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        ax.set(ylim=(29.95, 50))
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout()

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "use_fans.png"), dpi=300)
        else:
            plt.show()

    def summary_use_fans_and_population_tdb_max(self, save_fig):

        fig, ax = self.summary_use_fans_two_speeds(legend=True)

        # plot population
        df_queried = pd.read_csv(
            os.path.join(os.getcwd(), "code", "population_weather.csv"),
            encoding="ISO-8859-1",
        )

        df_queried = df_queried.dropna().reset_index()

        # selecting only the most 115 populous cities
        df_queried = df_queried[df_queried.index < 115]

        df_ashrae = pd.read_sql(
            "SELECT wmo, "
            '"cooling_DB_MCWB_0.4_DB" as cool_db, '
            '"cooling_DB_MCWB_0.4_MCWB" as cool_wb '
            "FROM data",
            con=self.conn,
        )

        df_ashrae[["wmo", "cool_db", "cool_wb"]] = df_ashrae[
            ["wmo", "cool_db", "cool_wb"]
        ].apply(pd.to_numeric, errors="coerce")

        df_queried = pd.merge(df_queried, df_ashrae, on="wmo", how="left")

        arr_rh = []
        df_queried.dropna(inplace=True)
        for ix, row in df_queried.iterrows():
            hr = psychrolib.GetHumRatioFromTWetBulb(
                row["cool_db"], row["cool_wb"], 101325
            )
            arr_rh.append(
                psychrolib.GetRelHumFromHumRatio(row["db_max"], hr, 101325) * 100
            )
        df_queried["rh"] = [round(x * 2) / 2 for x in arr_rh]

        t_cut_off = {}

        for ix, v in enumerate([0.2, 0.8, 4.5]):

            heat_strain_v = self.heat_strain[v]

            df_queried[f"exc_t_crit_{str(v).replace('.', '')}"] = [
                1 if x[1] > heat_strain_v[x[0]] else 0
                for x in df_queried[["rh", "db_max"]].values
            ]

            if v != 0.2:
                for rh in self.heat_strain[v].keys():
                    if self.heat_strain[0.2][rh] >= self.heat_strain[v][rh]:
                        t_cut_off[v] = self.heat_strain[0.2][rh]

                df_queried.loc[
                    df_queried["db_max"] > t_cut_off[v],
                    f"exc_t_crit_{str(v).replace('.', '')}",
                ] = 1

            df_queried[f"t_crit_{str(v).replace('.', '')}"] = [
                heat_strain_v[x] for x in df_queried["rh"]
            ]

        # df_queried[df_queried["db_max"] < 45.2].count()

        print("cities would benefit from use fans")
        cities_benefit_use_fans = (
            df_queried.shape[0]
            - df_queried[
                [
                    "exc_t_crit_02",
                    "exc_t_crit_08",
                    "exc_t_crit_45",
                ]
            ].sum()
        )
        print(cities_benefit_use_fans)

        save_var_latex(
            "cities_benefit_fan_02", cities_benefit_use_fans["exc_t_crit_02"]
        )
        save_var_latex(
            "cities_benefit_fan_08", cities_benefit_use_fans["exc_t_crit_08"]
        )
        save_var_latex(
            "cities_benefit_fan_45", cities_benefit_use_fans["exc_t_crit_45"]
        )

        # population that will be fine
        print("millions of people that would not experience heat strain")
        print("without fans")
        print(df_queried[df_queried["exc_t_crit_02"] == 0]["Value"].sum() / 10 ** 6)
        save_var_latex(
            "people_no_strain_still_air",
            round(
                df_queried[df_queried["exc_t_crit_02"] == 0]["Value"].sum() / 10 ** 6, 1
            ),
        )
        print("with fans 0.8")
        save_var_latex(
            "people_no_strain_v_08",
            round(
                df_queried[df_queried["exc_t_crit_08"] == 0]["Value"].sum() / 10 ** 6, 1
            ),
        )
        print("with fans 4.5")
        print(df_queried[df_queried["exc_t_crit_45"] == 0]["Value"].sum() / 10 ** 6)

        ax2 = ax.twinx()
        sc = ax2.scatter(
            df_queried["rh"],
            df_queried["db_max"],
            s=df_queried["Value"] / 10 ** 5,
            c="tab:gray",
            edgecolors="k",
            zorder=200,
        )

        ax2.get_yaxis().set_visible(False)
        ax2.set(
            ylim=(29.9, 50),
            xlim=(0, 100),
        )

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
        labels = [
            int(x.replace("$\\mathdefault{", "").replace("}$", "")) / 10 for x in labels
        ]
        ax2.legend(
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

        ax.annotate(
            "Jeddah, Saudi Arabia, pop. 3.8 million",
            xy=(16, 49.4),
            xytext=(49, 48.75),
            size=10,
            arrowprops=dict(
                relpos=(0, 0),
                arrowstyle="->",
                connectionstyle="angle",
            ),
        )

        ax.text(
            0.85,
            0.65,
            "Do not\nuse fans\n$V=0.8$m/s",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)
        ax.set(ylim=(29.95, 50))
        fig.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join(self.dir_figures, "use_fans_and_population.png"), dpi=300
            )
        else:
            plt.show()

    def summary_use_fans_and_population_twb_max(self, save_fig, tmp_use="cool_db"):

        fig, ax = self.summary_use_fans_two_speeds()

        # plot population
        df_queried = pd.read_csv(
            os.path.join(os.getcwd(), "code", "population_weather.csv"),
            encoding="ISO-8859-1",
        )

        df_queried = df_queried.dropna().reset_index()

        # selecting only the most 115 populous cities
        df_queried = df_queried[df_queried.index < 115]

        df_ashrae = pd.read_sql(
            "SELECT wmo, "
            '"cooling_DB_MCWB_0.4_DB" as cool_db, '
            '"cooling_DB_MCWB_0.4_MCWB" as cool_wb '
            "FROM data",
            con=self.conn,
        )

        df_ashrae[["wmo", "cool_db", "cool_wb"]] = df_ashrae[
            ["wmo", "cool_db", "cool_wb"]
        ].apply(pd.to_numeric, errors="coerce")

        df_queried = pd.merge(df_queried, df_ashrae, on="wmo", how="left")

        arr_rh = []
        df_queried.dropna(inplace=True)
        df_queried = df_queried[df_queried["cool_db"] > df_queried["wb_max"]]
        for ix, row in df_queried.iterrows():
            arr_rh.append(
                psychrolib.GetRelHumFromTWetBulb(row[tmp_use], row["wb_max"], 101325)
                * 100
            )
        df_queried["rh"] = [round(x * 2) / 2 for x in arr_rh]

        for ix, v in enumerate([0.2, 0.8, 4.5]):

            heat_strain_v = self.heat_strain[v]

            df_queried[f"exc_t_crit_{str(v).replace('.', '')}"] = [
                1 if x[1] > heat_strain_v[x[0]] else 0
                for x in df_queried[["rh", "db_max"]].values
            ]
            df_queried[f"t_crit_{str(v).replace('.', '')}"] = [
                heat_strain_v[x] for x in df_queried["rh"]
            ]

        print("cities in which people would experience heat strain")
        print(
            df_queried[
                [
                    "exc_t_crit_02",
                    "exc_t_crit_08",
                    "exc_t_crit_45",
                ]
            ].sum()
        )

        # population that will be fine
        print("millions of people that would not experience heat strain")
        print("without fans")
        print(df_queried[df_queried["exc_t_crit_02"] == 0]["Value"].sum() / 10 ** 6)
        print("with fans 0.8")
        print(df_queried[df_queried["exc_t_crit_08"] == 0]["Value"].sum() / 10 ** 6)
        print("with fans 4.5")
        print(df_queried[df_queried["exc_t_crit_45"] == 0]["Value"].sum() / 10 ** 6)

        ax2 = ax.twinx()
        sc = ax2.scatter(
            df_queried["rh"],
            df_queried[tmp_use],
            s=df_queried["Value"] / 10 ** 5,
            c=self.colors_f3[0],
            edgecolors="k",
        )

        ax2.get_yaxis().set_visible(False)
        ax2.set(
            ylim=(29.9, 50),
            xlim=(0, 100),
        )

        # produce a legend with a cross section of sizes from the scatter
        handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
        labels = [
            int(x.replace("$\\mathdefault{", "").replace("}$", "")) / 10 for x in labels
        ]
        ax2.legend(
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

        # ax.annotate(
        #     "Jeddah, Saudi Arabia, pop. 3.8 million",
        #     xy=(16, 49.4),
        #     xytext=(49, 48.75),
        #     size=10,
        #     arrowprops=dict(
        #         relpos=(0, 0),
        #         arrowstyle="->",
        #         connectionstyle="angle",
        #     ),
        # )

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.grid(c="lightgray")
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)
        fig.tight_layout()

        if save_fig:
            plt.savefig(
                os.path.join(
                    self.dir_figures, f"use_fans_and_population_twb_max_{tmp_use}.png"
                ),
                dpi=300,
            )
        else:
            plt.show()

    def summary_use_fans_comparison_experimental(self, save_fig):

        fig, ax = self.summary_use_fans_two_speeds(
            air_speeds=[0.2, 4.0], plot_heat_strain_lines=True
        )

        # ax.text(
        #     10,
        #     41.25,
        #     r"Use fans $V=4.0$m/s",
        #     size=12,
        #     ha="center",
        #     va="center",
        # )
        ax.text(
            0.85,
            0.75,
            "Do not\nuse fans\n$V=4.0$m/s",
            size=12,
            ha="center",
            transform=ax.transAxes,
        )
        text_dic = [
            {"txt": "Thermal strain\nv =0.2 m/s", "x": 80, "y": 31.5, "r": -21},
            # {"txt": "Thermal strain\nv=0.8 m/s", "x": 93, "y": 33, "r": -22},
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
            15,
            47,
            c="tab:red",
            label="fan not beneficial; Morris et al. (2019)",
            marker="s",
        )

        plt.scatter(
            50,
            40,
            c="#10451d",
            label="fan beneficial; Morris et al. (2019)",
            marker="s",
            zorder=200,
        )

        # ravanelli's results
        plt.scatter(
            80,
            36,
            c="#10451d",
            marker="+",
            label="fan beneficial; Ravanelli et al. (2015)",
            zorder=200,
        )

        plt.scatter(
            50,
            42,
            c="#10451d",
            marker="+",
            label="fan beneficial; Ravanelli et al. (2015)",
            zorder=200,
        )

        # plot enthalpy line
        reference_enthalpies = [100805.98, 73007.24]
        for ix, enthalpy in enumerate(reference_enthalpies):
            rh_const_enthalpy = []
            for tmp in self.ta_range:
                hr = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(enthalpy, tmp)
                rh_const_enthalpy.append(
                    psychrolib.GetRelHumFromHumRatio(tmp, hr, 101325) * 100
                )

            if ix == 0:
                ax.plot(
                    rh_const_enthalpy,
                    self.ta_range,
                    c="k",
                    linestyle=":",
                    label="isoenthalpic line",
                )
            else:
                ax.plot(
                    rh_const_enthalpy,
                    self.ta_range,
                    c="k",
                    linestyle=":",
                )

        # add legend
        plt.legend(
            facecolor="w",
            loc="lower left",
        )

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

    def phs_results(self, save_fig):

        rh_arr = np.arange(30, 110, 1)
        tmp = []

        for rh in rh_arr:

            def function(x):
                return phs(
                    x,
                    x,
                    0.8,
                    rh=rh,
                    met=self.defaults["met"] * 58.2,
                    clo=self.defaults["clo"],
                    posture=2,
                )["d_lim_t_re"] - (
                    phs(
                        x,
                        x,
                        0.2,
                        rh=rh,
                        met=self.defaults["met"] * 58.2,
                        clo=self.defaults["clo"],
                        posture=2,
                    )["d_lim_t_re"]
                    - 0.01
                )

            try:
                tmp.append(optimize.brentq(function, 33, 60))
            except ValueError:
                tmp.append(np.nan)

        x_new, y_new = interpolate(rh_arr, tmp)

        fig, ax = plt.subplots(
            1, 2, sharey=True, constrained_layout=True, figsize=(fig_size["2c"], 3.78)
        )

        # horizontal line showing limit imposed by most of the standards
        ax[1].plot([0, 100], [35, 35], c="tab:red")

        ax[1].fill_between(
            x_new,
            y_new,
            100,
            color="tab:red",
            alpha=0.2,
            zorder=100,
        )

        ax[1].fill_between(
            x_new,
            y_new,
            100,
            color="tab:orange",
            alpha=0.2,
            zorder=100,
        )

        ax[1].fill_between(
            [0, *x_new],
            0,
            [0, *y_new],
            color="tab:green",
            alpha=0.2,
            zorder=100,
        )

        ax[1].set(
            xlim=(5, 85.5),
            xticks=(np.arange(5, 95, 10)),
            xlabel=chart_labels["rh"],
        )

        ax[1].text(
            35,
            38.75,
            r"Use fans $V=0.8$m/s",
            size=12,
            ha="center",
            va="center",
        )
        ax[1].text(
            70,
            46,
            "Do not\nuse fans\n$V=0.8$m/s",
            size=12,
            ha="center",
        )

        ax[1].grid(c="lightgray")
        ax[1].xaxis.set_ticks_position("none")
        ax[1].yaxis.set_ticks_position("none")

        fig, ax[0] = self.summary_use_fans_two_speeds(
            fig=fig, ax=ax[0], air_speeds=[0.2, 0.8]
        )

        ax[0].text(
            35,
            38.75,
            r"Use fans $V=0.8$m/s",
            size=12,
            ha="center",
            va="center",
        )
        ax[0].text(
            70,
            46,
            "Do not\nuse fans\n$V=0.8$m/s",
            size=12,
            ha="center",
        )

        ax[0].text(80, 48.75, "A", size=12, ha="center", va="center")
        ax[1].text(80, 48.75, "B", size=12, ha="center", va="center")

        ax[0].grid(c="lightgray")
        ax[0].xaxis.set_ticks_position("none")
        ax[0].yaxis.set_ticks_position("none")
        sns.despine(left=True, bottom=True, right=True)

        if save_fig:
            plt.savefig(os.path.join(self.dir_figures, "phs_gagge.png"), dpi=300)
        else:
            plt.show()


def save_var_latex(key, value):
    import csv
    import os

    dict_var = {}

    file_path = os.path.join(os.getcwd(), "manuscript", "src", "mydata.dat")

    try:
        with open(file_path, newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                dict_var[row[0]] = row[1]
    except FileNotFoundError:
        pass

    dict_var[key] = value

    with open(file_path, "w") as f:
        for key in dict_var.keys():
            f.write(f"{key},{dict_var[key]}\n")


def ollie(is_fan_on, ta, rh, is_elderly):

    met = 65
    work = 0
    emissivity = 1
    boltzmann_const = 5.67 * 10 ** -8
    a_r_bsa = 0.7
    t_sk = 35.5
    s_w_lat = 2426

    if is_fan_on:
        r_cl_mean = (0.0497 + 0.0844) / 2

        v = 4.5

        if is_elderly:
            w = 0.5
        else:
            w = 0.65

        r_e_cl_f = 0.0112
        r_e_cl_r = 0.0161
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
    plt.figure(figsize=(fig_size["2c"], 3.78))
    [ax, m] = self.draw_map_contours(draw_par_mer="No")

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
        edgecolor="k",
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

    bounds = np.arange(36, 56, 4)
    sc.cmap.set_under("dimgray")
    sc.set_clim(35, 52)
    plt.colorbar(
        sc,
        fraction=0.1,
        pad=0.02,
        aspect=40,
        label="Extreme dry-bulb air temperature ($t_{db}$) [°C]",
        ticks=bounds,
        orientation="horizontal",
        extend="min",
    )
    [ax.spines[x].set_color("lightgray") for x in ["bottom", "top", "left", "right"]]
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


def compare_hospers_ashrae_weather(save_fig=True):
    df_hospers = pd.read_csv(
        os.path.join(os.getcwd(), "code", "extreme_weather_us.csv"),
        encoding="ISO-8859-1",
    )
    df_hospers["City"] = df_hospers["City"].str.lower()

    # get weather data
    df_population_weather = pd.read_csv(
        os.path.join(os.getcwd(), "code", "population_weather.csv"),
        encoding="ISO-8859-1",
    )

    df_population_weather = df_population_weather[
        df_population_weather["country"] == "u.s."
    ]

    df_merged = pd.merge(
        df_hospers,
        df_population_weather,
        left_on="City",
        right_on="city",
        how="left",
    )

    df_merged["state"] = df_merged["place"].str.split(", ", expand=True)[1]
    df_merged = df_merged[df_merged["state"] == df_merged["Abbreviation"]]

    df_ashrae = pd.read_sql(
        "SELECT wmo, "
        '"cooling_DB_MCWB_0.4_DB" as cool_db, '
        '"cooling_DB_MCWB_0.4_MCWB" as cool_wb, '
        '"n-year_return_period_values_of_extreme_DB_20_max" as db_max_50, '
        '"n-year_return_period_values_of_extreme_WB_20_max" as wb_max_50 '
        "FROM data",
        con=self.conn,
    )

    df_ashrae[["wmo", "cool_db", "cool_wb", "db_max_50", "wb_max_50"]] = df_ashrae[
        ["wmo", "cool_db", "cool_wb", "db_max_50", "wb_max_50"]
    ].apply(pd.to_numeric, errors="coerce")

    df_merged = pd.merge(
        df_merged,
        df_ashrae,
        on="wmo",
        how="left",
    )

    df_merged = df_merged.dropna(subset=["Value"])

    temperature_ref = "db_max_50"

    check = df_merged[["City", "Peak T", temperature_ref]]
    check["delta"] = check["Peak T"] - check[temperature_ref]

    arr_rh = []
    arr_rh_wb_extr = []
    for ix, row in df_merged.iterrows():
        hr = psychrolib.GetHumRatioFromTWetBulb(row["cool_db"], row["cool_wb"], 101325)
        arr_rh.append(
            psychrolib.GetRelHumFromHumRatio(row[temperature_ref], hr, 101325) * 100
        )
        arr_rh_wb_extr.append(
            psychrolib.GetRelHumFromTWetBulb(
                row[temperature_ref], row["wb_max"], 101325
            )
            * 100
        )

    df_merged["rh"] = arr_rh
    df_merged["rh_wb_extr"] = arr_rh_wb_extr

    df_merged["Constant HR"] = -df_merged["Peak RH"] + df_merged["rh"]
    df_merged["Concurrent extreme"] = -df_merged["Peak RH"] + df_merged["rh_wb_extr"]

    df_merged["delta max T"] = abs(df_merged["Peak T"] - df_merged["db_max_50"])
    df_merged["delta max T"].describe()
    # df_merged.loc[df_merged["delta max T"]>2, ['City', 'Abbreviation', 'Peak T', 'db_max_50', "Peak RH", "rh"]]

    df_merged[["Constant HR"]].describe().round(1)
    df_merged[["Concurrent extreme"]].describe().round(1)

    # drop data from LA and oxnard since have some issues
    df_merged = df_merged[~df_merged["City"].isin(["los angeles", "oxnard"])]

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(fig_size["1c"], 2.5))
    df_plot = df_merged[["Constant HR", "Concurrent extreme"]].unstack().reset_index()
    df_plot.columns = ["model", "constant", "Delta relative humidity (RH) [%]"]
    df_plot.constant = "1"
    sns.violinplot(
        x="constant",
        y="Delta relative humidity (RH) [%]",
        hue="model",
        split=True,
        ax=axs,
        cut=0,
        data=df_plot,
        inner="quartile",
        palette=self.colors[1:],
    )
    plt.xlabel("")
    plt.grid(c="lightgray")
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        frameon=False,
        ncol=2,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)
    axs.get_xaxis().set_ticks([])
    plt.tight_layout()

    if save_fig:
        plt.savefig(os.path.join(self.dir_figures, "delta_rh.png"), dpi=300)
    else:
        plt.show()

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(fig_size["2c"], 3.78))
    axs[0].scatter(
        df_merged["Peak RH"], df_merged["Peak T"], c="k", label="Hospers et al. (2020)"
    )
    axs[0].scatter(
        df_merged["rh"],
        df_merged[temperature_ref],
        c=self.colors[0],
        label="Constant HR",
    )
    for ix, row in df_merged.iterrows():
        axs[0].arrow(
            row["Peak RH"],
            row["Peak T"],
            row["rh"] - row["Peak RH"],
            row[temperature_ref] - row["Peak T"],
            shape="full",
            color="gray",
            length_includes_head=True,
            zorder=0,
        )
    axs[0].set_ylim(30, 50)
    axs[0].set_xlim(0, 60)
    axs[0].set_xlabel(chart_labels["rh"])
    axs[0].set_ylabel("Dry-bulb air temperature ($t_{db}$) [°C]")
    axs[0].legend(frameon=False)
    axs[0].grid(c="lightgray")

    axs[1].scatter(df_merged["Peak RH"], df_merged["Peak T"], c="k")
    axs[1].scatter(
        df_merged["rh_wb_extr"],
        df_merged[temperature_ref],
        c=self.colors[3],
        label="Concurrent extreme",
    )
    for ix, row in df_merged.iterrows():
        axs[1].arrow(
            row["Peak RH"],
            row["Peak T"],
            row["rh_wb_extr"] - row["Peak RH"],
            row[temperature_ref] - row["Peak T"],
            shape="full",
            color="gray",
            length_includes_head=True,
            zorder=0,
        )
    axs[1].set_ylim(30, 50)
    axs[1].set_xlim(0, 60)
    axs[1].set_xlabel(chart_labels["rh"])

    axs[1].legend()
    axs[1].grid(c="lightgray")
    axs[1].legend(frameon=False)
    sns.despine(left=True, bottom=True, right=True, top=True)
    plt.tight_layout()

    if save_fig:
        plt.savefig(
            os.path.join(self.dir_figures, "scatter_comparison_prediction.png"), dpi=300
        )
    else:
        plt.show()


def calculate_t_rh_combinations_enthalpy():
    # # calculations for the introduction section
    e_dry = psychrolib.GetMoistAirEnthalpy(
        47, psychrolib.GetHumRatioFromRelHum(47, 0.15, 101325)
    )

    save_var_latex("enthalpy_47_15", round(e_dry / 1000))

    e_humid = np.round(
        psychrolib.GetMoistAirEnthalpy(
            40, psychrolib.GetHumRatioFromRelHum(40, 0.51, 101325)
        )
        / 1000
    )

    save_var_latex("enthalpy_40_51", e_humid)

    rh_t_40_enthalpy_47_15 = np.round(
        psychrolib.GetRelHumFromHumRatio(
            40, psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(e_dry, 40), 101325
        )
        * 100
    )

    save_var_latex("rh_t_40_enthalpy_47_15", rh_t_40_enthalpy_47_15)

    rh_t_30_enthalpy_47_15 = np.round(
        psychrolib.GetRelHumFromHumRatio(
            30, psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(e_dry, 30), 101325
        )
        * 100
    )

    save_var_latex("rh_t_30_enthalpy_47_15", rh_t_30_enthalpy_47_15)

    hr_4_kpa = psychrolib.GetHumRatioFromVapPres(4000, 101325)

    save_var_latex("hr_4_kpa", round(hr_4_kpa, 3))


def effect_bsa_on_set():
    results = []
    for bsa in np.arange(1.7, 2.2, 0.1):
        for t in range(20, 50, 2):
            t_set = set_tmp(t, t, 0.2, 50, 1.2, 1, body_surface_area=bsa, round=True)
            results.append([bsa, t, t_set])

    df = pd.DataFrame(results, columns=["bsa", "tdb", "SET"])

    df_heat = df.set_index(["bsa", "tdb"]).unstack("tdb")

    plt.subplots(constrained_layout=True)
    sns.heatmap(df_heat, annot=True, fmt=".1f")
    plt.show()


if __name__ == "__main__":

    plt.close("all")

    # analyse_em_data()
    calculate_t_rh_combinations_enthalpy()

    self = DataAnalysis()

    figures_to_plot = [
        # "gagge_results_physio_heat_loss",
        # "gagge_results_physiological",
        # "weather_data_world_map",
        # "heat_strain_different_v",
        # "ravanelli_comp",
        # "met_clo",
        # "summary_use_fans_weather",
        "summary_use_fans_comparison_experimental",
        # "summary_use_fans_and_population_tdb_max",
        # "world_map_population_weather",
        # "met_clo_v",
        # "table_list_cities",
        # "compare_hospers_ashrae_weather",
        # "sweat_rate",
        # "phs",
        # graphical_abstract,
    ]

    save_figure = True

    for figure_to_plot in figures_to_plot:
        if figure_to_plot == "gagge_results_physio_heat_loss":
            self.gagge_results_physio_heat_loss(save_fig=save_figure)
        if figure_to_plot == "gagge_results_physiological":
            self.gagge_results_physiological(save_fig=save_figure)
        if figure_to_plot == "heat_strain_different_v":
            self.heat_strain_different_v(save_fig=save_figure)
        if figure_to_plot == "weather_data_world_map":
            self.weather_data_world_map(save_fig=save_figure)
        if figure_to_plot == "ravanelli_comp":
            self.comparison_ravanelli(save_fig=save_figure)
        if figure_to_plot == "met_clo":
            self.met_clo(save_fig=save_figure)
        if figure_to_plot == "met_clo_v":
            self.met_clo_v(
                save_fig=False,
                combinations=[
                    {"clo": 0.5, "met": 2, "ls": "dashed"},
                ],
                airspeeds=[0.2, 0.8],
            )
        if figure_to_plot == "summary_use_fans_comparison_experimental":
            self.summary_use_fans_comparison_experimental(save_fig=save_figure)
        if figure_to_plot == "summary_use_fans_weather":
            self.summary_use_fans(save_fig=save_figure)
        if figure_to_plot == "summary_use_fans_and_population_tdb_max":
            self.summary_use_fans_and_population_tdb_max(save_fig=save_figure)
            # self.summary_use_fans_and_population_twb_max(
            #     save_fig=save_figure, tmp_use="cool_db"
            # )
            # self.summary_use_fans_and_population_twb_max(
            #     save_fig=save_figure, tmp_use="db_max"
            # )
        if figure_to_plot == "world_map_population_weather":
            analyse_population_data(save_fig=save_figure)
        if figure_to_plot == "table_list_cities":
            table_list_cities()
        if figure_to_plot == "sweat_rate":
            self.sweat_rate_production(save_fig=save_figure)
        if figure_to_plot == "compare_hospers_ashrae_weather":
            compare_hospers_ashrae_weather(save_fig=save_figure)
        if figure_to_plot == "phs":
            self.phs_results(save_fig=save_figure)
        if figure_to_plot == "graphical_abstract":
            fig, ax = plt.subplots(constrained_layout=True, figsize=(2.25, 1.76))
            self.summary_use_fans_two_speeds(fig=fig, ax=ax, air_speeds=[0.2, 0.8])
            ax.text(
                0.5,
                0.8,
                "No fans $V=0.8$m/s",
                ha="center",
                transform=ax.transAxes,
            )
            ax.text(
                0.5,
                0.37,
                "Use fans\n$V=0.8$m/s",
                ha="center",
                transform=ax.transAxes,
            )
            ax.set(
                ylim=(29.95, 50),
                xlim=(5, 85.05),
                xticks=(np.arange(5, 95, 20)),
                xlabel=chart_labels["rh"],
                ylabel=r"Operative tmp ($t_{\mathrm{o}}$) [°C]",
            )
            sns.despine(left=True, bottom=True, right=True)
            plt.savefig(
                os.path.join(self.dir_figures, "graphical_abstract.png"), dpi=300
            )

    # self.summary_use_fans_two_speeds()

    #     self.defaults["met"] = 1.8
    #     self.summary_use_fans_two_speeds()
    #     plt.show()
    #
    # # benefit of increasing air speed
    # benefit = [
    #     x[0] - x[1]
    #     for x in zip(self.heat_strain[0.8].values(), self.heat_strain[0.2].values())
    # ]
    # pd.DataFrame({"benefit": benefit}).describe().round(1)

    # Figure 4
    # self.plot_other_variables(
    #     variable="energy_balance", levels_cbar=np.arange(0, 200, 10)
    # )
    # self.plot_other_variables(
    #     variable="skin_blood_flow", levels_cbar=np.arange(36, 43, 0.5)
    # )
    # self.plot_other_variables(
    #     variable="skin_blood_flow", levels_cbar=np.arange(50, 91, 1)
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

    # t = 46.87
    # use_fans_heatwaves(
    #     t,
    #     t,
    #     0.2,
    #     10,
    #     met=self.defaults["met"],
    #     clo=self.defaults["clo"],
    #     wme=0,
    # )["heat_strain"]
    #
    # self.heat_strain[0.2][20]

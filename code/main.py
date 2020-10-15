import matplotlib as mpl

mpl.use("Qt5Agg")  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pythermalcomfort.psychrometrics import p_sat, units_converter, p_sat_torr
import math


# def fan_use(tdb, tr, vr, rh, met, clo, wme=0, units="SI"):
#     """
#         Returns information on whether or not fans should be used during heat waves.
#         We use the equations developed by Ollie et al (2014) and by Hospers (2020).
#
#         Parameters
#         ----------
#         tdb : float
#             dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
#         tr : float
#             mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
#         vr : float
#             relative air velocity, default in [m/s] in [fps] if `units` = 'IP'
#
#             Note: vr is the relative air velocity caused by body movement and not the air
#             speed measured by the air velocity sensor.
#             It can be calculate using the function
#             :py:meth:`pythermalcomfort.psychrometrics.v_relative`.
#         rh : float
#             relative humidity, [%]
#         met : float
#             metabolic rate, [met]
#         clo : float
#             clothing insulation, [clo]
#
#             Note: The ASHRAE 55 Standard suggests that the dynamic clothing insulation is
#             used as input in the PMV model.
#             The dynamic clothing insulation can be calculated using the function
#             :py:meth:`pythermalcomfort.psychrometrics.clo_dynamic`.
#         w : critical skin wettedness, dimensionless
#         wme : float
#             external work, [met] default 0
#         units: str default="SI"
#             select the SI (International System of Units) or the IP (Imperial Units)
#             system.
#         """
#
#     is_elderly = True  # todo remove this variable
#
#     emissivity = 1
#     boltzmann_const = 5.67 * 10 ** -8  # Stefan-Boltzmann constant (W/m2K4)
#
#     # todo add back these two lines
#     # met = met * 58.2  # metabolic rate
#     # wme = wme * 58.2
#
#     p_atm = 101325
#     vapor_pressure = rh * p_sat_torr(tdb) / 100
#     a_r_bsa = 0.7
#     t_sk = 35.5
#     s_w_lat = 2426
#     bsa = 1.8a
#     icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
#     lr = 16  # Lewis ratio
#
#     h_r = 4 * emissivity * boltzmann_const * a_r_bsa * (273.2 + (t_sk + tr) / 2) ** 3
#
#     # from the SET equation
#     # h_cc corrected convective heat transfer coefficient
#     pressure_in_atmospheres = p_atm / 101325
#     h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
#
#     # h_fc forced convective heat transfer coefficient, W/(m2 °C)
#     h_fc = 8.600001 * pow((vr * pressure_in_atmospheres), 0.53)
#     h_cc = max(h_cc, h_fc)
#
#     h = h_r + h_cc
#     f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
#     r_a = 1.0 / (f_a_cl * h)  # resistance of air layer to dry heat
#     to = (h_r * tr + h_cc * tdb) / h  # operative temperature
#     # end from the SET equation
#
#     p_a = p_sat(tdb) / 1000 * rh / 100  # water vapor pressure in ambient air, kPa
#     p_sk_s = p_sat(t_sk) / 1000  # water vapor pressure skin, kPa
#
#     # calculation of the clothing area factor
#     if icl <= 0.078:
#         fcl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
#     else:
#         fcl = 1.05 + (0.645 * icl)
#
#     c_r = (t_sk - to) / (r_a + 1 / (fcl * h))
#
#     c_res_e_res = 0.0014 * met * (34 - tdb) + 0.0173 * met * (5.87 - p_a)
#
#     # amount of heat that needs to be loss via evaporation
#     e_req = met - wme - c_r - c_res_e_res
#
#     # todo choose r_cl based on fan on or off
#     w = 0.65
#     r_e_cl_f = 0.0112
#     r_e_cl_r = 0.0161
#     r_e_cl_mean = (r_e_cl_f + r_e_cl_r) / 2
#
#     # evaporative heat transfer coefficient
#     h_e = 16.5 * h_cc
#
#     e_max = w * (p_sk_s - p_a) / (r_e_cl_mean + 1 / (fcl * h_e))
#
#     # # from SET
#     # r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
#     # r_ecl = icl / (lr * icl)
#     # # e_max = maximum evaporative capacity
#     # e_max = (math.exp(18.6686 - 4030.183 / (t_sk + 235.0)) - vapor_pressure) / (
#     #     r_ea + r_ecl
#     # )
#     # # end from set
#
#     w_req = e_req / e_max # ratio heat loss sweating to max heat loss sweating
#
#     s_w_eff = 1 - (w_req ** 2) / 2
#
#     s_req = (e_req / s_w_eff * 3600) / s_w_lat
#
#     e_req_w = e_req * bsa
#     e_max_w = e_max * bsa
#     c_r_w = c_r * bsa
#
#     return({
#         "hl_evaporation_required": e_req_w,
#         "hl_evaporation_max": e_max_w,
#         "hl_dry": c_r_w,
#         "sweating_required": s_req,
#         "hl_sweating_ratio": w_req,
#     })


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

    e_req_w = e_req * bsa
    e_max_w = e_max * bsa
    c_r_w = c_r * bsa

    return {
        "e_req_w": e_req_w,
        "e_max_w": e_max_w,
        "hl_dry": c_r_w,
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

    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
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
        if skin_blood_flow < 0.5:
            skin_blood_flow = 0.5
        REGSW = c_sw * WARMB * math.exp(warms / 10.7)
        if REGSW > 500.0:
            REGSW = 500.0
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
        "hl_evaporation_required": e_sk * body_surface_area,
        "hl_evaporation_max": e_max * body_surface_area,
        "hl_dry": dry * body_surface_area,
        # "temp_skin": temp_skin,
        # "t_clothing": t_cl,
        # "sweating rate": REGSW,
        # "heat lost by vaporization sweat": e_rsw * body_surface_area,
        "temp_core": temp_core,
        "sweating_required": REGSW,
        # "hl_sweating_ratio": p_rsw,
        "skin_wetness": p_wet,
        "energy_storage_core": s_core,
        "energy_balance": m - hsk - q_res,
    }


def model_comparison():

    ta_range = range(26, 50)

    fig, ax = plt.subplots(4, 2, figsize=(7, 8), sharex="all", sharey="row")

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

            for ta in ta_range:

                r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                dry_heat_loss.append(r["hl_dry"])
                sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                sweat_rate.append(r["sweating_required"])
                max_latent_heat_loss.append(
                    r["hl_evaporation_max"] * max_skin_wettedness
                )

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

            ax[0][0].plot(ta_range, dry_heat_loss, label=f"{v} - {rh}")
            ax[0][1].plot(ta_range, dry_heat_loss_ollie, label=f"{v} - {rh}")
            ax[1][0].plot(ta_range, sensible_skin_heat_loss, label=f"{v} - {rh}")
            ax[1][1].plot(ta_range, sensible_skin_heat_loss_ollie, label=f"{v} - {rh}")

            ax[2][0].plot(ta_range, sweat_rate, label=f"{v} - {rh}")
            ax[2][1].plot(ta_range, sweat_rate_ollie, label=f"{v} - {rh}")
            ax[3][0].plot(ta_range, max_latent_heat_loss, label=f"{v} - {rh}")
            ax[3][1].plot(ta_range, max_latent_heat_loss_ollie, label=f"{v} - {rh}")

    ax[0][0].set(ylim=(-250, 200), title="SET", ylabel="dry heat loss (W)")
    ax[0][1].set(ylim=(-250, 200), title="Ollie")
    ax[1][0].set(ylim=(0, 300), ylabel="latent heat loss required (W)")
    ax[1][1].set(ylim=(0, 300))

    ax[2][0].set(ylim=(0, 550), ylabel="sweat rate")
    ax[2][1].set(ylim=(0, 550))
    ax[3][0].set(ylim=(0, 600), xlabel="Temperature", ylabel="max latent heat loss (W)")
    ax[3][1].set(ylim=(0, 600), xlabel="Temperature")

    for x in range(0, 4):
        ax[x][0].grid()
        ax[x][1].grid()

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"C:\\Users\\sbbfti\\Downloads\\comparison_models.png", dpi=300)


def fan_use_vs_no_use(v):

    ta_range = np.arange(30, 51, 0.5)
    rh_range = np.arange(10, 105, 5)

    fig, ax = plt.subplots(1, 2)

    tmp_array = []
    rh_array = []

    sensible_skin_heat_loss = []
    sensible_skin_heat_loss_ollie = []
    max_latent_heat_loss = []
    max_latent_heat_loss_ollie = []

    for rh in rh_range:

        max_skin_wettedness = fan_use_set(50, 50, v, 100, 1.2, 0.5, wme=0, units="SI")[
            "skin_wetness"
        ]

        for ta in ta_range:

            tmp_array.append(ta)
            rh_array.append(rh)

            r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

            sensible_skin_heat_loss.append(r["hl_evaporation_required"])
            max_latent_heat_loss.append(r["hl_evaporation_max"] * max_skin_wettedness)

            if v > 1:
                fan_on = True
            else:
                fan_on = False

            r = ollie(fan_on, ta, rh, is_elderly=False)

            sensible_skin_heat_loss_ollie.append(r["e_req_w"])
            max_latent_heat_loss_ollie.append(r["e_max_w"])

    sensible_skin_heat_loss = [x if x > 0 else 0 for x in sensible_skin_heat_loss]
    max_latent_heat_loss = [x if x > 0 else 0 for x in max_latent_heat_loss]
    max_latent_heat_loss_ollie = [x if x > 0 else 0 for x in max_latent_heat_loss_ollie]
    max_latent_heat_loss_ollie = [x if x > 0 else 0 for x in max_latent_heat_loss_ollie]

    ollie_model = [
        x[0] - x[1]
        for x in zip(max_latent_heat_loss_ollie, sensible_skin_heat_loss_ollie)
    ]
    ollie_model = [np.nan if x > 0 else 0 for x in ollie_model]
    set_model = [
        x[0] - x[1] for x in zip(max_latent_heat_loss, sensible_skin_heat_loss)
    ]
    set_model = [np.nan if x > 0 else 0 for x in set_model]

    df = pd.DataFrame(
        data={"tmp": tmp_array, "rh": rh_array, "ollie": ollie_model, "set": set_model}
    )

    df_ollie = df.pivot("tmp", "rh", "ollie").sort_index(ascending=False)
    sns.heatmap(df_ollie, ax=ax[1], cbar=False)
    ax[1].set(title="Ollie")

    df_set = df.pivot("tmp", "rh", "set").sort_index(ascending=False)

    t = []
    for col in df_set.columns:
        t.append(df_set[col].dropna().index[-1])

    ax[0].plot(rh_range, t)

    ax[0].grid()
    ax[1].grid()

    plt.suptitle(f"air speed {v}m/s")
    plt.tight_layout()
    plt.show()


def comparison_air_speed():

    ta_range = np.arange(26, 51, 0.5)
    rh_range = np.arange(10, 105, 5)
    v_range = [0.1, 0.2, 0.4, 0.8, 1.2, 2, 3, 4.5]

    fig, ax = plt.subplots()

    for v in v_range:

        tmp_array = []
        rh_array = []

        sensible_skin_heat_loss = []
        sensible_skin_heat_loss_ollie = []
        max_latent_heat_loss = []
        max_latent_heat_loss_ollie = []

        skin_wettedness = []
        core_tmp = []

        max_skin_wettedness = fan_use_set(50, 50, v, 100, 1.2, 0.5, wme=0, units="SI")[
            "skin_wetness"
        ]
        print(max_skin_wettedness)

        for rh in rh_range:

            for ta in ta_range:

                tmp_array.append(ta)
                rh_array.append(rh)

                r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                sensible_skin_heat_loss.append(r["hl_evaporation_required"])
                # todo check the following assumption since it is very important
                max_latent_heat_loss.append(
                    r["hl_evaporation_max"] * max_skin_wettedness
                )
                # max_latent_heat_loss.append(r["hl_evaporation_max"] * w)

                skin_wettedness.append(r["skin_wetness"])
                core_tmp.append(r["temp_core"])

                if v == 0.2:
                    fan_on = False

                    r = ollie(fan_on, ta, rh, is_elderly=False)

                    sensible_skin_heat_loss_ollie.append(r["e_req_w"])
                    max_latent_heat_loss_ollie.append(r["e_max_w"])

                elif v == 4.5:
                    fan_on = True

                    r = ollie(fan_on, ta, rh, is_elderly=False)

                    sensible_skin_heat_loss_ollie.append(r["e_req_w"])
                    max_latent_heat_loss_ollie.append(r["e_max_w"])

        sensible_skin_heat_loss = [x if x > 0 else 0 for x in sensible_skin_heat_loss]
        max_latent_heat_loss = [x if x > 0 else 0 for x in max_latent_heat_loss]
        max_latent_heat_loss_ollie = [
            x if x > 0 else 0 for x in max_latent_heat_loss_ollie
        ]
        max_latent_heat_loss_ollie = [
            x if x > 0 else 0 for x in max_latent_heat_loss_ollie
        ]

        ollie_model = [
            x[0] - x[1]
            for x in zip(max_latent_heat_loss_ollie, sensible_skin_heat_loss_ollie)
        ]
        ollie_model = [np.nan if x > 0 else 0 for x in ollie_model]
        set_model = [
            x[0] - x[1] for x in zip(max_latent_heat_loss, sensible_skin_heat_loss)
        ]
        set_model = [np.nan if x > 0 else 0 for x in set_model]

        df = pd.DataFrame(data={"tmp": tmp_array, "rh": rh_array, "set": set_model})

        if sensible_skin_heat_loss_ollie != []:
            df["ollie"] = ollie_model
            df_ollie = df.pivot("tmp", "rh", "ollie").sort_index(ascending=False)
            y, x = [], []
            for col in df_ollie.columns:
                try:
                    y.append(df_ollie[col].dropna().index[-1])
                    x.append(col)
                except:
                    pass

            # smooth line
            f2 = np.poly1d(np.polyfit(x, y, 2))
            xnew = np.linspace(min(x), max(x), 100)
            ax.plot(xnew, f2(xnew), linestyle=":", label=f"v = {v} - Ollie")

        df_set = df.pivot("tmp", "rh", "set").sort_index(ascending=False)

        y, x = [], []
        for col in df_set.columns:
            try:
                y.append(df_set[col].dropna().index[-1])
                x.append(col)
            except:
                pass

        # smooth line
        f2 = np.poly1d(np.polyfit(x, y, 2))

        xnew = np.linspace(min(x), max(x), 100)

        ax.plot(xnew, f2(xnew), label=f"v = {v} - SET")

    ax.grid()

    ax.set(xlabel="Relative Humidity", ylabel="Temperature")

    plt.legend()

    plt.tight_layout()
    plt.savefig(f"C:\\Users\\sbbfti\\Downloads\\comparison_air_speed.png", dpi=300)

    f, ax = plt.subplots()
    levels = np.arange(36, 43, 1)
    X, Y = np.meshgrid(rh_range, ta_range)
    df = pd.DataFrame({"tmp": tmp_array, "rh": rh_array, "z": core_tmp})
    df_w = df.pivot("tmp", "rh", "z")
    cf = ax.contourf(X, Y, df_w.values, levels)
    plt.colorbar(cf)
    ax.set(
        xlabel="Relative Humidity",
        ylabel="Temperature",
        title=f"core body tmp - air speed {v}",
    )
    plt.show()


def plot_other_variables(variable, levels_cbar):

    ta_range = np.arange(26, 51, 1)
    rh_range = np.arange(10, 105, 5)
    v_range = [0.2, 4.5]

    f, ax = plt.subplots(len(v_range), 1, sharex="all", sharey="all")

    df_comparison = pd.DataFrame()

    for ix, v in enumerate(v_range):

        tmp_array = []
        rh_array = []
        variable_arr = []

        for rh in rh_range:

            for ta in ta_range:

                tmp_array.append(ta)
                rh_array.append(rh)

                r = fan_use_set(ta, ta, v, rh, 1.2, 0.5, wme=0, units="SI")

                variable_arr.append(r[variable])

        # dataframe used to plot the two contour plots
        x, y = np.meshgrid(rh_range, ta_range)
        df = pd.DataFrame({"tmp": tmp_array, "rh": rh_array, "z": variable_arr})
        df_comparison[f"index_{ix}"] = [x if x > 0 else 0 for x in variable_arr]
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


if __name__ == "__main__":

    plt.close("all")
    # model_comparison()

    # fan_use_vs_no_use(v=0.2)
    # fan_use_vs_no_use(v=2)

    # comparison_air_speed()
    #
    plot_other_variables(
        variable="energy_storage_core", levels_cbar=np.arange(0, 150, 5)
    )
    # plot_other_variables(variable="energy_balance", levels_cbar=np.arange(-20, 160, 10))
    # plot_other_variables(variable="temp_core", levels_cbar=np.arange(36, 43, .5))

    # todo there is an issue with the SET model the latent heat loss decreases after a certain temperature

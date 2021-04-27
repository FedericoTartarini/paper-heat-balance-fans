import math
from numba import njit


def use_fans_heatwaves(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    wme=0,
    body_surface_area=1.8258,
    p_atmospheric=101325,
):
    """
    Parameters
    ----------
    tdb : float
        dry bulb air temperature, in [C]
    tr : float
        mean radiant temperature, in [C]
    v : float
        air velocity, in [m/s]
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
    p_atmospheric : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    """

    (
        e_sk,
        e_rsw,
        e_diff,
        e_max,
        w_max,
        dry,
        temp_core,
        temp_skin,
        exc_blood_flow,
        exc_p_wet,
        exc_reg_sw,
        skin_blood_flow,
        reg_sw,
        p_wet,
        m,
        hsk,
        q_res,
    ) = fans_function_optimized(
        tdb, tr, v, rh, met, clo, wme, body_surface_area, p_atmospheric
    )

    return {
        "hl_evaporation_required": e_sk,
        "e_rsw": e_rsw,
        "e_diff": e_diff,
        "hl_evaporation_max": e_max,
        "w_max": w_max,
        "hl_dry": dry,
        "temp_core": temp_core,
        "temp_skin": temp_skin,
        "heat_strain": any([exc_blood_flow, exc_p_wet, exc_reg_sw]),
        "exc_blood_flow": exc_blood_flow,
        "exc_pwet": exc_p_wet,
        "exc_reg_sw": exc_reg_sw,
        "skin_blood_flow": skin_blood_flow,
        "sweating_required": reg_sw,
        "sweating_required_ollie_equation": (3600 * e_sk / (1 - p_wet ** 2 / 2)) / 2426,
        "skin_wettedness": p_wet,
        "energy_balance": m - hsk - q_res,
    }


@njit()
def fans_function_optimized(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    wme=0,
    body_surface_area=1.8258,
    p_atmospheric=101325,
):
    vapor_pressure = rh * math.exp(18.6686 - 4030.183 / (tdb + 235.0)) / 100

    # variables to check if person is experiencing heat strain
    exc_blood_flow = False  # reached max blood flow
    exc_reg_sw = False  # reached max regulatory sweating
    exc_p_wet = False  # reached max skin wettedness

    # Initial variables as defined in the ASHRAE 55-2017
    air_velocity = max(v, 0.1)
    body_weight = 69.9  # body weight in kg
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 200  # driving coefficient for vasodilation todo ashrae says 50 see page 195
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alfa = 0.1  # fractional skin mass
    temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    temp_skin = temp_skin_neutral
    temp_core = temp_core_neutral
    skin_blood_flow = skin_blood_flow_neutral
    alfa = 0.1  # fractional skin mass

    # initial guess
    e_sk = 0.1 * met  # total evaporative heat loss, W
    dry = 0  # total sensible heat loss, W
    p_wet = 0  # skin wettedness

    pressure_in_atmospheres = p_atmospheric / 101325
    length_time_simulation = 60  # length time simulation
    i = 0  # iteration counter

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = met * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    if clo <= 0:
        w_max = 0.38 * pow(air_velocity, -0.29)  # critical skin wettedness
        i_cl = 1.0  # permeation efficiency of water vapour through the clothing layer
    else:
        w_max = 0.59 * pow(air_velocity, -0.08)  # critical skin wettedness
        i_cl = 0.45  # permeation efficiency of water vapour through the clothing layer

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 C)
    h_fc = 8.600001 * pow((air_velocity * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)

    c_hr = 4.7  # linearized radiative heat transfer coefficient
    CTC = c_hr + h_cc
    r_a = 1.0 / (f_a_cl * CTC)  # resistance of air layer to dry heat
    t_op = (c_hr * tr + h_cc * tdb) / CTC  # operative temperature

    while i < length_time_simulation:

        i += 1

        iteration_limit = 150  # for following while loop
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * temp_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:

            c_hr = 4.0 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.72
            CTC = c_hr + h_cc
            r_a = 1.0 / (f_a_cl * CTC)
            t_op = (c_hr * tr + h_cc * tdb) / CTC
            # todo check this equation
            t_cl_new = (r_a * temp_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations heat_strain")

        dry = (temp_skin - t_op) / (r_a + r_clo)
        # h_fcs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        h_fcs = (temp_core - temp_skin) * (5.28 + 1.163 * skin_blood_flow)
        q_res = 0.0023 * m * (44.0 - vapor_pressure)  # heat loss due to respiration
        c_res = 0.0014 * m * (34.0 - tdb)  # convective heat loss respiration
        s_core = m - h_fcs - q_res - c_res - wme  # rate of energy storage in the core
        s_skin = h_fcs - dry - e_sk  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (
            tc_sk * 60.0
        )  # temperature change C per minute
        d_t_cr = s_core * body_surface_area / (tc_cr * 60.0)
        temp_skin = temp_skin + d_t_sk
        temp_core = temp_core + d_t_cr
        t_body = alfa * temp_skin + (1 - alfa) * temp_core  # mean body temperature, C
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = temp_skin - temp_skin_neutral
        warms = (sk_sig > 0) * sk_sig  # vasodilation signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the skin, C
        c_reg_sig = temp_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        # bd_sig thermoregulatory control signal from the body
        bd_sig = t_body - temp_body_neutral
        WARMB = (bd_sig > 0) * bd_sig
        skin_blood_flow = (skin_blood_flow_neutral + c_dil * c_warm) / (
            1 + c_str * colds
        )
        if skin_blood_flow > 90.0:
            skin_blood_flow = 90.0
            exc_blood_flow = True
        if skin_blood_flow < 0.5:
            skin_blood_flow = 0.5
        reg_sw = c_sw * WARMB * math.exp(warms / 10.7)  # regulatory sweating
        if reg_sw > 500.0:
            reg_sw = 500.0
            exc_reg_sw = True
        e_rsw = 0.68 * reg_sw  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        # e_max = maximum evaporative capacity
        e_max = (
            math.exp(18.6686 - 4030.183 / (temp_skin + 235.0)) - vapor_pressure
        ) / (r_ea + r_ecl)
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        p_wet = 0.06 + 0.94 * p_rsw
        e_diff = p_wet * e_max - e_rsw  # vapor diffusion through skin
        if p_wet > w_max:
            p_wet = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
            exc_p_wet = True
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            p_wet = w_max
        e_sk = (
            e_rsw + e_diff
        )  # total evaporative heat loss sweating and vapor diffusion
        met_shivering = 19.4 * colds * c_cold
        m = rm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417)

    hsk = dry + e_sk  # total heat loss from skin, W

    return [
        e_sk,
        e_rsw,
        e_diff,
        e_max,
        w_max,
        dry,
        temp_core,
        temp_skin,
        exc_blood_flow,
        exc_p_wet,
        exc_reg_sw,
        skin_blood_flow,
        reg_sw,
        p_wet,
        m,
        hsk,
        q_res,
    ]


if __name__ == "__main__":
    print(
        use_fans_heatwaves(tdb=47, tr=47, v=4.5, rh=10, met=1.1, clo=0.5, wme=0)[
            "sweating_required"
        ]
    )

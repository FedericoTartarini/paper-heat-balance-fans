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
        body surface area, default value 1.8258 [m2]
    p_atmospheric : float
        atmospheric pressure, default value 101325 [Pa] in [atm]

    Returns
    -------
    e_skin : float
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff
    e_rsw : float
        Rate of evaporative heat loss from sweat evaporation, [W/m2]
    e_diff : float
        Rate of evaporative heat loss from moisture diffused through the skin, [W/m2]
    e_max : float
        Maximum rate of evaporative heat loss from skin, [W/m2]
    q_sensible : float
        Sensible heat loss from skin, [W/m2]
    q_skin : float
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin
    q_res : float
        Total rate of heat loss through respiration, [W/m2]
    t_core : float
        Core temperature, [C]
    t_skin : float
        Skin temperature, [C]
    m_bl : float
        Skin blood flow, [L/(hm2)]
    m_rsw : float
        Rate at which regulatory sweat is generated, [mL/h2]
    w : float
        Skin wettedness, adimensional. Ranges from 0 and 1.
    w_max : float
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 and 1.
    heat_strain : bool
        True if the model predict that the person may be experiencing heat strain
    heat_strain_blood_flow : bool
        True if heat strain is caused by skin blood flow (m_bl) reaching its maximum value
    heat_strain_w : bool
        True if heat strain is caused by skin wettedness (w) reaching its maximum value
    heat_strain_sweating : bool
        True if heat strain is caused by regulatory sweating (m_rsw) reaching its maximum value

    """

    (
        e_skin,
        e_rsw,
        e_diff,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        heat_strain_blood_flow,
        heat_strain_w,
        heat_strain_sweating,
    ) = fans_function_optimized(
        tdb, tr, v, rh, met, clo, wme, body_surface_area, p_atmospheric
    )

    return {
        "e_skin": e_skin,
        "e_rsw": e_rsw,
        "e_diff": e_diff,
        "e_max": e_max,
        "q_sensible": q_sensible,
        "q_skin": q_skin,
        "q_res": q_res,
        "t_core": t_core,
        "t_skin": t_skin,
        "m_bl": m_bl,
        "m_rsw": m_rsw,
        "w": w,
        "w_max": w_max,
        "heat_strain_blood_flow": heat_strain_blood_flow,
        "heat_strain_w": heat_strain_w,
        "heat_strain_sweating": heat_strain_sweating,
        "heat_strain": any(
            [heat_strain_blood_flow, heat_strain_w, heat_strain_sweating]
        ),
        "sweating_required_ollie_equation": (3600 * e_skin / (1 - w ** 2 / 2)) / 2426,
        "energy_balance": met * 58.2 - q_skin - q_res,
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
    max_skin_blood_flow=80,
):
    vapor_pressure = rh * math.exp(18.6686 - 4030.183 / (tdb + 235.0)) / 100

    # variables to check if person is experiencing heat strain
    heat_strain_blood_flow = False  # reached max blood flow
    heat_strain_sweating = False  # reached max regulatory sweating
    heat_strain_w = False  # reached max skin wettedness

    # Initial variables as defined in the ASHRAE 55-2017
    air_speed = max(v, 0.1)
    body_weight = 69.9  # body weight in kg
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 200  # driving coefficient for vasodilation ashrae says 50 see page 195
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alfa = 0.1
    temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    m_bl = skin_blood_flow_neutral

    # initialize some variables
    e_skin = 0.1 * met  # total evaporative heat loss, W
    q_sensible = 0  # total sensible heat loss, W
    w = 0  # skin wettedness
    e_rsw = 0  # heat lost by vaporization sweat
    e_diff = 0  # vapor diffusion through skin
    e_max = 0  # maximum evaporative capacity
    m_rsw = 0  # regulatory sweating
    q_res = 0  # heat loss due to respiration

    pressure_in_atmospheres = p_atmospheric / 101325
    length_time_simulation = 60  # length time simulation
    n_simulation = 0

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = met * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    if clo <= 0:
        w_max = 0.38 * pow(air_speed, -0.29)  # critical skin wettedness
        i_cl = 1.0  # permeation efficiency of water vapour through the clothing layer
    else:
        w_max = 0.59 * pow(air_speed, -0.08)  # critical skin wettedness
        i_cl = 0.45  # permeation efficiency of water vapour through the clothing layer

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 C)
    h_fc = 8.600001 * pow((air_speed * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)
    h_c_met = 5.66 * (met - 0.85) ** 0.39
    h_cc = max(h_cc, h_c_met)

    h_r = 4.7  # linearized radiative heat transfer coefficient
    h_t = h_r + h_cc  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_a = 1.0 / (f_a_cl * h_t)  # resistance of air layer to dry heat
    t_op = (h_r * tr + h_cc * tdb) / h_t  # operative temperature

    while n_simulation < length_time_simulation:

        n_simulation += 1

        iteration_limit = 150  # for following while loop
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:

            # 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
            # 0.7 ratio between radiation area of the body and the body area
            h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.7
            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr + h_cc * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations exceeded")

        q_sensible = (t_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W
        # hf_cs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        q_res = 0.0023 * m * (44.0 - vapor_pressure)  # heat loss due to respiration
        c_res = 0.0014 * m * (34.0 - tdb)  # convective heat loss respiration
        s_core = m - hf_cs - q_res - c_res - wme  # rate of energy storage in the core
        s_skin = hf_cs - q_sensible - e_skin  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (
            tc_sk * 60.0
        )  # rate of change skin temperature C per minute
        d_t_cr = (
            s_core * body_surface_area / (tc_cr * 60.0)
        )  # rate of change core temperature C per minute
        t_skin = t_skin + d_t_sk
        t_core = t_core + d_t_cr
        t_body = alfa * t_skin + (1 - alfa) * t_core  # mean body temperature, C
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = t_skin - temp_skin_neutral
        warm_sk = (sk_sig > 0) * sk_sig  # vasodilation signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the skin, C
        c_reg_sig = t_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        # bd_sig thermoregulatory control signal from the body
        bd_sig = t_body - temp_body_neutral
        warm_b = (bd_sig > 0) * bd_sig
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1 + c_str * colds)
        if m_bl > max_skin_blood_flow:
            m_bl = max_skin_blood_flow
            heat_strain_blood_flow = True
        if m_bl < 0.5:
            m_bl = 0.5
        m_rsw = c_sw * warm_b * math.exp(warm_sk / 10.7)  # regulatory sweating
        if m_rsw > 500.0:
            m_rsw = 500.0
            heat_strain_sweating = True
        e_rsw = 0.68 * m_rsw  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        e_max = (math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (
            r_ea + r_ecl
        )
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        w = 0.06 + 0.94 * p_rsw  # skin wetness
        e_diff = w * e_max - e_rsw  # vapor diffusion through skin
        if w > w_max:
            w = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
            heat_strain_w = True
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            w = w_max
        e_skin = (
            e_rsw + e_diff
        )  # total evaporative heat loss sweating and vapor diffusion
        met_shivering = 19.4 * colds * c_cold  # met shivering W/m2
        m = rm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417)

    q_skin = q_sensible + e_skin  # total heat loss from skin, W

    return [
        e_skin,
        e_rsw,
        e_diff,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        heat_strain_blood_flow,
        heat_strain_w,
        heat_strain_sweating,
    ]

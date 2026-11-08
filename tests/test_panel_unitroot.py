"""
Panel unit-root tests (Stata xtunitroot) and panel cointegration tests (Stata xtcointtest) — tests/test_panel_unitroot.py

Reference: Stata 19.5 (StataNow MP). Data: sample_data/Panel_UnitRoot_Regions.csv (25 regions × 38 years, written by
sample_data/make_panel_unitroot_sample_data.py); its unbalanced copy (`unbalanced` in that generator: ids divisible by 3
lose their first (id % 4)*2 years); the unbalanced copy with one interior gap (id 5, year 2000 dropped).
Commands: xtunitroot llc / ht / breitung / ips / fisher / hadri and xtcointtest kao / pedroni / westerlund with the options
listed in UR_REF and CO_REF (do-files kept outside the repository: ur_balanced.do,
ur_unbalanced.do, ur_failed_panel.do, coint_balanced.do, coint_unbalanced.do; moments/moments.do recovers the Westerlund adjustment terms and
checks the Pedroni table 2 terms on independent random walks with 1-7 regressors).
Values: `import delimited ..., asdouble`, then `return list` and `matrix list r(stats)` / `r(p)` at full precision,
hard-coded below (Stata is not needed to run the tests).

Tolerances
- xtunitroot: relative 1e-8 (closed-form statistics; observed agreement about 1e-12).
- xtcointtest kao / pedroni: relative 2e-6 plus absolute 2e-6. Stata keeps the AR(1), fixed-effects and per-panel
  cointegrating residuals in float (single-precision) variables; rounding them to float32 reproduces Stata's long-run
  variance lags to all digits, so the remaining difference is Stata's storage precision.
- xtcointtest westerlund: absolute 1e-6 (the adjustment terms are carried to 8 decimals).
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "reconstructed"))
sys.path.insert(0, os.path.join(ROOT_DIR, "sample_data"))

import pyjamovi as ost  # noqa: E402
from make_panel_unitroot_sample_data import unbalanced  # noqa: E402
from pyjamovi.analyses import extensions  # noqa: E402
from pyjamovi.analyses import panel_unitroot as U  # noqa: E402
from pyjamovi.stata_transpiler import transpile_stata_line  # noqa: E402

DATA = os.path.join(ROOT_DIR, "sample_data", "Panel_UnitRoot_Regions.csv")

# ----------------------------------------------------------------------------- Stata 19.5 reference values
UR_REF = {
    'balanced': [
        ('xtunitroot llc y_stat', {'p_tds': 4.91766753636e-09, 'sig_adj': 0.8636, 'mu_adj': -0.5402000000000001,
            'delta': -0.3758377985510333, 'se_delta': 0.0284288659196405, 'Var_ep': 1.087456577473557, 'sbar':
            0.6505755660357923, 'ttilde': 36.0, 'td': -13.22028812592838, 'tds': -5.733544319696003, 'N_g': 25.0,
            'N_t': 38.0, 'N': 900.0, 'hac_lagm': 10.0, 'adf_lags': 1.0}),
        ('xtunitroot llc y_stat, lags(aic 4)', {'p_tds': 1.748574516e-18, 'sig_adj': 0.8626480000000001, 'mu_adj':
            -0.539976, 'delta': -0.4341380622233494, 'se_delta': 0.0275461565727257, 'Var_ep': 1.090121393964548,
            'sbar': 0.6674781161624332, 'ttilde': 36.28, 'td': -15.76038606609835, 'tds': -8.694059336228905,
            'N_g': 25.0, 'N_t': 38.0, 'N': 907.0, 'hac_lagm': 10.0, 'adf_lagm': 0.72}),
        ('xtunitroot llc y_stat, lags(bic 4) demean', {'p_tds': 1.44403267713e-18, 'sig_adj': 0.8606079999999999,
            'mu_adj': -0.539496, 'delta': -0.390332999633721, 'se_delta': 0.0258859728702226, 'Var_ep':
            1.089434026751622, 'sbar': 0.6411751273562369, 'ttilde': 36.88, 'td': -15.07893875925104, 'tds':
            -8.715763151020896, 'N_g': 25.0, 'N_t': 38.0, 'N': 922.0, 'hac_lagm': 10.0, 'adf_lagm': 0.12}),
        ('xtunitroot llc y_trend, trend', {'p_tds': 2.5804823637e-16, 'sig_adj': 0.9190000000000002, 'mu_adj':
            -0.6498000000000002, 'delta': -0.686680740421692, 'se_delta': 0.0373295068008503, 'Var_ep':
            1.031061324939475, 'sbar': 0.5168847872257347, 'ttilde': 36.0, 'td': -18.39511955207643, 'tds':
            -8.107646447171781, 'N_g': 25.0, 'N_t': 38.0, 'N': 900.0, 'hac_lagm': 10.0, 'adf_lags': 1.0}),
        ('xtunitroot llc y_trend, trend lags(0)', {'p_tds': 1.00441185418e-37, 'sig_adj': 0.9319999999999999, 'mu_adj':
            -0.6466000000000001, 'delta': -0.6445657946068524, 'se_delta': 0.0308177291681342, 'Var_ep':
            1.040449703399706, 'sbar': 0.5080556483129886, 'ttilde': 37.0, 'td': -20.91542147996219, 'tds':
            -12.78421439781797, 'N_g': 25.0, 'N_t': 38.0, 'N': 925.0, 'hac_lagm': 10.0, 'adf_lags': 0.0}),
        ('xtunitroot llc y_ur, noconstant lags(2)', {'p_td': 0.9962503535859883, 'p_tds': 0.9953690242703902, 'sig_adj':
            1.027, 'mu_adj': 0.002, 'delta': 0.0013734335965051, 'se_delta': 0.0005136599102298, 'Var_ep':
            1.042511485065446, 'sbar': 1.540127334481449, 'ttilde': 35.0, 'td': 2.673818939637824, 'tds':
            2.602230735984203, 'N_g': 25.0, 'N_t': 38.0, 'N': 875.0, 'hac_lagm': 10.0, 'adf_lags': 2.0}),
        ('xtunitroot llc y_ur, noconstant lags(aic 3)', {'p_td': 0.9970785914562844, 'p_tds': 0.996385858637449,
            'sig_adj': 1.025704, 'mu_adj': 0.002, 'delta': 0.0013546696734456, 'se_delta': 0.0004914501042832,
            'Var_ep': 1.052367755484913, 'sbar': 1.531348109608356, 'ttilde': 36.08, 'td': 2.756474485688719,
            'tds': 2.686139850003987, 'N_g': 25.0, 'N_t': 38.0, 'N': 902.0, 'hac_lagm': 10.0, 'adf_lagm': 0.92}),
        ('xtunitroot llc y_stat, kernel(parzen nwest)', {'p_tds': 5.8919382227e-11, 'sig_adj': 0.8636, 'mu_adj':
            -0.5402000000000001, 'delta': -0.3758377985510333, 'se_delta': 0.0284288659196405, 'Var_ep':
            1.087456577473557, 'sbar': 0.6024322210831583, 'ttilde': 36.0, 'td': -13.22028812592838, 'tds':
            -6.44209077182246, 'N_g': 25.0, 'N_t': 38.0, 'N': 900.0, 'hac_lagm': 14.44, 'adf_lags': 1.0}),
        ('xtunitroot llc y_stat, kernel(quadraticspectral nwest)', {'p_tds': 2.45189730756e-11, 'sig_adj': 0.8636,
            'mu_adj': -0.5402000000000001, 'delta': -0.3758377985510333, 'se_delta': 0.0284288659196405, 'Var_ep':
            1.087456577473557, 'sbar': 0.5934811346323132, 'ttilde': 36.0, 'td': -13.22028812592838, 'tds':
            -6.573827786640674, 'N_g': 25.0, 'N_t': 38.0, 'N': 900.0, 'hac_lagm': 7.762402683537214, 'adf_lags':
            1.0}),
        ('xtunitroot llc y_stat, kernel(quadraticspectral 4.5)', {'p_tds': 2.05182437817e-08, 'sig_adj': 0.8636,
            'mu_adj': -0.5402000000000001, 'delta': -0.3758377985510333, 'se_delta': 0.0284288659196405, 'Var_ep':
            1.087456577473557, 'sbar': 0.6673727974458938, 'ttilde': 36.0, 'td': -13.22028812592838, 'tds':
            -5.486332192394005, 'N_g': 25.0, 'N_t': 38.0, 'N': 900.0, 'hac_lags': 4.5, 'adf_lags': 1.0}),
        ('xtunitroot llc y_stat, kernel(bartlett nwest) lags(hqic 3)', {'p_tds': 1.06858664373e-17, 'sig_adj':
            0.8611519999999999, 'mu_adj': -0.539624, 'delta': -0.4086559119368833, 'se_delta': 0.026197826931612,
            'Var_ep': 1.091113000083735, 'sbar': 0.6970750116581123, 'ttilde': 36.72, 'td': -15.5988476831936,
            'tds': -8.486083813205168, 'N_g': 25.0, 'N_t': 38.0, 'N': 918.0, 'hac_lagm': 12.56, 'adf_lagm':
            0.28}),
        ('xtunitroot llc y_trend, trend lags(aic 4) kernel(bartlett 5)', {'p_tds': 7.28794110719e-15, 'sig_adj':
            0.92316, 'mu_adj': -0.648776, 'delta': -0.6937599762998773, 'se_delta': 0.0334962526728747, 'Var_ep':
            1.053876303455324, 'sbar': 0.7269663326440947, 'ttilde': 36.32, 'td': -20.71156983066598, 'tds':
            -7.691202237007342, 'N_g': 25.0, 'N_t': 38.0, 'N': 908.0, 'hac_lags': 5.0, 'adf_lagm': 0.68}),
        ('xtunitroot llc y_ur, lags(1) kernel(parzen 3)', {'p_tds': 0.598662376234137, 'sig_adj': 0.8636, 'mu_adj':
            -0.5402000000000001, 'delta': -0.0426336746027047, 'se_delta': 0.0085141359036619, 'Var_ep':
            1.038298036496669, 'sbar': 1.310150180840387, 'ttilde': 36.0, 'td': -5.007398881707765, 'tds':
            0.2498863396733248, 'N_g': 25.0, 'N_t': 38.0, 'N': 900.0, 'hac_lags': 3.0, 'adf_lags': 1.0}),
        ('xtunitroot ht y_stat', {'p': 8.39003628862e-91, 'z': -20.1733712751796, 'Var_rho': 0.000260305822436,
            'mean_rho': 0.9230769230769231, 'rho': 0.5975998344249887, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot ht y_stat, demean', {'p': 4.04379535667e-91, 'z': -20.20943038500926, 'Var_rho': 0.000260305822436,
            'mean_rho': 0.9230769230769231, 'rho': 0.5970180568962532, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot ht y_trend, trend', {'p': 2.81075814315e-74, 'z': -18.19531488732525, 'Var_rho':
            0.0005863444010417, 'mean_rho': 0.8125, 'rho': 0.3719086494147113, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot ht y_ur, noconstant', {'p': 0.6150071086559162, 'z': 0.2923934931468206, 'Var_rho':
            5.68990042674e-05, 'mean_rho': 1.0, 'rho': 1.002205565891324, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot ht y_stat, altt', {'p': 2.43480380991e-85, 'z': -19.54153444649146, 'Var_rho': 0.0002739709384264,
            'mean_rho': 0.9210526315789473, 'rho': 0.5975998344249887, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot ht y_trend, trend altt', {'p': 2.11354271295e-69, 'z': -17.56938789663216, 'Var_rho':
            0.0006152174114981, 'mean_rho': 0.8076923076923077, 'rho': 0.3719086494147113, 'N_t': 38.0, 'N_g':
            25.0, 'N': 950.0}),
        ('xtunitroot breitung y_stat', {'lags': 0.0, 'p': 9.77221905535e-12, 'lambda': -6.709386490613663, 'N_t': 38.0,
            'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_stat, lags(2)', {'lags': 2.0, 'p': 4.04896998156e-07, 'lambda': -4.932992177162756,
            'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_trend, trend', {'lags': 0.0, 'p': 3.46673495395e-19, 'lambda': -8.875956887767076,
            'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_trend, trend lags(1)', {'lags': 1.0, 'p': 1.48932763731e-11, 'lambda':
            -6.64762226838781, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_ur, noconstant', {'lags': 0.0, 'p': 0.9995875660812201, 'lambda': 3.344312440110637,
            'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_ur, noconstant lags(2)', {'lags': 2.0, 'p': 0.9960724334382817, 'lambda':
            2.658234002850306, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_stat, robust', {'lags': 0.0, 'p_lrobust': 1.69577420972e-11, 'lrobust':
            -6.628482708758783, 'p': 9.77221905535e-12, 'lambda': -6.709386490613663, 'N_t': 38.0, 'N_g': 25.0,
            'N': 950.0}),
        ('xtunitroot breitung y_stat, robust lags(1)', {'lags': 1.0, 'p_lrobust': 2.63771171612e-08, 'lrobust':
            -5.441768397408893, 'p': 9.64492061209e-10, 'lambda': -6.00367687439853, 'N_t': 38.0, 'N_g': 25.0,
            'N': 950.0}),
        ('xtunitroot breitung y_trend, trend robust', {'lags': 0.0, 'p_lrobust': 4.10230850182e-15, 'lrobust':
            -7.764379073462, 'p': 3.46673495395e-19, 'lambda': -8.875956887767076, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot breitung y_trend, trend robust lags(2)', {'lags': 2.0, 'p_lrobust': 0.0002867477049071, 'lrobust':
            -3.443849230631268, 'p': 7.92694359122e-06, 'lambda': -4.316477445684666, 'N_t': 38.0, 'N_g': 25.0,
            'N': 950.0}),
        ('xtunitroot breitung y_ur, noconstant robust lags(2)', {'lags': 2.0, 'p_lrobust': 0.9984394788164409,
            'lrobust': 2.955557774398351, 'p': 0.9960724334382817, 'lambda': 2.658234002850306, 'N_t': 38.0,
            'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot breitung y_stat, demean', {'lags': 0.0, 'p': 5.18816116084e-12, 'lambda': -6.801183400543758,
            'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot ips y_stat', {'N': 925.0, 'N_t': 38.0, 'N_g': 25.0, 'p_zttildebar': 9.30426018417e-20, 'cv_10':
            -1.75, 'cv_5': -1.82, 'cv_1': -1.93, 'zttildebar': -9.021173442144066, 'zt': -10.26277623036237,
            'ttildebar': -2.896843838894525, 'tbar': -3.329148746210804}),
        ('xtunitroot ips y_trend, trend', {'N': 925.0, 'N_t': 38.0, 'N_g': 25.0, 'p_zttildebar': 4.49295374963e-35,
            'cv_10': -2.38, 'cv_5': -2.44, 'cv_1': -2.55, 'zttildebar': -12.30064020617188, 'zt':
            -14.94816114714223, 'ttildebar': -3.419833967084175, 'tbar': -4.15345697159907}),
        ('xtunitroot ips y_stat, demean', {'N': 925.0, 'N_t': 38.0, 'N_g': 25.0, 'p_zttildebar': 1.68020639702e-19,
            'cv_10': -1.75, 'cv_5': -1.82, 'cv_1': -1.93, 'zttildebar': -8.956206197648429, 'zt':
            -10.13668791478647, 'ttildebar': -2.886483243721068, 'tbar': -3.306965801265909}),
        ('xtunitroot ips y_stat, lags(2)', {'lags': 2.0, 'N': 875.0, 'N_t': 38.0, 'N_g': 25.0, 'p_wtbar':
            9.16662085654e-14, 'wtbar': -7.360420045743222}),
        ('xtunitroot ips y_stat, lags(aic 4)', {'lagm': 0.72, 'N': 907.0, 'N_t': 38.0, 'N_g': 25.0, 'p_wtbar':
            3.25690742627e-29, 'wtbar': -11.15842106650534}),
        ('xtunitroot ips y_trend, trend lags(bic 3)', {'lagm': 0.04, 'N': 924.0, 'N_t': 38.0, 'N_g': 25.0, 'p_wtbar':
            7.49357001868e-36, 'wtbar': -12.4444724094056}),
        ('xtunitroot ips y_ur, lags(0)', {'lags': 0.0, 'N': 925.0, 'N_t': 38.0, 'N_g': 25.0, 'p_wtbar':
            0.9999990777794666, 'wtbar': 4.769761563825596}),
        ('xtunitroot ips y_ur, trend lags(hqic 2) demean', {'lagm': 0.7600000000000001, 'N': 906.0, 'N_t': 38.0, 'N_g':
            25.0, 'p_wtbar': 0.8273718368498181, 'wtbar': 0.9438303937311676}),
        ('xtunitroot fisher y_stat, dfuller lags(1)', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            14.10940040342939, 'df_L': 129.0, 'p_L': 0.0, 'L': -10.12942603665943, 'p_Z': 2.20595182092e-19, 'Z':
            -8.926125392029137, 'df_P': 50.0, 'p_P': 2.29421171662e-18, 'P': 191.0940040342939}),
        ('xtunitroot fisher y_stat, dfuller drift lags(2) demean', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.0,
            'Pm': 22.4688950680386, 'df_L': 129.0, 'p_L': 0.0, 'L': -15.1561973293755, 'p_Z': 6.44613912144e-38,
            'Z': -12.81865176334641, 'df_P': 50.0, 'p_P': 8.90086546696e-33, 'P': 274.688950680386}),
        ('xtunitroot fisher y_trend, dfuller trend lags(1)', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            13.85379467446139, 'df_L': 129.0, 'p_L': 0.0, 'L': -10.14802818518486, 'p_Z': 2.33765033936e-20, 'Z':
            -9.171258132971383, 'df_P': 50.0, 'p_P': 5.98721316592e-18, 'P': 188.5379467446139}),
        ('xtunitroot fisher y_stat, pperron lags(2)', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            18.90521043034073, 'df_L': 129.0, 'p_L': 0.0, 'L': -13.04255426789668, 'p_Z': 1.23510339041e-29, 'Z':
            -11.24430999522224, 'df_P': 50.0, 'p_P': 1.79110009719e-26, 'P': 239.0521043034073}),
        ('xtunitroot fisher y_trend, pperron trend lags(3)', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            23.11710585829238, 'df_L': 129.0, 'p_L': 0.0, 'L': -15.4673838809731, 'p_Z': 2.42492972186e-36, 'Z':
            -12.53424040777274, 'df_P': 50.0, 'p_P': 6.06589277204e-34, 'P': 281.1710585829238}),
        ('xtunitroot fisher y_ur, dfuller lags(0)', {'N': 950.0, 'N_t': 38.0, 'N_g': 25.0, 'p_Pm': 0.998423084528452,
            'Pm': -2.952332529557775, 'df_L': 124.0, 'p_L': 0.9998519306088955, 'L': 3.724158380492059, 'p_Z':
            0.9998872983592084, 'Z': 3.688699462389566, 'df_P': 50.0, 'p_P': 0.9999323989910021, 'P':
            20.47667470442225}),
        ('xtunitroot hadri y_stat', {'var': 0.0222222222222222, 'mu': 0.1666666666666667, 'p': 6.59664199679e-58, 'z':
            15.99800626638357, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot hadri y_trend, trend', {'var': 0.0017460317460317, 'mu': 0.0666666666666667, 'p':
            2.65598493071e-16, 'z': 8.104140484976575, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot hadri y_stat, robust', {'var': 0.0222222222222222, 'mu': 0.1666666666666667, 'p':
            7.12203567172e-30, 'z': 11.29279086276499, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot hadri y_trend, trend robust', {'var': 0.0017460317460317, 'mu': 0.0666666666666667, 'p':
            4.04335812127e-13, 'z': 7.159677353302635, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot hadri y_stat, kernel(bartlett 3)', {'lags': 3.0, 'var': 0.0222222222222222, 'mu':
            0.1666666666666667, 'p': 1.57105855028e-05, 'z': 4.162914017949824, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot hadri y_trend, trend kernel(parzen 4)', {'lags': 4.0, 'var': 0.0017460317460317, 'mu':
            0.0666666666666667, 'p': 0.01006823827733, 'z': 2.32379513614355, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot hadri y_stat, kernel(quadraticspectral 2) demean', {'lags': 2.0, 'var': 0.0222222222222222, 'mu':
            0.1666666666666667, 'p': 1.23798757549e-05, 'z': 4.216978551635339, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot hadri share, kernel(bartlett)', {'lags': 1.0, 'var': 0.0222222222222222, 'mu': 0.1666666666666667,
            'p': 2.78323424834e-10, 'z': 6.202241175242795, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
        ('xtunitroot hadri share, kernel(bartlett 12)', {'lags': 12.0, 'var': 0.0222222222222222, 'mu':
            0.1666666666666667, 'p': 0.003123330191962, 'z': 2.734544749596512, 'N_t': 38.0, 'N_g': 25.0, 'N':
            950.0}),
        ('xtunitroot hadri y_ur', {'var': 0.0222222222222222, 'mu': 0.1666666666666667, 'p': 0.0, 'z':
            81.28521264380574, 'N_t': 38.0, 'N_g': 25.0, 'N': 950.0}),
    ],
    'unbalanced': [
        ('xtunitroot ips y_stat', {'N': 901.0, 'N_t': 37.04, 'N_g': 25.0, 'p_zttildebar': 3.82444061861e-19,
            'zttildebar': -8.865022225768158, 'zt': -10.07467121764329, 'ttildebar': -2.867929029098188, 'tbar':
            -3.298430876206575}),
        ('xtunitroot ips y_stat, lags(1)', {'lags': 1.0, 'N': 876.0, 'N_t': 37.04, 'N_g': 25.0, 'p_wtbar':
            2.70647854172e-14, 'wtbar': -7.521552529324104}),
        ('xtunitroot ips y_trend, trend', {'N': 901.0, 'N_t': 37.04, 'N_g': 25.0, 'p_zttildebar': 3.41017488227e-34,
            'zttildebar': -12.13584355743149, 'zt': -14.73622481231693, 'ttildebar': -3.388909984922848, 'tbar':
            -4.119512432594903}),
        ('xtunitroot ips y_stat, lags(aic 3) demean', {'lagm': 0.44, 'N': 890.0, 'N_t': 37.04, 'N_g': 25.0, 'p_wtbar':
            5.8066094854e-25, 'wtbar': -10.25182305903123}),
        ('xtunitroot fisher y_stat, dfuller lags(1)', {'N': 926.0, 'N_t': 37.04, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            12.17311143718578, 'df_L': 129.0, 'p_L': 9.99200722163e-16, 'L': -9.036313215324974, 'p_Z':
            1.49636788806e-16, 'Z': -8.17361912622027, 'df_P': 50.0, 'p_P': 2.93616496289e-15, 'P':
            171.7311143718578}),
        ('xtunitroot fisher y_stat, pperron lags(2)', {'N': 926.0, 'N_t': 37.04, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            18.37303276317734, 'df_L': 129.0, 'p_L': 0.0, 'L': -12.74611539101225, 'p_Z': 8.04625397638e-29, 'Z':
            -11.07771617047159, 'df_P': 50.0, 'p_P': 1.50140148068e-25, 'P': 233.7303276317734}),
        ('xtunitroot fisher y_trend, dfuller trend lags(2) demean', {'N': 926.0, 'N_t': 37.04, 'N_g': 25.0, 'p_Pm': 0.0,
            'Pm': 8.47159913685452, 'df_L': 129.0, 'p_L': 1.13549947223e-10, 'L': -6.884865527768881, 'p_Z':
            1.51151814119e-11, 'Z': -6.645444368630429, 'df_P': 50.0, 'p_P': 1.05096253503e-09, 'P':
            134.7159913685452}),
    ],
    'gaps': [
        ('xtunitroot fisher y_stat, dfuller lags(1)', {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            12.42459490186851, 'df_L': 129.0, 'p_L': 4.4408920985e-16, 'L': -9.17621505231358, 'p_Z':
            7.26831340324e-17, 'Z': -8.260251159740255, 'df_P': 50.0, 'p_P': 1.17732093387e-15, 'P':
            174.2459490186851}),
        ('xtunitroot fisher y_stat, pperron lags(2)', {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            18.29799785116288, 'df_L': 129.0, 'p_L': 0.0, 'L': -12.70255171963684, 'p_Z': 1.20001815697e-28, 'Z':
            -11.04186337560063, 'df_P': 50.0, 'p_P': 2.02427778735e-25, 'P': 232.9799785116288}),
        ('xtunitroot fisher y_trend, pperron trend lags(1) demean', {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'p_Pm': 0.0,
            'Pm': 23.50258222509656, 'df_L': 129.0, 'p_L': 0.0, 'L': -15.699705087013, 'p_Z': 2.61706070664e-37,
            'Z': -12.7095403119768, 'df_P': 50.0, 'p_P': 1.2206358724e-34, 'P': 285.0258222509656}),
        ('xtunitroot fisher y_stat, dfuller drift lags(1)', {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'p_Pm': 0.0, 'Pm':
            24.05963566104975, 'df_L': 129.0, 'p_L': 0.0, 'L': -16.04088915610154, 'p_Z': 6.17992372125e-41, 'Z':
            -13.34682947673445, 'df_P': 50.0, 'p_P': 1.19413357257e-35, 'P': 290.5963566104975}),
    ],
}

# (command, statistics in Stata's display order, p-values, scalars)
CO_REF = {
    'balanced': [
        ('xtcointtest kao y_coint x1 x2', [-11.93247156156242, -9.605672115359619, -6.759196093934453,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 6.93800572548753e-12, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(2)', [-11.93247156156242, -9.605672115359619, -4.556698997883,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 2.59819035741682e-06, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 2.0}),
        ('xtcointtest kao y_coint x1 x2, lags(aic 4)', [-11.93247156156242, -9.605672115359619, -6.759196093934453,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 6.93800572548753e-12, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(bic 3)', [-11.93247156156242, -9.605672115359619, -6.759196093934453,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 6.93800572548753e-12, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(hqic 3)', [-11.93247156156242, -9.605672115359619, -6.759196093934453,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 6.93800572548753e-12, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(parzen nwest)', [-6.437161931397507, -6.923197245535006,
            -4.327837230311088, -22.03416534160923, -11.83499981838438], [6.08639805221856e-11,
            2.20778950676959e-12, 7.52903412093442e-06, 0.0, 0.0], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0,
            'hac_lagm': 2.96, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(quadraticspectral nwest)', [-6.580542377795333, -7.011282610326137,
            -4.406219939516714, -22.03416534160923, -11.83499981838438], [2.34368080498371e-11,
            1.18072218668885e-12, 5.25950791430141e-06, 0.0, 0.0], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0,
            'hac_lagm': 2.671215885935577, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(bartlett 3)', [-9.226396869563485, -8.452017118392522,
            -5.700334823061792, -22.03416534160923, -11.83499981838438], [0.0, 0.0, 5.97861704587643e-09, 0.0,
            0.0], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(parzen 4)', [-9.117665257514327, -8.399290150208977, -5.652505037171746,
            -22.03416534160923, -11.83499981838438], [0.0, 0.0, 7.90630816371873e-09, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 4.0, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(quadraticspectral 3)', [-5.099739346248487, -6.046612168315687,
            -3.551498806578849, -22.03416534160923, -11.83499981838438], [1.70060766135904e-07,
            7.39616257128262e-10, 0.0001915218648038675, 0.0, 0.0], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0,
            'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, demean', [-11.86270664960335, -9.613652729178478, -6.686627900929858,
            -22.12653091869762, -11.89170650229377], [0.0, 0.0, 1.14185327859673e-11, 0.0, 0.0], {'N': 900.0,
            'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.12, 'adf_lags': 1.0}),
        ('xtcointtest kao y_nocoint x3', [-0.0603097158628451, 0.1044128720830925, 0.0727937730921696,
            -0.9787296741978273, -0.531000899593737], [0.475954481946946, 0.4584208540360076, 0.4709851130824034,
            0.1638567829472037, 0.2977090778873756], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.16,
            'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1', [-6.403766917715679, -5.554897126877048, -3.211355447490942, -12.93110172269069,
            -7.60524245180589], [7.57947038465545e-11, 1.38887651379704e-08, 0.0006605520822013933, 0.0,
            1.4210854715202e-14], {'N': 900.0, 'N_t': 36.0, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2', [-7.933086124029193, -11.5619239113556, -11.91667396128471],
            [1.11022302462516e-15, 0.0, 0.0], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, ar(same)', [15.90543960601535, -8.892351124908592, -9.95062445708055,
            -10.25025012821095], [0.0, 0.0, 0.0, 0.0], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 3.0,
            'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, trend', [-4.917162246932995, -11.1613216184619, -11.70514138640439],
            [4.39038869370911e-07, 0.0, 0.0], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, trend ar(same)', [10.55250468135963, -6.365922947497434,
            -10.55593663737756, -11.12508757796738], [0.0, 9.70594715710149e-11, 0.0, 0.0], {'N': 925.0, 'N_t':
            37.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, noconstant', [-8.140578736213637, -11.01895484710237, -11.58389381924969],
            [2.22044604925031e-16, 0.0, 0.0], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, noconstant ar(same)', [14.05195757584109, -7.576460877929645,
            -7.762755577563682, -8.136875747975067], [0.0, 1.77635683940025e-14, 4.10782519111308e-15,
            2.22044604925031e-16], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_nocoint x3', [-0.1772572677614, -1.028699523009726, -0.8044526783045121],
            [0.4296531573024058, 0.1518104470758375, 0.2105677942266643], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0,
            'hac_lagm': 1.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_nocoint x3, ar(same)', [-0.3061640315594927, -0.4408007885408282, -0.8470894700618288,
            -0.634992328917697], [0.3797398784481224, 0.3296786113106662, 0.198472626953571, 0.2627167090455403],
            {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 1.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1', [-8.38804744576069, -9.515674631006044, -9.796456204487747], [0.0, 0.0, 0.0],
            {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 1.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1, ar(same)', [13.5805873628594, -8.998420282688052, -8.104526819345779,
            -8.222495391118587], [0.0, 0.0, 2.22044604925031e-16, 1.11022302462516e-16], {'N': 925.0, 'N_t': 37.0,
            'N_g': 25.0, 'hac_lagm': 1.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, lags(aic 3) kernel(parzen 4)', [-8.252482372662376, -11.66339605409773,
            -12.34137667008601], [1.11022302462516e-16, 0.0, 0.0], {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0,
            'hac_lagm': 4.0, 'adf_lags': 3.0}),
        ('xtcointtest pedroni y_coint x1 x2, lags(2) kernel(bartlett 2) demean ar(same)', [5.730490901104309,
            -9.22022424660923, -10.41535292614958, -11.15211282647528], [5.00702002081965e-09, 0.0, 0.0, 0.0],
            {'N': 925.0, 'N_t': 37.0, 'N_g': 25.0, 'hac_lagm': 2.0, 'adf_lags': 2.0}),
        ('xtcointtest westerlund y_coint x1 x2', [-4.497294214003476], [3.44118602436e-06], {'N': 950.0, 'N_t': 38.0,
            'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, allpanels', [-3.276549382324879], [0.0005254196725938], {'N': 950.0,
            'N_t': 38.0, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, trend', [-4.441818617743994], [4.46008697608e-06], {'N': 950.0, 'N_t':
            38.0, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, trend allpanels', [-3.694430494787607], [0.0001101900816805], {'N':
            950.0, 'N_t': 38.0, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, demean', [-4.204801985241933], [1.30655415146e-05], {'N': 950.0, 'N_t':
            38.0, 'N_g': 25.0}),
        ('xtcointtest westerlund y_nocoint x3', [0.2303961517080052], [0.408891975611157], {'N': 950.0, 'N_t': 38.0,
            'N_g': 25.0}),
        ('xtcointtest westerlund y_nocoint x3, allpanels', [0.5887033051316138], [0.2780301601814287], {'N': 950.0,
            'N_t': 38.0, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1', [-3.419422260276709], [0.0003137713047796], {'N': 950.0, 'N_t': 38.0,
            'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1, allpanels', [-2.630108727196172], [0.0042678781961042], {'N': 950.0, 'N_t':
            38.0, 'N_g': 25.0}),
    ],
    'unbalanced': [
        ('xtcointtest kao y_coint x1 x2', [-11.66274035939279, -9.352761435926045, -6.492203872292001,
            -21.0380626821517, -11.44060214851297], [0.0, 0.0, 4.2294834301515e-11, 0.0, 0.0], {'N': 876.0, 'N_t':
            35.04, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(2)', [-11.66274035939279, -9.352761435926045, -4.44674596284885,
            -21.0380626821517, -11.44060214851297], [0.0, 0.0, 4.35904169127088e-06, 0.0, 0.0], {'N': 876.0,
            'N_t': 35.04, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 2.0}),
        ('xtcointtest kao y_coint x1 x2, lags(aic 4)', [-11.66274035939279, -9.352761435926045, -6.492203872292001,
            -21.0380626821517, -11.44060214851297], [0.0, 0.0, 4.2294834301515e-11, 0.0, 0.0], {'N': 876.0, 'N_t':
            35.04, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(bic 3)', [-11.66274035939279, -9.352761435926045, -6.492203872292001,
            -21.0380626821517, -11.44060214851297], [0.0, 0.0, 4.2294834301515e-11, 0.0, 0.0], {'N': 876.0, 'N_t':
            35.04, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, lags(hqic 3)', [-11.66274035939279, -9.352761435926045, -6.492203872292001,
            -21.0380626821517, -11.44060214851297], [0.0, 0.0, 4.2294834301515e-11, 0.0, 0.0], {'N': 876.0, 'N_t':
            35.04, 'N_g': 25.0, 'hac_lagm': 2.04, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(parzen nwest)', [-6.381030869028756, -6.805192516020473,
            -4.179068471453118, -21.0380626821517, -11.44060214851297], [8.79498696093606e-11,
            5.04574160231641e-12, 1.46352778364056e-05, 0.0, 0.0], {'N': 876.0, 'N_t': 35.04, 'N_g': 25.0,
            'hac_lagm': 2.88, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(quadraticspectral nwest)', [-6.497260482330029, -6.875744404911019,
            -4.241883643408586, -21.0380626821517, -11.44060214851297], [4.08978406696292e-11,
            3.08331138398898e-12, 1.10825802431824e-05, 0.0, 0.0], {'N': 876.0, 'N_t': 35.04, 'N_g': 25.0,
            'hac_lagm': 2.61236262397053, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(bartlett 3)', [-8.84769106615316, -8.153337986823585,
            -5.389803264841628, -21.0380626821517, -11.44060214851297], [0.0, 2.22044604925031e-16,
            3.52674286441257e-08, 0.0, 0.0], {'N': 876.0, 'N_t': 35.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(parzen 4)', [-8.723152226811983, -8.092380223762497, -5.334516276132033,
            -21.0380626821517, -11.44060214851297], [0.0, 3.33066907387547e-16, 4.78997975728035e-08, 0.0, 0.0],
            {'N': 876.0, 'N_t': 35.04, 'N_g': 25.0, 'hac_lagm': 4.0, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, kernel(quadraticspectral 3)', [-4.967330450312209, -5.885676626023386,
            -3.364800703957176, -21.0380626821517, -11.44060214851297], [3.39404310767399e-07,
            1.98214378244188e-09, 0.0003829948284431728, 0.0, 0.0], {'N': 876.0, 'N_t': 35.04, 'N_g': 25.0,
            'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest kao y_coint x1 x2, demean', [-11.19972794821966, -9.204174397675045, -6.303111972779335,
            -21.09676017912265, -11.48187996337761], [0.0, 0.0, 1.45863987555117e-10, 0.0, 0.0], {'N': 876.0,
            'N_t': 35.04, 'N_g': 25.0, 'hac_lagm': 2.28, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2', [-7.905592492570807, -11.12363146448759, -11.51292951304278],
            [1.33226762955019e-15, 0.0, 0.0], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, ar(same)', [16.39304487617764, -8.872435361386778, -9.601901848188248,
            -9.935737951067455], [0.0, 0.0, 0.0, 0.0], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 3.0,
            'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, trend', [-4.924562305603796, -10.77635405329381, -11.43018474555069],
            [4.22746935235807e-07, 0.0, 0.0], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, trend ar(same)', [11.10815867596482, -6.454659325579116,
            -10.26310170002152, -10.9239508197988], [0.0, 5.42312861284699e-11, 0.0, 0.0], {'N': 901.0, 'N_t':
            36.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, noconstant', [-8.143295815087246, -10.6529150989664, -11.21572364477924],
            [2.22044604925031e-16, 0.0, 0.0], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags':
            1.0}),
        ('xtcointtest pedroni y_coint x1 x2, noconstant ar(same)', [14.43576762792941, -7.519599337518041,
            -7.500228422105659, -7.871615462193507], [0.0, 2.74225087082414e-14, 3.1863400806742e-14,
            1.77635683940025e-15], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 3.0, 'adf_lags': 1.0}),
        ('xtcointtest pedroni y_coint x1 x2, lags(aic 3) kernel(parzen 4)', [-8.217507985212281, -11.21432333727871,
            -12.08775294272639], [1.11022302462516e-16, 0.0, 0.0], {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0,
            'hac_lagm': 4.0, 'adf_lags': 3.0}),
        ('xtcointtest pedroni y_coint x1 x2, lags(2) kernel(bartlett 2) demean ar(same)', [5.985209233823038,
            -9.11237121995623, -10.05205369919912, -10.79301327045559], [1.0805593086971e-09, 0.0, 0.0, 0.0],
            {'N': 901.0, 'N_t': 36.04, 'N_g': 25.0, 'hac_lagm': 2.0, 'adf_lags': 2.0}),
        ('xtcointtest westerlund y_coint x1 x2', [-4.5044613794477], [3.32707486983e-06], {'N': 926.0, 'N_t': 37.04,
            'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, allpanels', [-3.276769596211324], [0.0005250100782165], {'N': 926.0,
            'N_t': 37.04, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, trend', [-4.576017645004472], [2.3695529574e-06], {'N': 926.0, 'N_t':
            37.04, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, trend allpanels', [-3.821553133828483], [6.63069164649e-05], {'N':
            926.0, 'N_t': 37.04, 'N_g': 25.0}),
        ('xtcointtest westerlund y_coint x1 x2, demean', [-4.229034113681458], [1.17348379978e-05], {'N': 926.0, 'N_t':
            37.04, 'N_g': 25.0}),
    ],
}


# Stata 19.5 on independent random walks (moments/moments.do; data = random_walk_panel() below, seed 1001):
# xtcointtest pedroni v0 v1..vk, [trend|noconstant] ar(...) kernel(bartlett 2) lags(1) and westerlund v0 v1..vk.
# Checks the adjustment terms for 3, 5 and 7 regressors (including Stata's +25.889 for 7 regressors, no constant).
PED_RW = {
    (3, 'constant', 'panelspecific'): [1.207773794154498, -2.131240014323231, -1.033357621728937],
    (3, 'constant', 'same'): [0.1982658008884901, -0.4127710460063928, -2.60251426381739, -1.8414122110057],
    (3, 'trend', 'panelspecific'): [2.868321629344962, -0.6222627508809134, 0.4768437523913588],
    (3, 'trend', 'same'): [-2.071722593549265, 1.295342081544849, -1.420728929449205, -0.5492456573707198],
    (3, 'noconstant', 'panelspecific'): [1.195466099698045, -1.464243923704432, -1.082864380956791],
    (3, 'noconstant', 'same'): [-0.7895595307792884, 0.3510575148115738, -1.138549124398182, -0.9429506287475262],
    (5, 'constant', 'panelspecific'): [2.68368313606309, -2.270530127070878, -1.670566597161553],
    (5, 'constant', 'same'): [-1.429713304507725, 1.114699475311739, -2.689654127273865, -2.326621200914596],
    (5, 'trend', 'panelspecific'): [3.758944508979388, -1.772653035959825, -1.543562022647564],
    (5, 'trend', 'same'): [-3.084915889623625, 2.331932534745413, -2.15674410223322, -2.020530301273338],
    (5, 'noconstant', 'panelspecific'): [1.752037502457937, -3.124224840227596, -3.011728285552348],
    (5, 'noconstant', 'same'): [-1.720771236764929, 0.4774160484103827, -2.657320578797969, -2.589678250230823],
    (7, 'constant', 'panelspecific'): [3.899755386885412, -3.42084546663681, -3.033901143576383],
    (7, 'constant', 'same'): [-3.075728918455335, 2.456447246576852, -3.455582586880471, -3.398119496221107],
    (7, 'trend', 'panelspecific'): [4.931376592765564, -2.723048668609045, -2.687065715883347],
    (7, 'trend', 'same'): [-4.535716755137346, 3.447354964979082, -3.039757605446207, -3.227983374428735],
    (7, 'noconstant', 'panelspecific'): [3.035676972918475, -4.031316985577497, -3.803252858711838],
    (7, 'noconstant', 'same'): [-2.953323871766266, -17.58704862559908, -3.533981964306493, -3.391765736092911],
}
WES_RW = {
    (3, 'constant', 'somepanels'): -1.591555705729323, (3, 'constant', 'allpanels'): -1.351278962005015,
    (3, 'trend', 'somepanels'): 1.785209302519564, (3, 'trend', 'allpanels'): 0.5116938079272508,
    (5, 'constant', 'somepanels'): -1.064656619978478, (5, 'constant', 'allpanels'): -0.8845829748576959,
    (5, 'trend', 'somepanels'): 0.7415805383986793, (5, 'trend', 'allpanels'): 0.6050566603972414,
    (7, 'constant', 'somepanels'): -0.8031265424224956, (7, 'constant', 'allpanels'): -0.5860885639087551,
    (7, 'trend', 'somepanels'): 1.466149138210845, (7, 'trend', 'allpanels'): 0.9501263408991323,
}

# ----------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def frames():
    bal = pd.read_csv(DATA)
    unb = unbalanced(bal)
    gaps = unb[~((unb["id"] == 5) & (unb["year"] == 2000))].reset_index(drop=True)
    return {"balanced": bal, "unbalanced": unb, "gaps": gaps}


def run_stata(df, line):
    """Execute a Stata command through the transpiler, as the program does after `xtset id year`."""
    sess = ost.Session(df)
    sess.xtset("id", "year")
    code = transpile_stata_line(line, {})
    assert code is not None and not code.startswith("raise"), code
    code = code.replace("sess.run('panelUnitRoot', ", "sess.run('panelUnitRoot', plotSeries=False, ")
    ns = {"sess": sess}
    exec(code, ns)
    return ns["res"]


KEYMAP = {"adf_lagm": "adf_lags", "hac_lagm": "hac_lags", "lagm": "lags"}
P_KEYS = {"p", "p_td", "p_tds", "p_lrobust", "p_zttildebar", "p_wtbar", "p_P", "p_Z", "p_L", "p_Pm"}


def _check_ur(res, ref):
    var = res.options["vars"][0]
    r = dict(res.r[var])
    if r.get("cv"):
        r["cv_1"], r["cv_5"], r["cv_10"] = r["cv"]
    for k, v in ref.items():
        mk = KEYMAP.get(k, k)
        assert mk in r and r[mk] is not None, k
        if k in P_KEYS:
            assert r[mk] == pytest.approx(v, rel=1e-8, abs=1e-14), k
        else:
            assert r[mk] == pytest.approx(v, rel=1e-8, abs=1e-12), k


UR_CASES = [(sec, cmd, ref) for sec, rows in UR_REF.items() for cmd, ref in rows]
CO_CASES = [(sec, cmd, st, p, ex) for sec, rows in CO_REF.items() for cmd, st, p, ex in rows]


# ----------------------------------------------------------------------------- xtunitroot vs Stata 19.5
@pytest.mark.parametrize("sec,cmd,ref", UR_CASES, ids=[f"{s}:{c}" for s, c, _ in UR_CASES])
def test_xtunitroot_matches_stata(frames, sec, cmd, ref):
    _check_ur(run_stata(frames[sec], cmd), ref)


# ----------------------------------------------------------------------------- xtcointtest vs Stata 19.5
@pytest.mark.parametrize("sec,cmd,st,p,ex", CO_CASES, ids=[f"{s}:{c}" for s, c, *_ in CO_CASES])
def test_xtcointtest_matches_stata(frames, sec, cmd, st, p, ex):
    res = run_stata(frames[sec], cmd)
    r = res.r
    mine = [s[2] for s in r["stats"]]
    minep = [s[3] for s in r["stats"]]
    assert len(mine) == len(st)
    if r["test"] == "westerlund":
        assert mine == pytest.approx(st, abs=1e-6)
        assert minep == pytest.approx(p, abs=1e-7)
    else:          # Stata's residual variables are float (single precision): see module docstring
        assert mine == pytest.approx(st, rel=2e-6, abs=2e-6)
        assert minep == pytest.approx(p, rel=1e-4, abs=1e-8)
    for k, v in ex.items():
        mk = {"hac_lagm": "hac_lags"}.get(k, k)
        tol = 1e-7 if k == "hac_lagm" else 1e-12          # QS lags are real numbers computed from float residuals
        assert r[mk] == pytest.approx(v, rel=tol), k



def random_walk_panel(seed=1001, N=20, T=30):
    """Data of moments/moments.do (rw_1.csv): v0..v7 independent random walks, rounded to 6 decimals."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, N + 1):
        X = np.cumsum(rng.normal(size=(T, 8)), axis=0)
        for t in range(T):
            rows.append({"id": i, "t": t + 1, **{f"v{k}": round(float(X[t, k]), 6) for k in range(8)}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("key", list(PED_RW), ids=[f"pedroni-k{k}-{d}-{a}" for k, d, a in PED_RW])
def test_pedroni_adjustment_terms_many_regressors(key):
    k, det, ar = key
    df = random_walk_panel()
    r = U.pedroni_test(df, "id", "t", "v0", [f"v{j}" for j in range(1, k + 1)], det, ar, False, 1, "bartlett 2")
    assert [s[2] for s in r["stats"]] == pytest.approx(PED_RW[key], rel=2e-6, abs=2e-6)


@pytest.mark.parametrize("key", list(WES_RW), ids=[f"westerlund-k{k}-{d}-{a}" for k, d, a in WES_RW])
def test_westerlund_adjustment_terms_many_regressors(key):
    k, det, a = key
    df = random_walk_panel()
    r = U.westerlund_test(df, "id", "t", "v0", [f"v{j}" for j in range(1, k + 1)], det == "trend", a == "allpanels", False)
    assert r["stats"][0][2] == pytest.approx(WES_RW[key], abs=1e-6)

def test_stata_moment_tables_used():
    # Pedroni (1999) table 2 rows m2..m7 = number of regressors; single regressor = Pedroni (2004)
    assert U._PED["constant"][2][:2] == (11.754, 104.546)
    assert U._PED["constant"][1][6:8] == (-9.05, 35.98)
    assert U._PED["trend"][7][8:] == (-4.217, 0.518)
    # LLC (2002) table 2 as tabulated by Stata (model 3, T = 40: 0.971)
    assert U.llc_adjustment(40, 3) == (-0.637, 0.971)
    assert U.llc_adjustment(36, 2) == pytest.approx((-0.5402, 0.8636))
    # IPS (2003) table 2 lookup: first row/column whose heading is >= N, T (no interpolation)
    assert U.ips_critical(27, 34, False) == [-1.81, -1.73, -1.68]


# ----------------------------------------------------------------------------- Stata errors and option checks
@pytest.mark.parametrize("sec,cmd,msg", [
    ("unbalanced", "xtunitroot llc y_stat", "강한 균형 패널"),
    ("unbalanced", "xtunitroot ht y_stat", "Harris-Tzavalis test requires strongly balanced data".replace("-", "–")),
    ("unbalanced", "xtunitroot breitung y_stat", "강한 균형 패널"),
    ("unbalanced", "xtunitroot hadri y_stat", "강한 균형 패널"),
    ("gaps", "xtunitroot ips y_stat", "cannot have gaps"),
    ("gaps", "xtunitroot llc y_stat", "강한 균형 패널"),
    ("balanced", "xtunitroot fisher y_stat, dfuller noconstant lags(1)", "p-values are not available"),
    ("balanced", "xtunitroot fisher y_stat, dfuller", "must specify lags()"),
    ("balanced", "xtunitroot fisher y_stat, pperron drift lags(1)", "cannot specify drift with pperron"),
    ("balanced", "xtunitroot llc y_stat, trend noconstant", "cannot specify both noconstant and trend"),
    ("balanced", "xtunitroot hadri y_stat, robust kernel(bartlett 2)", "cannot specify both kernel() and robust"),
    ("balanced", "xtunitroot hadri y_stat, kernel(bartlett nwest)", "kernel()"),
    ("balanced", "xtunitroot ips y_stat, lags(9)", "not available for T = 38 with 9 lags"),
    ("balanced", "xtunitroot breitung y_stat, robust lags(15)", "more time periods net of lags than panels"),
    ("balanced", "xtunitroot ips y_stat, noconstant", "noconstant 사용 불가"),
    ("balanced", "xtunitroot llc y_stat, lags(aic)", None),
    ("balanced", "xtcointtest pedroni y_coint x1 x2, noconstant trend", "cannot specify both noconstant and trend"),
    ("balanced", "xtcointtest kao y_coint x1 x2, kernel(bartlett llc)", "llc"),
    ("balanced", "xtcointtest pedroni y_coint x1 x2, lags(0)", "1 이상"),
    ("balanced", "xtcointtest kao y_coint x1 x2, trend", "trend 사용 불가"),
    ("balanced", "xtcointtest westerlund y_coint y_coint x1", "종속변수"),
])
def test_errors_follow_stata(frames, sec, cmd, msg):
    if msg is None:              # lags(aic) without a number: Stata default maximum 1
        res = run_stata(frames[sec], cmd)
        assert res.r["y_stat"]["adf_method"] == "aic"
        return
    with pytest.raises(ValueError) as e:
        run_stata(frames[sec], cmd)
    assert msg in str(e.value)


def test_invalid_variables_name_the_available_ones(frames):
    sess = ost.Session(frames["balanced"])
    sess.xtset("id", "year")
    with pytest.raises(ValueError, match="사용 가능한 변수"):
        sess.run("xtunitroot", var="nope", test="ips")
    with pytest.raises(ValueError, match="xtset"):
        ost.Session(frames["balanced"]).run("panelUnitRoot", var="y_stat")
    with pytest.raises(ValueError, match="알 수 없는 값"):
        sess.run("panelUnitRoot", var="y_stat", test="kpss2")
    with pytest.raises(ValueError, match="최대 7개"):
        df = frames["balanced"].copy()
        for j in range(8):
            df[f"z{j}"] = df["x1"] + j
        s2 = ost.Session(df)
        s2.xtset("id", "year")
        s2.run("xtcointtest", dep="y_coint", covs=[f"z{j}" for j in range(8)], test="westerlund")


# ----------------------------------------------------------------------------- library cross-checks and notes
def test_ips_panel_t_equals_arch_adf(frames):
    from arch.unitroot import ADF
    df = frames["balanced"]
    P = U.URPanel(df, "id", "year", ["y_stat"], "t")
    t_arch = [float(ADF(y, lags=2, trend="c").stat) for y in P.series("y_stat")]
    r = U.ips_test(P, "y_stat", False, False, 2)
    assert r["tbar"] == pytest.approx(np.mean(t_arch), rel=1e-12)


def test_fisher_reports_choi_scaling(frames):
    res = run_stata(frames["balanced"], "xtunitroot fisher y_stat, dfuller lags(1)")
    r = res.r["y_stat"]
    N = r["N_g"]
    assert r["L_choi"] / r["L"] == pytest.approx(math.sqrt((5 * N + 4) / (5 * N + 3)))
    notes = " ".join(res.get_table("stats").notes)
    assert "Choi (2001)" in notes and f"{r['L_choi']:.4f}" in notes


def test_tables_and_notes(frames):
    res = run_stata(frames["balanced"], "xtunitroot llc y_trend, trend")
    info = {x["item"]: x["value"] for x in res.get_table("info").rows}
    assert info["H0"] == "패널에 단위근" and info["시간 추세"] == "포함" and info["패널 수"] == "25"
    assert [x["stat"] for x in res.get_table("stats").rows] == ["Unadjusted t", "Adjusted t*"]
    assert any("0.971" in n for n in res.get_table("stats").notes)        # T-tilde 36: Stata table value at T = 40
    res = run_stata(frames["balanced"], "xtunitroot ips y_stat")
    st = res.get_table("stats")
    assert [c["name"] for c in st.columns][-3:] == ["cv1", "cv5", "cv10"]
    assert [x["stat"] for x in st.rows] == ["t-bar", "t-tilde-bar", "Z-t-tilde-bar"]
    res = run_stata(frames["unbalanced"], "xtunitroot ips y_stat")
    assert res.r["y_stat"]["cv"] is None and "불균형" in " ".join(res.get_table("stats").notes)
    res = run_stata(frames["balanced"], "xtcointtest pedroni y_coint x1 x2")
    assert any("첫 패널" in n for n in res.get_table("stats").notes)
    res = run_stata(frames["balanced"], "xtcointtest kao y_nocoint x3")
    assert res.r["stats"][1][2] > 0 and any("부호" in n for n in res.get_table("stats").notes)
    for t in res.tables:
        for n in t.notes:
            assert "신택스" not in n and "자체 구현" not in n


def test_pedroni_per_panel_bandwidth_option(frames):
    df = frames["balanced"]
    a = U.pedroni_test(df, "id", "year", "y_coint", ["x1", "x2"], "constant", "panelspecific", False, 1, None)
    b = U.pedroni_test(df, "id", "year", "y_coint", ["x1", "x2"], "constant", "panelspecific", False, 1, None,
                       per_panel_nw=True)
    assert a["stats"][2][2] == pytest.approx(b["stats"][2][2])           # group ADF t does not use the kernel
    assert a["stats"][0][2] != pytest.approx(b["stats"][0][2])
    sess = ost.Session(df)
    sess.xtset("id", "year")
    r = sess.run("panelCoint", dep="y_coint", covs=["x1", "x2"], test="pedroni", nwPerPanel=True)
    assert r.r["stats"][0][2] == pytest.approx(b["stats"][0][2])
    assert "nwPerPanel=True" in r.syntax_python


# ----------------------------------------------------------------------------- session, aliases, several variables
def test_session_aliases_after_xtset(frames):
    sess = ost.Session(frames["balanced"])
    sess.xtset("id", "year")
    a = sess.run("xtunitroot", var="y_stat", test="ht", demean=True)
    b = sess.run("panel_unit_root", var="y_stat", test="ht", demean=True, entity="id", time="year")
    c = sess.run("panelUnitRoot", var="y_stat", test="ht", demean=True)
    assert a.r["y_stat"]["z"] == b.r["y_stat"]["z"] == c.r["y_stat"]["z"]
    assert a.syntax_python == ("res = sess.run('panelUnitRoot', entity='id', time='year', var='y_stat', test='ht', "
                               "demean=True)")
    k = sess.run("xtcointtest", dep="y_coint", covs="x1 x2", test="westerlund", allpanels=True)
    assert k.name == "panelCoint" and k.options["entity"] == "id" and len(k.r["stats"]) == 1
    assert k.model_stats == {"N": 950, "Groups": 25}
    assert a.plots, "series plot"
    assert "Harris–Tzavalis" in a.apa_narrative


def test_several_variables_in_one_run(frames):
    sess = ost.Session(frames["balanced"])
    sess.xtset("id", "year")
    r = sess.run("panelUnitRoot", vars=["y_stat", "y_ur"], test="ips", lags="aic 3", plotSeries=False)
    assert set(r.r) == {"y_stat", "y_ur"}
    assert r.get_table("stats_1") is not None and r.get_table("stats_2") is not None
    one = sess.run("panelUnitRoot", var="y_ur", test="ips", lags="aic 3", plotSeries=False)
    assert r.r["y_ur"]["wtbar"] == one.r["y_ur"]["wtbar"]
    assert extensions.stata_code("panelUnitRoot", r.options)[-2:] == ["xtunitroot ips y_stat, lags(aic 3)",
                                                                     "xtunitroot ips y_ur, lags(aic 3)"]


# ----------------------------------------------------------------------------- transpiler
def test_transpile_lines():
    idx = "entity=getattr(sess, 'panel_spec', {}).get('idVar'), time=getattr(sess, 'panel_spec', {}).get('timeVar')"
    assert transpile_stata_line("xtunitroot llc y, lags(aic 10) demean", {}) == \
        f"res = sess.run('panelUnitRoot', {idx}, var='y', test='llc', lags='aic 10', demean=True)"
    assert transpile_stata_line("xtunitroot fisher y, dfuller drift lags(2)", {}) == \
        f"res = sess.run('panelUnitRoot', {idx}, var='y', test='fisher', method='dfuller', drift=True, lags=2)"
    assert transpile_stata_line("xtunitroot hadri y, kernel(bartlett 5) tr", {}) == \
        f"res = sess.run('panelUnitRoot', {idx}, var='y', test='hadri', kernel='bartlett 5', trend=True)"
    assert transpile_stata_line("xtunitroot fisher y, pp lags(1) regress", {}) == \
        f"res = sess.run('panelUnitRoot', {idx}, var='y', test='fisher', method='pperron', lags=1)"
    assert transpile_stata_line("xtcointtest pedroni y x1 x2, ar(same) trend lags(aic 3)", {}) == \
        f"res = sess.run('panelCoint', {idx}, dep='y', covs=['x1', 'x2'], test='pedroni', ar='same', trend=True, lags='aic 3')"
    assert transpile_stata_line("xtcointtest westerlund y x, somepanels", {}) == \
        f"res = sess.run('panelCoint', {idx}, dep='y', covs=['x'], test='westerlund')"
    for line, frag in [("xtunitroot ips y if year > 1990", "if·in"), ("xtunitroot ips D.y", "시계열 연산자"),
                       ("xtunitroot kpss y", "검정 'kpss' 미지원"), ("xtunitroot llc y, nointerpolate", "옵션"),
                       ("xtunitroot llc y z", "변수 1개"), ("xtcointtest kao y", "설명변수"),
                       ("xtcointtest pedroni y x, ar(foo)", "ar(foo)"), ("xtcointtest johansen y x", "미지원")]:
        out = transpile_stata_line(line, {})
        assert out.startswith("raise NotImplementedError") and frag in out, (line, out)


# ----------------------------------------------------------------------------- Stata and R export
def test_stata_export():
    p = {"entity": "region", "time": "year", "var": "y", "test": "llc", "lags": "aic 4", "kernel": "parzen nwest",
         "demean": True}
    assert extensions.stata_code("panelUnitRoot", p) == [
        "capture confirm numeric variable region", "if _rc encode region, generate(_panel_id)",
        "if _rc xtset _panel_id year", "else xtset region year",
        "xtunitroot llc y, demean lags(aic 4) kernel(parzen nwest)"]
    p = {"entity": "id", "time": "year", "var": "y", "test": "fisher", "method": "pperron", "trend": True, "lags": 2}
    assert extensions.stata_code("xtunitroot", p)[-1] == "xtunitroot fisher y, pperron trend lags(2)"
    p = {"entity": "id", "time": "year", "dep": "y", "covs": ["x1", "x2"], "test": "pedroni", "ar": "same",
         "trend": True}
    assert extensions.stata_code("panelCoint", p)[-1] == "xtcointtest pedroni y x1 x2, trend ar(same)"
    p.update(nwPerPanel=True)
    assert extensions.stata_code("panelCoint", p)[-2].startswith("* nwPerPanel")
    # form parameters: one deterministics select, kernel type + lags
    p = {"entity": "id", "time": "year", "vars": ["y"], "test": "llc", "deterministics": "trend",
         "kernelType": "quadraticspectral", "kernelLags": "nwest"}
    assert extensions.stata_code("panelUnitRoot", p)[-1] == "xtunitroot llc y, trend kernel(quadraticspectral nwest)"
    p = {"entity": "id", "time": "year", "dep": "y", "covs": ["x"], "test": "pedroni", "deterministics": "none"}
    assert extensions.stata_code("panelCoint", p)[-1] == "xtcointtest pedroni y x, noconstant"


def test_form_parameters_and_failed_panels(frames):
    sess = ost.Session(frames["balanced"])
    sess.xtset("id", "year")
    a = sess.run("panelUnitRoot", var="y_trend", test="llc", deterministics="trend", kernelType="parzen",
                 kernelLags="nwest", lags="", plotSeries=False)
    b = run_stata(frames["balanced"], "xtunitroot llc y_trend, trend kernel(parzen nwest)")
    assert a.r["y_trend"]["tds"] == b.r["y_trend"]["tds"]
    df = frames["balanced"]
    df = df[~((df["id"] == 3) & (df["year"] > 1991))]          # panel 3 keeps 5 years: dfuller lags(2) not computable
    r = run_stata(df, "xtunitroot fisher y_stat, dfuller lags(2)").r["y_stat"]
    # Stata 19.5 (ur_failed_panel.do): dfuller gives Z(t) = . and p = 1 for that panel; fisher keeps p = 1
    assert r["failed"] == ["3"] and r["df_P"] == 50 and r["N"] == 917 and r["N_t"] == pytest.approx(36.68)
    for k, v in {"P": 156.2877179046266, "Z": -7.745152631926995, "L": -8.410869354449035, "Pm": 10.62877179046265}.items():
        assert r[k] == pytest.approx(v, rel=1e-10), k


def test_r_export():
    r = extensions.r_code("panelUnitRoot", {"entity": "id", "time": "year", "var": "y", "test": "ips", "lags": 2})
    assert r[0] == 'pdata <- plm::pdata.frame(data, index = c("id", "year"))'
    assert r[1].startswith('plm::purtest(pdata$y, test = "ips", exo = "intercept", lags = 2, ips.stat = "Wtbar")')
    r = extensions.r_code("panelUnitRoot", {"entity": "id", "time": "year", "var": "y", "test": "fisher", "lags": 1,
                                            "demean": True})
    assert r[0].startswith("data$y_dm <- data$y - ave(data$y, data$year)")
    assert [x.split(",")[1].strip() for x in r[2:]] == ['test = "madwu"', 'test = "invnormal"', 'test = "logit"',
                                                        'test = "Pm"']
    r = extensions.r_code("panelUnitRoot", {"entity": "id", "time": "year", "var": "y", "test": "hadri", "robust": True})
    assert 'test = "hadri", exo = "intercept", Hcons = TRUE' in r[-1]
    assert "R에 동일 함수 없음" in extensions.r_code("panelUnitRoot", {"var": "y", "test": "ht"})[0]
    assert "R에 동일 함수 없음" in extensions.r_code("panelCoint", {"dep": "y", "covs": ["x"], "test": "kao"})[0]
    assert "pco::pedroni99m" in " ".join(extensions.r_code("xtcointtest", {"dep": "y", "covs": ["x"], "test": "pedroni"}))


def test_extension_registration():
    assert {"panelUnitRoot", "panelCoint"} <= extensions.ids()
    al = extensions.aliases()
    assert al["xtunitroot"] == "panelUnitRoot" and al["xtcointtest"] == "panelCoint"
    assert "panel_unitroot" not in extensions.load_errors()


def test_dialog_keys_are_read_only_for_their_test(frames):
    """The web form sends every key; values left from another test must not change or break the chosen test."""
    sess = ost.Session(frames["balanced"])
    sess.xtset("id", "year")
    gui = dict(vars=["y_stat"], detCommon="none", detSpecific="constant", adfLags="", breitungLags=0, fisherLags=1,
               llcKernel="bartlett", llcKernelLags="", hadriKernel="", hadriKernelLags="", breitungRobust=True,
               hadriRobust=False, htAltt=True, fisherMethod="dfuller", fisherDrift=True, demean=False, plotSeries=False)
    ips = sess.run("panelUnitRoot", **dict(gui, test="ips"))
    assert ips.r["y_stat"]["zttildebar"] == run_stata(frames["balanced"], "xtunitroot ips y_stat").r["y_stat"]["zttildebar"]
    br = sess.run("panelUnitRoot", **dict(gui, test="breitung"))
    assert br.r["y_stat"]["lrobust"] == run_stata(frames["balanced"],
                                                  "xtunitroot breitung y_stat, noconstant robust").r["y_stat"]["lrobust"]
    fi = sess.run("panelUnitRoot", **dict(gui, test="fisher"))
    assert fi.options["drift"] and fi.options["lags"] == 1
    cg = dict(dep="y_coint", covs=["x1", "x2"], detPedroni="none", detWesterlund="trend", coLags="aic 3",
              coKernel="parzen", coKernelLags="", pedroniAr="same", westerlundAll=True, pedroniNw=False, demean=False)
    w = sess.run("panelCoint", **dict(cg, test="westerlund"))
    assert w.r["stats"][0][2] == run_stata(frames["balanced"],
                                           "xtcointtest westerlund y_coint x1 x2, trend allpanels").r["stats"][0][2]
    k = sess.run("panelCoint", **dict(cg, test="kao"))
    assert k.r["adf_method"] == "aic" and k.r["hac_kernel"] == "parzen"

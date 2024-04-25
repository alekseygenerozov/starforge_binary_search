import numpy as np
import sys
import pickle
import h5py
import glob

import pytreegrav
import starforge_constants as sfc
import find_multiples_new2 as fmn
from find_multiples_new2 import cluster, system


def subtract_path(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff

def path_divide(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones(len(p1)) * np.inf
    filt = (~np.isinf(p1)) & (~np.isinf(p2))
    diff[filt] = p1[filt] / p2[filt]

    return diff

base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/"
r2 = "_TidesTrue_smaoFalse_mult4_ngridx_hmTrue_ft8.0_coFalse.p".replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)

start_snap = 100
end_snap = 489
sinks_all = []
ts = []
tags = []
accels = []
for ss in range(start_snap, end_snap + 1):
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
    sinks_all.append(tmp_sink)

    tmp_accel_stars = pytreegrav.Accel(tmp_sink[:,1:4],
                                       tmp_sink[:,-1], tmp_sink[:,-2], theta=0.5, G=sfc.GN, method='bruteforce')
    accels.append(tmp_accel_stars)
    ts.append(ss * np.ones(len(tmp_sink)))

ts = np.concatenate(ts)
ts.shape = (-1, 1)
sinks_all = np.vstack(sinks_all)
# sinks_all = np.hstack((ts, sinks_all))
accels = np.vstack(accels)
sinks_all = np.hstack((ts, sinks_all, accels))
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m", "ax", "ay", "az"))

utags = np.unique(sinks_all[:, 1])
utags_str = utags.astype(int).astype(str)
path_lookup = {}
for ii, uu in enumerate(utags):
    tmp_sel = sinks_all[sinks_all[:, 1] == uu]
    tmp_path1 = np.ones((end_snap + 1, len(sink_cols))) * np.inf
    tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel

    path_lookup[utags_str[ii]] = tmp_path1
############################################################################
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]

mult_lookup_file = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/\
M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/M2e4_snapshot_489_TidesTrue_smaoFalse_mult4_ngridx_hmFalse_ft8.0_coFalse.p"
with open(mult_lookup_file, "rb") as ff:
    cl_a = pickle.load(ff)
    mults_a = np.array([sys1.multiplicity for sys1 in cl_a.systems])
    ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)
    bin_ids = ids_a[mults_a==2]

internal_accel_lookup = {}
tidal_accel_lookup = {}
tides_norm_series = {}
for ii,row in enumerate(bin_ids):
    bin_key = str(np.sort(list(bin_ids[ii])))
    lrow = list(row)
    p1 = path_lookup[str(lrow[0])]
    p2 = path_lookup[str(lrow[1])]
    m1 = p1[:, mcol]
    m2 = p2[:, mcol]


    ai1 = np.ones((len(p1), 3)) * np.inf
    ai2 = np.ones((len(p1), 3)) * np.inf
    atot1 = path_lookup[str(lrow[0])][:, -3:]
    atot2 = path_lookup[str(lrow[1])][:, -3:]
    a_com_series = [[np.inf, np.inf, np.inf] if (np.isinf(m1[xx]) or np.isinf(m2[xx])) else (m1[xx] * atot1[xx] + m2[xx] * atot2[xx]) / (m1[xx] + m2[xx])  for xx in range(len(m1))]
    a_com_series = np.array(a_com_series)

    for jj in range(len(p1)):
        if ~np.isinf(p1[jj,0]) and ~np.isinf(p2[jj, 0]):
            pbin = np.vstack((p1[jj], p2[jj]))
            tmp_a = pytreegrav.Accel(pbin[:, pxcol:pzcol + 1], pbin[:, mcol], pbin[:, hcol], theta=0.5, G=sfc.GN, method='bruteforce')
            ai1[jj] = tmp_a[0]
            ai2[jj] = tmp_a[1]

    internal_accel_lookup[str(lrow[0])] = np.sum(ai1 * ai1, axis=1)**.5
    internal_accel_lookup[str(lrow[1])] = np.sum(ai2 * ai2, axis=1)**.5

    atides = subtract_path(atot1, a_com_series)
    atides = subtract_path(atides, ai1)
    atides_ck = np.copy(atides)
    tidal_accel_lookup[str(lrow[0])] = np.sum(atides * atides, axis=1)**.5
    atides = subtract_path(atot2, a_com_series)
    atides = subtract_path(atides, ai2)
    tidal_accel_lookup[str(lrow[1])] = np.sum(atides * atides, axis=1)**.5

    tmp_acc_norm_0 = path_divide(tidal_accel_lookup[str(lrow[0])], internal_accel_lookup[str(lrow[0])])
    tmp_acc_norm_1 = path_divide(tidal_accel_lookup[str(lrow[1])], internal_accel_lookup[str(lrow[1])])
    
    tmp_acc_norm = np.max((tmp_acc_norm_0, tmp_acc_norm_1), axis=0)
    tides_norm_series[bin_key] = np.copy(tmp_acc_norm)
# np.savez("{0}/tide_stars.npz".format(sys.argv[1]), tides_norm_series)
with open(base + aa  + "tide_stars.p", "wb") as ff:
    pickle.dump(tides_norm_series, ff)

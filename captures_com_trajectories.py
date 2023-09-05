#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
from analyze_multiples import snap_lookup
import find_multiples_new2
import cgs_const as cgs

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_ARR_IDX = 2
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7

sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/"
r2 = sys.argv[1].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
start_snap = 100
end_snap = 489

lookup = np.load(base + aa + "/system_lookup_table.npz")['arr_0']
bin_ids_unique = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
t_first = ic[:,0]


sinks_all = {}
for ss in range(start_snap, end_snap + 1):
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
    sinks_all[ss] = (tmp_sink)

coms = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf
vel_all1 = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf
vel_all2 = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf
seps_all = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf 
v_sec_all = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf 
v_prim_all = np.ones((len(bin_ids_unique), end_snap, 3)) * np.inf 

ms_all1 = np.ones((len(bin_ids_unique), end_snap)) * np.inf 
smas = np.ones((len(bin_ids_unique), end_snap)) * np.inf 
ms_all2 = np.ones((len(bin_ids_unique), end_snap)) * np.inf 
cang1 = np.ones((len(bin_ids_unique), end_snap)) * np.inf
cang2 = np.ones((len(bin_ids_unique), end_snap)) * np.inf 

##Getting coms of all the capture candidates at all times 
for ss in range(start_snap, end_snap + 1):
    print(ss)
    tmp_sink = sinks_all[ss]

    for jj, test_ids_s in enumerate(bin_ids_unique):
        test_ids = list(test_ids_s)
        if np.isin(test_ids[0], tmp_sink[:, 0].astype(int)):
            w1 = snap_lookup(tmp_sink, test_ids[0])[1]
            p1 = tmp_sink[w1, 1:4]
            v1 = tmp_sink[w1, 4:7]
            h1 = tmp_sink[w1, 7]
            m1 = tmp_sink[w1, -1]
            m1_gas = lookup[(lookup[:, LOOKUP_PID] == test_ids[0]) & (lookup[:, LOOKUP_SNAP] == ss)][0, LOOKUP_MTOT]
        else: 
            continue
        if np.isin(test_ids[1], tmp_sink[:, 0].astype(int)):
            w1 = snap_lookup(tmp_sink, test_ids[1])[1]
            p2 = tmp_sink[w1, 1:4]
            v2 = tmp_sink[w1, 4:7]
            h2 = tmp_sink[w1, 7]
            m2 = tmp_sink[w1, -1]
            m2_gas = lookup[(lookup[:, LOOKUP_PID] == test_ids[1]) & (lookup[:, LOOKUP_SNAP] == ss)][0, LOOKUP_MTOT]
        else:
            continue
        ##Make sure we get primary/secondary...    
        tmp_orb = find_multiples_new2.get_orbit(p1, p2, v1, v2, m1_gas, m2_gas, h1=h1, h2=h2)
        smas[jj, ss] = tmp_orb[0]
        coms[jj, ss] = (m1 * p1 + m2 * p2)/(m1+m2)
        vel_all1[jj, ss] = v1 / np.linalg.norm(v1)
        vel_all2[jj, ss] = v2 / np.linalg.norm(v2)
        ms_all1[jj, ss] = m1
        ms_all2[jj, ss] = m2
        
        if m1 > m2:
            v_sec_prim = v2 - v1
            tmp_sep = (p1 - p2)
            shat = tmp_sep / np.linalg.norm(tmp_sep)
            
            vhat = v_sec_prim / np.linalg.norm(v_sec_prim)
            cang1[jj, ss] = np.dot(vhat, shat)
            cang2[jj, ss] = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
        else:
            v_sec_prim = v1 - v2
            tmp_sep = (p2 - p1)
            shat = tmp_sep / np.linalg.norm(tmp_sep)
            
            vhat = v_sec_prim / np.linalg.norm(v_sec_prim)
            cang1[jj, ss] = np.dot(vhat, shat)
            cang2[jj, ss] = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
            
        seps_all[jj, ss] = np.linalg.norm(tmp_sep)

fig,ax = plt.subplots()
ax.set_ylabel("N")
ax.set_xlabel(r"$cos(\tilde{\theta}_i)$")

ci1 = [row[~np.isinf(row)][0] for row in cang1]
plt.hist(ci1, bins=np.arange(-1, 1.01, 0.1), histtype='step')
fig.savefig(base + aa + "/ang1_plot.pdf")


fig, ax = plt.subplots()
ci2 = [row[~np.isinf(row)][0] for row in cang2]
plt.hist(ci2, bins=np.arange(-1, 1.01, 0.1), histtype='step')
fig.savefig(base + aa + "/ang2_plot.pdf")



##Getting indices where stars are actually in the same system 
s_common = np.zeros((len(bin_ids_unique), end_snap))

for jj, test_ids_s in enumerate(bin_ids_unique):
    test_ids = list(test_ids_s)
    tmp_sys_idx_1 = lookup[test_ids[0] == lookup[:, 1].astype(int)][:, [0, 2]]
    tmp_sys_idx_2 = lookup[test_ids[1] == lookup[:, 1].astype(int)][:, [0, 2]]
    tmp1 = [str(row) for row in tmp_sys_idx_1]
    tmp2 = [str(row) for row in tmp_sys_idx_2]

    tmp_common, tmp_id1, tmp_id2 = np.intersect1d(tmp1, tmp2, return_indices=True)
    tmp_s_common = tmp_sys_idx_1[tmp_id1][:,0].astype(int)
    s_common[jj, tmp_s_common] = 1

s_common = s_common.astype(bool)

##Getting indices of closest stars at the time of capture
closest_stars_at_capture = np.zeros((len(bin_ids_unique), 3))

for jj, test_ids_s in enumerate(bin_ids_unique):
    test_ids = list(test_ids_s)
    ss = int(t_first[jj])
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))

    pos_all_stars = tmp_sink[:, 1:4] - coms[jj, ss]
    tmp_filt = (tmp_sink[:, 0] != test_ids[0]) & (tmp_sink[:, 0] != test_ids[1])
    pos_all_stars = pos_all_stars[tmp_filt]
    tmp_sink = tmp_sink[tmp_filt]
    order = np.argsort(np.sum(pos_all_stars * pos_all_stars, axis=1))
    closest_stars_at_capture[jj] = tmp_sink[order[:3], 0]
    
closest_stars_at_capture = closest_stars_at_capture.astype(int)

##Organizing star positions for plotting
pos_all_for_plot = np.ones((len(bin_ids_unique), end_snap, 5, 3)) * np.inf

for jj, test_ids_s in enumerate(bin_ids_unique):
    test_ids = list(test_ids_s)
    filt_existence = np.where(~np.isinf(coms[jj,:,0]))[0]
    ids_to_plot = np.concatenate(([test_ids[0], test_ids[1]], closest_stars_at_capture[jj]))
    for ss in filt_existence:
        for kk,tmp_id in enumerate(ids_to_plot):
            try:
                tmp_row, tmp_row_idx = snap_lookup(sinks_all[ss], tmp_id)
            except IndexError:
                continue
            pos_all_for_plot[jj, ss, kk] = tmp_row[1:4] - coms[jj, ss]

smas_first_together = np.zeros(len(bin_ids_unique))

for jj in range(len(bin_ids_unique)):
    first_together = np.where(~np.isinf(pos_all_for_plot[jj,:,0,0]))[0][0]
    smas_first_together[jj] = smas[jj, first_together]
np.savetxt(base + aa + "/smas_first_together", smas_first_together)


for jj, test_ids in enumerate(bin_ids_unique):
    fig,ax = plt.subplots(figsize=(10,10), constrained_layout=True)
    ax.set_xlabel('x [pc]')
    ax.set_ylabel('y [pc]')
    
    ax.plot(pos_all_for_plot[jj, :, 0, 0], pos_all_for_plot[jj, :, 0, 1], 'r--', alpha=0.3)
    ax.plot(pos_all_for_plot[jj, :, 1, 0], pos_all_for_plot[jj, :, 1, 1], 'b--', alpha=0.3)
    first_together = np.where(~np.isinf(pos_all_for_plot[jj,:,0,0]))[0][0]
    ax.plot(pos_all_for_plot[jj, first_together, 0, 0], pos_all_for_plot[jj,first_together, 0, 1], 'rs', alpha=0.3)
    ax.plot(pos_all_for_plot[jj, first_together, 1, 0], pos_all_for_plot[jj,first_together, 1, 1], 'bs', alpha=0.3)
    delta = np.abs(pos_all_for_plot[jj, first_together,0] - pos_all_for_plot[jj, first_together,1])
    ax.set_xlim(-5 * delta[0], 5 * delta[0])
    ax.set_ylim(-5 * delta[1], 5 * delta[1])

    ax.plot(pos_all_for_plot[jj, :, 0, 0][s_common[jj]], pos_all_for_plot[jj, :, 0, 1][s_common[jj]], 'r')
    ax.plot(pos_all_for_plot[jj, :, 1, 0][s_common[jj]], pos_all_for_plot[jj, :, 1, 1][s_common[jj]], 'b')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    ax.legend(title="a_i = {0:.2g}".format(smas[jj, first_together]))

    for kk in range(2,5):
        ax.plot(pos_all_for_plot[jj,:,kk,0], pos_all_for_plot[jj,:,kk,1], color='0.5', alpha=0.3)

    fig.savefig(base + aa + '/com_path_{0:03d}.pdf'.format(jj))
    plt.clf()    

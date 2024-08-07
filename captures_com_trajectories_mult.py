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
r2 = sys.argv[2].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
start_snap = 100
end_snap = 489

lookup = np.load(base + aa + "/system_lookup_table.npz")['arr_0']
ids_unique = np.load(base + aa + "/tri_ids.npz", allow_pickle=True)['arr_0']
ic = np.load(base + aa + "/tri_ic.npz", allow_pickle=True)['arr_0']
fst = np.load(base + aa + "/tri_fst.npz", allow_pickle=True)['arr_0']
tri_fst_sma = np.load(base + aa + "/tri_fst_sma.npz", allow_pickle=True)['arr_0']
tri_mults_filt = np.load(base + aa + "/tri_mults_filt.npz", allow_pickle=True)['arr_0']
tri_mults_filt = tri_mults_filt.astype(bool)
tri_mults_filt_b = np.load(base + aa + "/tri_mults_filt_b.npz", allow_pickle=True)['arr_0']
tri_mults_filt_b = tri_mults_filt_b.astype(bool)
tri_class = np.load(base + aa + "/tri_class.npz", allow_pickle=True)['arr_0']
tri_promise = np.load(base + aa + "/tri_promise.npz", allow_pickle=True)['arr_0']

tmp_sink_all = []
times_all = []
mass_all = []

##Separate into a different script
for ss in range(start_snap, end_snap + 1):
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
    tmp_sink = tmp_sink[np.argsort(tmp_sink[:, 0])]
    tmp_sink_all.append(tmp_sink)
    times_all.append(np.ones(len(tmp_sink)) * ss)
    lookup_sel = lookup[(lookup[:, LOOKUP_SNAP] == ss)]
    lookup_sel = lookup_sel[np.argsort(lookup_sel[:, LOOKUP_PID])]
    mass_all.append(lookup_sel[:, LOOKUP_MTOT])

    assert np.all(lookup_sel[:, LOOKUP_PID] == tmp_sink[:, 0])

tmp_sink_all = np.vstack(tmp_sink_all)
times_all = np.concatenate(times_all)
mass_all = np.concatenate(mass_all)

##Getting last snapshot together (MOVE TO DIFFERENT SCRIPT)
lst = np.zeros(len(ids_unique))
for ii, row in enumerate(ids_unique):
    row_li = list(row)
    tmax = end_snap
    for jj, pp in enumerate(row_li):
        tmax = min(np.max(times_all[tmp_sink_all[:, 0] == pp]), tmax)
    lst[ii] = tmax

###Getting COM at all times
coms = []
for ii, row in enumerate(ids_unique):
    row_li = list(row)
    mass_sel = []
    tmp_sel = []
    time_sel = []
    for jj, pp in enumerate(row_li):
        filt = (tmp_sink_all[:, 0] == pp) & (times_all >= fst[ii]) & (times_all <= lst[ii])
        tmp_sel.append(tmp_sink_all[filt][:, 1:4])
        mass_sel.append(np.atleast_2d(mass_all[filt]).T)
    com_series = np.sum([tmp_sel[kk] * mass_sel[kk] for kk in range(3)], axis=0)
    com_series = com_series / np.sum(mass_sel, axis=0)
    coms.append(com_series)

print("test")

##Plot positions of triple stars (and other companions)
cols = ['r', 'g', 'b']
lstyles=['-', '-.', '--']
for ii, row in enumerate(ids_unique):
    print(ic[ii], fst[ii])
    row_li = list(row)
    fig, ax = plt.subplots()
    ax.set_xlabel('x [pc]')
    ax.set_ylabel('y [pc]')
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('x2 [pc]')
    ax2.set_ylabel('y2 [pc]')

    for jj, pp in enumerate(row_li):
        filt = (tmp_sink_all[:, 0] == pp) & (times_all >= fst[ii]) & (times_all <= lst[ii])
        tmp_sel = tmp_sink_all[filt]
        mass_sel = mass_all[filt]
        times_sel = times_all[filt]

        ax.plot((tmp_sel[:, 1] - coms[ii][:, 0])[times_sel >= ic[ii]], (tmp_sel[:, 2] - coms[ii][:, 1])[times_sel >= ic[ii]],  color=cols[jj], alpha=0.5)
        ax.plot(tmp_sel[:, 1] - coms[ii][:, 0], tmp_sel[:, 2] - coms[ii][:, 1], '--',  color=cols[jj], alpha=0.5)
        ax.annotate("class={0} ai={1:.2g} {2} {3}\npromise={4}".format(tri_class[ii], tri_fst_sma[ii],
                                                                       tri_mults_filt[ii], tri_mults_filt_b[ii], tri_promise[ii]),
                    (0.99, 0.99), xycoords='axes fraction', va='top', ha='right')
        ##Add annotations
        ##Lab frame
        ax2.plot((tmp_sel[:, 1])[times_sel >= ic[ii]],
                (tmp_sel[:, 2])[times_sel >= ic[ii]], '-', color=cols[jj])
        ax2.plot(tmp_sel[:, 1], tmp_sel[:, 2], '--', color=cols[jj], alpha=0.5)
        ##Label the most promising capture candidates.
        ##Draw any companion stars

    fig.savefig("com_tri_pos_{0:03d}.pdf".format(ii))
    fig2.savefig("tri_pos_{0:03d}.pdf".format(ii))


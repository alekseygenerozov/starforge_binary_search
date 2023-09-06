import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

import cgs_const as cgs

##Columns in the lookup table...
LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6
LOOKUP_ECC = 7

##Paths
sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/"
r2 = sys.argv[1].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)

##Lookup tables...
bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
lookup = np.load(base + aa + "system_lookup_table.npz")['arr_0']
fate_tags = np.load(base + aa + "/fate_tags.npz", allow_pickle=True)['arr_0']
fate_tags = fate_tags.astype(str)
lookup_ref = np.load(base + aa.replace("hmTrue", "hmFalse") + "system_lookup_table.npz")['arr_0']

ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
t_first = ic[:, 0]
sma_first = ic[:, 1]
mass1_first = ic[:, 6]
mass2_first = ic[:, 7]
##Try to avoid hard-coding this(!)
snap_interval = 2.47e4

cols = ['r', 'b']
alphas = [1, 0.5]
for jj in range(len(bin_ids)):
    fig, axs = plt.subplots(figsize=(20, 10), ncols=2, constrained_layout=True)
    ax = axs[0]
    ax.set_ylim(1, 1e5)
    ax.set_xlabel('t [yr]')
    ax.set_ylabel('sma [au]')
    ax.tick_params(axis="x", which="both", rotation=30)

    # ax.legend(title=fate_tags[jj] + "\n" + str(bin_ids[jj]), loc='upper right')
    ax.loglog(t_first[jj] * snap_interval,
              sma_first[jj] * cgs.pc / cgs.au, 'o', color='k')

    ax = axs[1]
    ax.set_ylim(0.01, 100)
    ax.set_xlabel('t [yr]')
    ax.set_ylabel(r'$m [M_{\odot}]$')
    ax.tick_params(axis="x", which="both", rotation=30)

    ax.legend(title=fate_tags[jj] + "\n" + str(bin_ids[jj]), loc='upper right')
    ax.loglog(t_first[jj] * snap_interval,
              mass1_first[jj], 'o', color='k')

    for kk in (0, 1):
        test_id = list(bin_ids[jj])[kk]
        tmp_sys_idx = lookup[test_id == lookup[:, LOOKUP_PID].astype(int)]
        tmp_sys_idx_ref = lookup_ref[test_id == lookup_ref[:, LOOKUP_PID].astype(int)]

        axs[0].loglog(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_SMA] * cgs.pc / cgs.au,
                  color=cols[kk], alpha=alphas[kk])
        axs[0].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx_ref[:, LOOKUP_SMA] * cgs.pc / cgs.au,
                  color=cols[kk], alpha=alphas[kk], linestyle='--')
        axs[1].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_MTOT],
                  color=cols[kk], alpha=alphas[kk])
        axs[1].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_M],
                  color=cols[kk], alpha=alphas[kk], linestyle='--')
        fig.savefig(base + aa + "/sma_history_{0:03d}.pdf".format(jj))

    plt.clf()

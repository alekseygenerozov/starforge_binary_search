import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cgs_const as cgs
import matplotlib.transforms as mtransforms

##Columns in the lookup table...
LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6
LOOKUP_ECC = 7

##Paths
sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/"
r2 = sys.argv[1].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)

##Lookup tables...
bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
lookup = np.load(base + aa + "system_lookup_table.npz")['arr_0']
fate_tags = np.load(base + aa + "/fate_tags.npz", allow_pickle=True)['arr_0']
fate_tags = fate_tags.astype(str)
# lookup_ref = np.load(base + aa.replace("hmTrue", "hmFalse") + "system_lookup_table.npz")['arr_0']

ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
t_first = ic[:, 0]
sma_first = ic[:, 1]
mass1_first = ic[:, 6]
mass2_first = ic[:, 7]
##Try to avoid hard-coding this(!)
snap_interval = 2.47e4

cols = ['r', 'b']
alphas = [1, 0.5]
fig_idx = 0
# final_bins_arr_id = np.load(base + aa + "/final_bins_arr_id.npz")["arr_0"]

for jj in range(len(bin_ids[:5])):
    # if not np.isin(jj, final_bins_arr_id):
    #     continue
    fig_sma, ax_sma = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax_sma.set_ylim(20, 2e3)
    ax_sma.set_xlabel('t [yr]')
    ax_sma.set_ylabel('a [au]')

    fig_ecc, ax_ecc = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax_ecc.set_ylim(0, 1)
    ax_ecc.set_xlabel('t [yr]')
    ax_ecc.set_ylabel('e')

    fig, axs = plt.subplots(figsize=(50, 10), ncols=5, constrained_layout=True)
    ax = axs[0]
    ax.set_ylim(1, 1e5)
    ax.set_xlabel('t [yr]')
    ax.set_ylabel('sma [au]')
    ax.tick_params(axis="x", which="both", rotation=30)

    # ax.legend(title=fate_tags[jj] + "\n" + str(bin_ids[jj]), loc='upper right')
    ax.loglog(t_first[jj] * snap_interval,
              sma_first[jj] * cgs.pc / cgs.au, 'o', color='k')

    ax = axs[1]
    ax.set_ylim(0,1)
    # ax.set_ylim(0.01, 100)
    ax.set_xlabel('t [yr]')
    ax.set_ylabel('e')
    # ax.set_ylabel(r'$m [M_{\odot}]$')
    # ax.tick_params(axis="x", which="both", rotation=30)

    # ax.legend(title=fate_tags[jj] + "\n" + str(bin_ids[jj]), loc='upper right')
    # ax.loglog(t_first[jj] * snap_interval,
    #           mass1_first[jj], 'o', color='k')

    ##Could clean this up(!!) -- e.g. by using the lookup table generated in captures_com_trajectory...
    test_id1, test_id2 = list(bin_ids[jj])
    sys1_info = lookup[test_id1 == lookup[:, LOOKUP_PID].astype(int)]
    sys2_info = lookup[test_id2 == lookup[:, LOOKUP_PID].astype(int)]
    sys1_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys1_info]
    sys2_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys2_info]
    ##NOT ENOUGH!! STARS COULD BE IN THE SAME MULTIPLE BUT NOT BOUND--HAVE TO DO FURTHER FILTERING BASED ON SMA
    same_sys_filt1 = np.in1d(sys1_tag, sys2_tag)
    same_sys_filt2 = np.in1d(sys2_tag, sys1_tag)
    sys1_info = sys1_info[same_sys_filt1]
    sys2_info = sys2_info[same_sys_filt2]
    ##Making the stars have the same sma -- implies they are bound to each other...
    bound_filt = sys1_info[:, LOOKUP_SMA] == sys2_info[:, LOOKUP_SMA]

    for kk in (0, 1):
        test_id = list(bin_ids[jj])[kk]
        tmp_sys_idx = lookup[test_id == lookup[:, LOOKUP_PID].astype(int)]
        # tmp_sys_idx_ref = lookup_ref[test_id == lookup_ref[:, LOOKUP_PID].astype(int)]

        tmp_halo_mass = tmp_sys_idx[:, LOOKUP_MTOT] - tmp_sys_idx[:, LOOKUP_M]
        halo_snap = np.where(tmp_halo_mass > 0.01 * tmp_sys_idx[:, LOOKUP_M])[0]
        if len(halo_snap > 0):
            last_halo_snap = halo_snap[-1]
            axs[0].loglog(tmp_sys_idx[last_halo_snap, LOOKUP_SNAP] * snap_interval,
                          tmp_sys_idx[last_halo_snap, LOOKUP_SMA] * cgs.pc / cgs.au, "X",
                          color=cols[kk], alpha=alphas[kk], ms=10)
            ax_sma.loglog(tmp_sys_idx[last_halo_snap, LOOKUP_SNAP] * snap_interval,
                          tmp_sys_idx[last_halo_snap, LOOKUP_SMA] * cgs.pc / cgs.au, "X",
                          color=cols[kk], alpha=alphas[kk], ms=20)
            axs[1].semilogx(tmp_sys_idx[last_halo_snap, LOOKUP_SNAP] * snap_interval,
                            tmp_sys_idx[last_halo_snap, LOOKUP_ECC], "X",
                            color=cols[kk], alpha=alphas[kk], ms=10)
            ax_ecc.semilogx(tmp_sys_idx[last_halo_snap, LOOKUP_SNAP] * snap_interval,
                            tmp_sys_idx[last_halo_snap, LOOKUP_ECC], "X",
                            color=cols[kk], alpha=alphas[kk], ms=20)

        # assert np.all(tmp_sys_idx_ref[:, LOOKUP_SNAP] == tmp_sys_idx[:, LOOKUP_SNAP])
        axs[0].loglog([tmp_sys_idx[1, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[-1, LOOKUP_SNAP] * snap_interval], [20, 20], '--', color='0.5')
        axs[0].loglog(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_SMA] * cgs.pc / cgs.au,
                  color=cols[kk], alpha=alphas[kk])
        # axs[0].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx_ref[:, LOOKUP_SMA] * cgs.pc / cgs.au,
        #           color=cols[kk], alpha=alphas[kk], linestyle='--')
        ax_sma.loglog(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_SMA] * cgs.pc / cgs.au,
                  color=cols[kk], alpha=alphas[kk])
        # ax_sma.loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx_ref[:, LOOKUP_SMA] * cgs.pc / cgs.au,
        #           color=cols[kk], alpha=alphas[kk], linestyle='--')
        axs[1].semilogx(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_ECC],
                  color=cols[kk], alpha=alphas[kk])
        # axs[1].semilogx(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx_ref[:, LOOKUP_ECC],
        #           color=cols[kk], alpha=alphas[kk], linestyle='--')
        ax_ecc.semilogx(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_ECC],
                  color=cols[kk], alpha=alphas[kk])
        # ax_ecc.semilogx(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx_ref[:, LOOKUP_ECC],
        #           color=cols[kk], alpha=alphas[kk], linestyle='--')
        # axs[3].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_MTOT],
        #           color=cols[kk], alpha=alphas[kk])
        # axs[3].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_M],
        #           color=cols[kk], alpha=alphas[kk], linestyle='--')
        #
        # axs[4].loglog(tmp_sys_idx_ref[:, LOOKUP_SNAP] * snap_interval, tmp_sys_idx[:, LOOKUP_MULT],
        #           color=cols[kk], alpha=alphas[kk])



    ###Only plot points when the binary is in a common system(!!!)
    assert np.all(sys1_info[:, LOOKUP_ECC][bound_filt] == sys2_info[:, LOOKUP_ECC][bound_filt])
    axs[2].scatter(sys1_info[:, LOOKUP_SMA][bound_filt] * cgs.pc / cgs.au, sys1_info[:, LOOKUP_ECC][bound_filt],
              c=sys1_info[:, LOOKUP_SNAP][bound_filt], s=4)
    axs[2].semilogx(sys1_info[:, LOOKUP_SMA][bound_filt][-1] * cgs.pc / cgs.au, sys1_info[:, LOOKUP_ECC][bound_filt][-1], "rs")
    axs[2].semilogx(sys1_info[:, LOOKUP_SMA][bound_filt][0] * cgs.pc / cgs.au, sys1_info[:, LOOKUP_ECC][bound_filt][0], "s", color="k")
    ax_sma.semilogx(sys1_info[:, LOOKUP_SNAP][bound_filt][0] * snap_interval, sys1_info[:, LOOKUP_SMA][bound_filt][0] * cgs.pc / cgs.au,
                    "s", color="k")
    # if fig_idx==7:
    #     breakpoint()

    axs[2].set_xlabel("a [au]")
    axs[2].set_ylabel("e")
    fig.savefig(base + aa + "/sma_history_{0:03d}.png".format(fig_idx))
    # extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Saving it to a pdf file.
    # fig.savefig(base + aa + "/sma_history_p1_{0:03d}.png".format(fig_idx), bbox_inches=mtransforms.Bbox([[0, 1], [0, .25]]).transformed(
    #     fig.transFigure - fig.dpi_scale_trans))
    # fig_sma.savefig(base + aa + "/sma_history_p1_{0:03d}.png".format(fig_idx))
    # fig_ecc.savefig(base + aa + "/ecc_history_p1_{0:03d}.png".format(fig_idx))

    fig_idx += 1
    plt.clf()


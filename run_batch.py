import sys
import subprocess

def bash_command(cmd, **kwargs):
	'''Run command from the bash shell'''
	process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
	return process.communicate()[0]


# snap_base = "M2e4_R10_S0_T1_B1_Res271_n2_sol0.5_42_snapshot"
# start = int(sys.argv[1]) * 10
# end = (int(sys.argv[1]) + 1) * 10
# end = start + 1
start = 250
end = 251
snap_base = "snapshot"
print(start, end)
for ii in range(start, end, 1):
# 	bash_command("python find_multiples_new2.py --halo_mass_file halo_masses_sing_\
# npTrue_c0.5 --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
# 	bash_command("python find_multiples_new2.py --halo_mass_file halo_masses_sing_\
# npTrue_c0.5 --compress --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
	bash_command("python find_multiples_new2.py --halo_mass_file halo_masses_sing_\
npTrue_c0.5 --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 1.0))
#
# 	bash_command("python find_multiples_new2.py --tides_factor {2} --ngrid 4 --snap_base {0} {1}".format(snap_base, ii, 1.0))
# 	bash_command("python find_multiples_new2.py --tides_factor {2} --ngrid 4 --snap_base {0} {1}".format(snap_base, ii, 8.0))
# 	bash_command("python find_multiples_new2.py --tides_factor {2} --compress --ngrid 4 --snap_base {0} {1}".format(snap_base, ii, 8.0))

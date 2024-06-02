import sys
import subprocess

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]


with open("data_loc", "r") as ff:
	snap_base = ff.read()
start = int(sys.argv[1])
interval = 20
if len(sys.argv)==3:
        interval = int(sys.argv[2])
end = start + interval 
print(start, end)
##FOR TESTING
# start = 250
# end = 251
# snap_base = "snapshot"
for ii in range(start, end, 1):
        bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_npTrue_c0.5 --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
#         bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_\
# npTrue_c0.5 --compress --ngrid 8 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
#         bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_\
# npTrue_c0.5 --ngrid 8 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 1.0))
#
#         bash_command("python3 find_multiples_new2.py --ngrid 8 --snap_base {0} {1} --tides_factor {2} ".format(snap_base, ii, 8.0))
#         bash_command("python3 find_multiples_new2.py --compress --ngrid 8 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
#         bash_command("python3 find_multiples_new2.py --ngrid 8 --snap_base {0} {1} --tides_factor {2} ".format(snap_base, ii, 1.0))
#

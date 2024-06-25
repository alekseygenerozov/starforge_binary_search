import sys
import subprocess

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]


with open("data_loc", "r") as ff:
	snap_base = ff.read()
start = int(sys.argv[1])
interval = int(sys.argv[2])
end = start + interval 
print(start, end)
# snap_base = "snapshot"
for ii in range(start, end, 1):
        bash_command(f"python3 find_multiples_new2.py --halo_mass_file halo_masses/M2e4halo_masses_sing_npTrue_c0.5 --ngrid 1 --snap_base {snap_base} {ii} --tides_factor {sys.argv[3]}")

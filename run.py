import sys
import subprocess
from mpi4py import run
from subprocess import call
import os

# bash = "mpiexec -n 4 python ./mpi_test.py"
# bash ="ls"
# print("argument list:", (sys.argv[1:]))
# process = subprocess.Popen(bash.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# run.run_command_line('mpi_halo_exchange')

# call(bash)
# os.path.expandvars("$PATH")
# subprocess.run(["mpiexec", "-n", "4", "mpi_test.py"])

print(subprocess.Popen("mpiexec -n 4 ~/PycharmProjects/mpi_halo_exchange/mpi_test.py", shell=True, stdout=subprocess.PIPE).stdout.read())

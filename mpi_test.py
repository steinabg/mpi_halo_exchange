from mpi4py import MPI, run
import sys

hwmess = "Hello, world! I am process %d of %d on %s.\n"
myrank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()
procnm = MPI.Get_processor_name()
# if __name__ == '__main__':
#     run
sys.stdout.write(hwmess % (myrank, nprocs, procnm))
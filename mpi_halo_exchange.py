from mpi4py import MPI
import numpy as np
import sys
from sympy.ntheory import factorint  # To find processor grid size

Nx = 4
Ny = 30
IMG_X = Nx
IMG_Y = Ny

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

neighbor_processes = [0, 0, 0, 0]

local_petri_A = np.zeros(())
local_petri_B = np.zeros(())

ITERATIONS = 1

def define_px_py_dims(num_procs, ny, nx):
    '''

    :param num_procs: total number of processors
    :param ny: grid size in y direction
    :param nx: grid size in x direction
    :return: Number of processors in x and y direction
    '''
    assert num_procs > 0
    if np.sqrt(num_procs) == int(np.sqrt(num_procs)): # If square number
        p_y_dims = np.sqrt(num_procs)
        p_x_dims = np.sqrt(num_procs)
    elif num_procs % 2 == 0:
        if ny >= nx:
            p_x_dims = 2
            p_y_dims = num_procs/2.0
        else:
            p_x_dims = num_procs / 2.0
            p_y_dims = 2
    else:
        raise Exception("Please use an even or square number of processors!")
    return int(p_y_dims), int(p_x_dims)


def define_local_hexgrid_size(IMG_dim, p_xy_dim, my_dim_coord):
    '''

    :param IMG_dim: Size of combined grid in x or y direction
    :param p_xy_dim: Number of procs in x or y direction
    :param my_dim_coord: my row/col in the cartesian grid \
    if defining y direction size -> give my_row
    if defining x direction size -> give my_col
    :return: Size of the local grid in either x or y direction
    '''
    if int(IMG_dim / p_xy_dim) == IMG_dim / p_xy_dim:
        p_local_grid_dim = IMG_dim / p_xy_dim
    else:
        if my_dim_coord == 0:
            remainder = IMG_dim - np.floor(IMG_dim / p_xy_dim) * p_xy_dim
            p_local_grid_dim = np.floor(IMG_dim / p_xy_dim) + remainder
        else:
            p_local_grid_dim = np.floor(IMG_dim / p_xy_dim)
    return int(p_local_grid_dim)

def exchange_borders():
    if (num_iterations + 1) % 2:

        # Send data south and receive from north
        comm.Sendrecv(
            [local_petri_A[-2, :], 1, border_row_t],  # send the second last row
            neighbor_processes[DOWN],
            0,
            [local_petri_A[0, :], 1, border_row_t],  # recvbuf = first row
            neighbor_processes[UP],
            0
        )

        # # Send data north and receive from south
        comm.Sendrecv(
            [local_petri_A[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
            neighbor_processes[UP],  # destination
            1,  # sendtag
            [local_petri_A[-1, :], 1, border_row_t],  # recvbuf = last row
            neighbor_processes[DOWN],  # source
            1
        )
        #
        # Send west and receive from east
        local_petri_ev[:] = local_petri_A[:, 1].copy()
        comm.Sendrecv(
            [local_petri_ev, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            2,
            [local_petri_wb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            2
        )
        local_petri_A[:, -1] = local_petri_wb.copy()

        # # Send east and receive from west
        local_petri_wv[:] = local_petri_A[:, -2].copy()
        # print("my value = ", local_petri_wv)
        # print("my destination = ", local_petri_eb)

        comm.Sendrecv(
            [local_petri_wv, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            0,
            [local_petri_eb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            0
        )
        local_petri_A[:, 0] = local_petri_eb.copy()
    else:

        # Send data south and receive from north
        comm.Sendrecv(
            [local_petri_B[-2, :], 1, border_row_t],  # send the second last row
            neighbor_processes[DOWN],
            0,
            [local_petri_B[0, :], 1, border_row_t],  # recvbuf = first row
            neighbor_processes[UP],
            0
        )

        # # Send data north and receive from south
        comm.Sendrecv(
            [local_petri_B[1, :], 1, border_row_t],  # sendbuf = second row (1.row = border)
            neighbor_processes[UP],  # destination
            1,  # sendtag
            [local_petri_B[-1, :], 1, border_row_t],  # recvbuf = last row
            neighbor_processes[DOWN],  # source
            1
        )
        #
        # Send west and receive from east
        local_petri_ev[:] = local_petri_B[:, 1].copy()
        comm.Sendrecv(
            [local_petri_ev, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            2,
            [local_petri_wb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            2
        )
        local_petri_B[:, -1] = local_petri_wb.copy()

        # # Send east and receive from west
        local_petri_wv[:] = local_petri_B[:, -2].copy()
        # print("my value = ", local_petri_wv)
        # print("my destination = ", local_petri_eb)

        comm.Sendrecv(
            [local_petri_wv, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[RIGHT],
            0,
            [local_petri_eb, p_local_petri_y_dim + 2, MPI.DOUBLE],
            neighbor_processes[LEFT],
            0
        )
        local_petri_B[:, 0] = local_petri_eb.copy()


def iterateCA():
    pass


def gather_petri():
    send = np.zeros((p_local_petri_y_dim, p_local_petri_x_dim), dtype=np.double)
    TEMP = np.zeros((IMG_Y, IMG_X), dtype=np.double)

    if ((num_iterations + 1) % 2 == 0):
        send[:, :] = local_petri_B[1:-1, 1:-1].copy()
    else:
        send[:, :] = local_petri_A[1:-1, 1:-1].copy()

    if rank != 0:
        print("rank = {0}\n"
              "send = \n"
              "{1}".format(rank, send))
        comm.Send([send, MPI.DOUBLE], dest=0, tag=rank)
    else:
        i = 0 # Receive from rank i
        x_start = 0
        y_start = 0
        for row in range(p_y_dims):
            x_start = 0
            for col in range(p_x_dims):
                if i > 0:
                    dest = np.zeros((local_dims[i][0],local_dims[i][1]), dtype=np.double)
                    print("i = {0}, dest.shape = {1}".format(i,dest.shape))
                    # dest = TEMP[(row * local_dims[i][0]):((row + 1) * local_dims[i][0]),
                    #        col * local_dims[i][1]:((col + 1) * local_dims[i][1])]
                    comm.Recv([dest, MPI.DOUBLE], source=i, tag=i)
                    print("recved = \n"
                          "{0}\n"
                          "from rank {1}\n"
                          "put in TEMP[{2}:{3},{4}:{5}]\n".format(dest, i,y_start, y_start + local_dims[i][0],
                                                                 x_start, x_start + local_dims[i][1]))
                    print("x_start = {0}, y_start = {1}".format(x_start, y_start))
                    TEMP[(y_start):(y_start) + local_dims[i][0],
                            x_start:(x_start + local_dims[i][1])] = dest.copy()
                i += 1
                x_start += local_dims[i-1][1]
            y_start += local_dims[i-1][0]

            # Insert own local grid
            TEMP[0:local_dims[0][0], 0:local_dims[0][1]] = send.copy()
    comm.barrier()
        # print(TEMP)




    return TEMP

if __name__ == "__main__":

    # print(sys.argv[0])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # p_y_dims = 2 # num procs y dir
    # p_x_dims = 2 # num procs x dir
    p_y_dims, p_x_dims = define_px_py_dims(size, IMG_Y, IMG_X)
    # p_y_dims = int(np.sqrt(size))
    # p_x_dims = int(np.sqrt(size))

    cartesian_communicator = comm.Create_cart((p_y_dims, p_x_dims), periods=(False, False))

    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)

    neighbor_processes[UP], neighbor_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbor_processes[LEFT], neighbor_processes[RIGHT] = cartesian_communicator.Shift(1, 1)


    # p_local_petri_x_dim = int(IMG_X / p_x_dims)  # Må være delelig
    # p_local_petri_y_dim = int(IMG_Y / p_y_dims)
    p_local_petri_x_dim = define_local_hexgrid_size(IMG_X, p_x_dims, my_mpi_col)
    p_local_petri_y_dim = define_local_hexgrid_size(IMG_Y, p_y_dims, my_mpi_row)

    if rank == 0:
        local_dims = []
        r = 0
        for row in range(p_y_dims):
            for col in range(p_x_dims):
                local_dims.append([])
                local_dims[r].append(define_local_hexgrid_size(IMG_Y, p_y_dims, row))
                local_dims[r].append(define_local_hexgrid_size(IMG_X, p_x_dims, col))
                r += 1
        print("local_dims = \n", local_dims, "\n")
    print("Process = %s\n"
          "my_mpi_row = %s\n"
          "my_mpi_column = %s --->\n"
          "neighbour_processes[UP] = %s\n"
          "neighbour_processes[DOWN] = %s\n"
          "neighbour_processes[LEFT] = %s\n"
          "neighbour_processes[RIGHT] = %s\n"
          "local_x = %s, local_y = %s\n" % (rank, my_mpi_row, my_mpi_col,
                                                 neighbor_processes[UP], neighbor_processes[DOWN],
                                                 neighbor_processes[LEFT], neighbor_processes[RIGHT],
                                                 p_local_petri_x_dim, p_local_petri_y_dim))

    local_petri_A = np.zeros((p_local_petri_y_dim + 2, p_local_petri_x_dim + 2))
    local_petri_B = np.zeros((p_local_petri_y_dim + 2, p_local_petri_x_dim + 2))
    local_petri_wb = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_eb = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_ev = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    local_petri_wv = np.zeros((p_local_petri_y_dim + 2), dtype=np.double, order='C')
    TEMP = np.empty((1), dtype=np.double, order='C')

    local_petri_A[:] = comm.Get_rank()+1
    # print("process = ", comm.rank,"\n",
    #       local_petri_A)

    border_row_t = MPI.DOUBLE.Create_vector(p_local_petri_x_dim + 2,
                                            1,
                                            1)
    border_row_t.Commit()

    border_col_t = MPI.DOUBLE.Create_vector(p_local_petri_y_dim + 2,
                                            1,
                                            p_local_petri_x_dim + 2)
    border_col_t.Commit()

    for num_iterations in range(ITERATIONS):
        exchange_borders()
        iterateCA()

    comm.barrier()
    IMAGE = gather_petri()
    if rank == 0:
        print (IMAGE, "\nImage.shape = ", IMAGE.shape)
        print(len(np.where(IMAGE==6)[0]))

        for i in range(size):
            no = len(np.where(IMAGE==(i+1))[0])
            no_correct = local_dims[i][0] * local_dims[i][1]
            if no == no_correct:
                s = "True!"
            else:
                s = "False"
            print("found {0} occurrences of rank {1}\n"
                  "There should be {2} x {3} = {4} -- {5}".format(no, (i+1),
                                                           local_dims[i][0],
                                                           local_dims[i][1],
                                                           no_correct, s))
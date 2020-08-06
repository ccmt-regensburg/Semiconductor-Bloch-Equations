import numpy as np
from hfsbe.utility import evaluate_njit_matrix as ev_mat


def emission_exact(sys, paths, solution, E_dir, idxrange=None):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    n_time_steps = np.size(solution, axis=2)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time in range(n_time_steps):

        for i_path, path in enumerate(paths):
            path = np.array(path)
            if (idxrange is None):
                kx_in_path = path[:, 0]
                ky_in_path = path[:, 1]
                pathsolution = solution[:, i_path, :, :]
            else:
                kx_in_path = path[:, 0][idxrange[0]:idxrange[1]]
                ky_in_path = path[:, 1][idxrange[0]:idxrange[1]]
                pathsolution = solution[idxrange[0]:idxrange[1], i_path, :, :]

            h_deriv_x = ev_mat(sys.hderivfjit[0], kx=kx_in_path,
                               ky=ky_in_path)
            h_deriv_y = ev_mat(sys.hderivfjit[1], kx=kx_in_path,
                               ky=ky_in_path)

            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.Uf(kx=kx_in_path, ky=ky_in_path)
            U_h = sys.Uf_h(kx=kx_in_path, ky=ky_in_path)

            for i_k in range(np.size(kx_in_path)):

                dH_U_E_dir = np.matmul(h_deriv_E_dir[:, :, i_k], U[:, :, i_k])
                U_h_H_U_E_dir = np.matmul(U_h[:, :, i_k], dH_U_E_dir)

                dH_U_ortho = np.matmul(h_deriv_ortho[:, :, i_k], U[:, :, i_k])
                U_h_H_U_ortho = np.matmul(U_h[:, :, i_k], dH_U_ortho)

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0, 0])\
                    * np.real(pathsolution[i_k, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1, 1])\
                    * np.real(pathsolution[i_k, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                             * pathsolution[i_k, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0, 0])\
                    * np.real(pathsolution[i_k, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1, 1])\
                    * np.real(pathsolution[i_k, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                             * pathsolution[i_k, i_time, 2])

    return I_E_dir, I_ortho



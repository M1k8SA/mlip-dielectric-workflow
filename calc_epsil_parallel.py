#!/usr/bin/env python3

# Read eigenmodes from phonopy's output. See https://phonopy.github.io/phonopy/phonopy-module.html#

import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import numpy as np
from optparse import OptionParser
from multiprocessing import Pool

def print_vec(vecs):
    '''
    Print a vector.
    '''
    vec_str = " ".join(['{0:8.4f}'.format(elem) for elem in vecs])
    print(vec_str)

    return 0


def print_mat(mat):
    '''
    Print real and imaginary part of the eigenvector matrix
    '''
    print("Real part:")
    for vec in mat:
        vec_str = " ".join(['{0:8.4f}'.format(elem.real) for elem in vec])
        print(vec_str)
    print("Imag part:")
    for vec in mat:
        vec_str = " ".join(['{0:8.4f}'.format(elem.imag) for elem in vec])
        print(vec_str)

if __name__ == '__main__':

    # ================================================================================
    parser = OptionParser()
    parser.add_option("-c", "--charge", dest="fcharge", type="string", default="BORN_CHARGE",
                      help="Give Born charge file. [Default: %default]")
    parser.add_option("-l", "--lambda", dest="wstr", type="string", default="2,16,100",
                      help="Specify wavelength range, 'wmin,wmax,wnum'. [Default: %default]")
    parser.add_option("-t", "--relaxation", dest="itau", type="float", default=0,
                      help="Inverse relaxation time/scattering time in 'Lorentz oscillator model'. Default: 0.")
    parser.add_option("-n", "--number-of-cores", dest="ncore", type="int", default=12,
                      help="Number of cores used in parallel calculation. Default: 12")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="is_verbose", default=False,
                      help="Print verbosely. [Default: %default]")
    (options, args) = parser.parse_args()

    fcharge = options.fcharge
    is_verbose = options.is_verbose
    words = options.wstr.split(',')
    wmin, wmax, wnum = float(words[0]), float(words[1]), int(words[2])

    #print("The relaxation time is not considered.")
    itau = options.itau
    ncore = options.ncore
    # ================================================================================


    # Read Born charges from file
    print(f"Reading BORN CHARGE from '{fcharge}'")
    charges = np.loadtxt(fcharge, skiprows=1)

    phonon = phonopy.load("phonopy_disp.yaml")

    # Read masses and cell volume
    pcell = phonon.primitive
    masses = pcell.masses           # a.m.u
    vol = pcell.volume              # in Bhor^3
    #print(vol, masses)

    # Extract freuqnecies and eigenmodes at the Gamma point
    # Eigenvectors are given as the column vectors.
    phonon.run_mesh([1, 1, 1], with_eigenvectors=True)
    mdict = phonon.get_mesh_dict()

    eigen_freqs = mdict['frequencies'][0] # THz
    wleng = 299.792458 / eigen_freqs       # wavelength, micro meter
    eigen_vecs = mdict['eigenvectors'][0] # normalized

    ndim = len(eigen_freqs)
    natom = ndim // 3

    if is_verbose:
        print("Eigenvalues [THz]")
        print_vec(eigen_freqs)
        print("Eigenvalues [um]")
        print_vec(wleng)
        print("Eigenvectors")
        print_mat(eigen_vecs)

    ########################################################
    #
    # Compute ionic contribution
    #
    ########################################################

    # # Convert units to eV
    # eigen_freqs_dc = eigen_freqs * 1e12 / (3e14) # freq f [THz] to f/c, [um]
    # masses_c2 = masses * 931.49410372e6           # masess m to mc^2, [eV]
    # vol = vol * (5.29177e-5) ** 3          # Bhor^3 to um^3, [um^3]
    # epsilon_0 = 55.26349406                  # e^2 / eV / um

    # E = np.ones((ndim, ndim))

    # w = np.linspace(wmin, wmax, wnum) # um
    # omegas = 299.792458 / w * 1e12/(3e14) # f/c, [um]
    # #print(omegas)

    # charges = np.array(charges)
    # E = np.identity(3)
    # charges = charges[:, np.newaxis, np.newaxis] * E[np.newaxis, :, :]
    # #print(charges)

    # masses_c2 = masses * 931.49410372e6           # masess m to mc^2, [eV]
    # for o_id, omega in enumerate(omegas):
    #     e_tot = 0
    #     for mu in range(ndim):
    #         e_u = 0
    #         for alpha in range(natom):
    #             for beta in range(natom):
    #                 for i in range(3):
    #                     for j in range(3):
    #                         for k in range(3):
    #                             e_u += charges[alpha, i, j] * charges[beta, i, k] * eigen_vecs[alpha * 3 +j, mu] * eigen_vecs[beta * 3 + k, mu] / 3 / np.sqrt(masses_c2[alpha] *masses_c2[beta])/vol/epsilon_0/( (2*3.141592)**2 * (eigen_freqs_dc[mu]**2-omega**2 - 1j * omega*itau))

    #         e_tot += e_u
    #         #print(e_u.real)
    #     print(w[o_id], e_tot.real, e_tot.imag)


    ############################################################
    # Calculating frequency dependent dielectric tensor

    print("Calculating frequency dependent dielectric tensor")

    # Convert units to eV
    eigen_freqs_dc = eigen_freqs * 1e12 / (3e14) # freq f [THz] to f/c, [um]
    masses_c2 = masses * 931.49410372e6           # masess m to mc^2, [eV]
    vol = vol * (5.29177e-5) ** 3          # Bhor^3 to um^3, [um^3]
    epsilon_0 = 55.26349406                  # e^2 / eV / um

    E = np.ones((ndim, ndim))

    w = np.linspace(wmin, wmax, wnum) # um
    omegas = 299.792458 / w * 1e12/(3e14) # f/c, [um]
    #print(omegas)

    charges = np.reshape(charges, (-1, 3, 3))
    diel_infty = charges[0, :, :]
    charges = charges[1:, :, :]
    #print(charges)

    epsil_ion = np.zeros((wnum, 18))
    masses_c2 = masses * 931.49410372e6           # masess m to mc^2, [eV]

    def calc_diel(omega):
        e_tot = np.zeros((3,3), dtype='complex')
        for j in range(3):
            for k in range(3):
                for mu in range(ndim):
                    e_u = 0
                    for alpha in range(natom):
                        for beta in range(natom):
                            for i in range(3):
                                e_u += charges[alpha, i, j] * charges[beta, i, k] * eigen_vecs[alpha * 3 +j, mu] * eigen_vecs[beta * 3 + k, mu] / np.sqrt(masses_c2[alpha] *masses_c2[beta])/vol/epsilon_0/( (2*3.141592)**2 * (eigen_freqs_dc[mu]**2-omega**2 - 1j * omega*itau))

                    e_tot[j,k] += e_u
        #res = np.array([*(np.ravel(e_tot).real), *(np.ravel(e_tot).imag)])
        return [*(np.ravel(e_tot).real), *(np.ravel(e_tot).imag)]

    with Pool(ncore) as p:
        epsil_ion = np.array(p.map(calc_diel, omegas))

    # Save dielectric tensor to file 'espil_ion.dat'
    # Format:
    # 1st column: wavelength [um]
    # 2-19: real part of diel tensor.
    # 20-37: imag part of diel tensor.
    # tensor format: e00, e01, e02, e11, ..., e33
    datas = np.concatenate((w[:, np.newaxis], epsil_ion), axis=1)
    np.savetxt('epsil_ion.dat', datas)

    print('Saved to "epsil_ion.dat"')
    print('Format:')
    print('w[um] e00_r e01_r ... e33_r e00_i e01_i ... e33_i\n')

    ##########################################################
    # Calculating frequency independent dielectric tensor in limit of 0 Hz
    #
    # Due to numerical error in calculating eigenmodes frequency, which ought to have 3 zero frequency modes
    # In real calculation, these 3 zero frequency modes have non-zero eigen frequency.
    # To avoid such numerical, the dielectric tensor is caluclated in the limit of smallest eigen frequency

    print("Calculating dielectric in low frequency limit")
    # in limit of smallest eigen freuqency, which is scaled by a factor 'lamb'
    lamb = 10
    omega = lamb * np.min(np.abs(eigen_freqs_dc))

    e_tot = np.zeros((3,3), dtype='complex')
    for j in range(3):
        for k in range(3):
            for mu in range(ndim):
                e_u = 0
                for alpha in range(natom):
                    for beta in range(natom):
                        for i in range(3):
                            e_u += charges[alpha, i, j] * charges[beta, i, k] * eigen_vecs[alpha * 3 +j, mu] * eigen_vecs[beta * 3 + k, mu] / np.sqrt(masses_c2[alpha] *masses_c2[beta])/vol/epsilon_0/( (2*3.141592)**2 * (eigen_freqs_dc[mu]**2-omega**2 - 1j * omega*itau))

                e_tot[j,k] += e_u
    print("Dielectric tensor in high frequency limit:")
    print_mat(diel_infty)
    print("Dielectric tensor in low frequency limit (samllesteigen freqs * lamb):")
    print_mat(e_tot)
    print("Total Dielectric tensor:")
    print_mat(e_tot + diel_infty)

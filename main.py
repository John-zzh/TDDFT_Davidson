from pyscf import gto, dft, tddft
import numpy as np
import eigen_solver


def gen_TDDFT_mv(pyscf_TDDFT_vind):
    '''convert pyscf style matrix-vector function to my style 
        pyscf performs
        [  A  B ] [ X ] = [ AX + BY ]
        [ -B -A ] [ Y ]   [ -AY - BX ]
        hint: X & Y are row vectors in PySCF
    
        my style performs
        [  A  B ] [ X ] = [ AX + BY ]
        [  B  A ] [ Y ]   [ AY + BX ]  
        
        returns AX + BY and AY + BX
        hint: X & Y are column vectors in my eigensolver
    '''

    def TDDFT_mv(X, Y):
        XY = np.vstack((X,Y)).T
        U = pyscf_TDDFT_vind(XY)
        A_size = U.shape[1]//2
        U1 = U[:,:A_size].T
        U2 = -U[:,A_size:].T
        return U1, U2
    return TDDFT_mv

def gen_TDA_mv(pyscf_TDA_vind):
    def TDA_mv(X):
        '''convert pyscf style matrix-vector function to my style 
            X are row vectors in PySCF, but I need column vectors
        '''
        return pyscf_TDA_vind(X.T).T
    return TDA_mv

def main():

    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = '''
    C         -4.89126        3.29770        0.00029
    H         -5.28213        3.05494       -1.01161
    O         -3.49307        3.28429       -0.00328
    H         -5.28213        2.58374        0.75736
    H         -5.23998        4.31540        0.27138
    H         -3.22959        2.35981       -0.24953
    '''
    mol.basis = 'def2-SVP'
    mol.build()

    mf = dft.RKS(mol)
    mf = mf.density_fit()
    mf.xc = 'pbe0'
    mf.kernel()
    
    mf.verbose = 5
    '''
    TDA_vind & TDDFT_vind are matrix-vector product function
    hdiag is one dinension matrix, (n_occ*n_vir,)
    '''
    TDA_obj = tddft.TDA(mf)
    TDDFT_obj = tddft.TDDFT(mf)

    TDA_vind, hdiag = TDA_obj.gen_vind(mf)
    TDDFT_vind, Hdiag = TDDFT_obj.gen_vind(mf)


    Hartree_to_eV = 27.211385050


    print('############### TDDFT using home made solver starts ###############')
    '''Full TDDFT with hybrid functional, use Casida Davidson diagonalization
    '''
    TDDFT_mv = gen_TDDFT_mv(TDDFT_vind)
    
    energies, X, Y = eigen_solver.Davidson_Casida(TDDFT_mv, hdiag,
                                                N_states = 5,
                                                conv_tol = 1e-5,
                                                max_iter = 30)
    energies = energies*Hartree_to_eV
    print('home made solver TDDFT energies', energies)

    

    print('############### TDDFT using PySCF solver starts ###############')
    '''compare with pyscf TDDFT solver''' 
    TDDFT_obj.kernel(nstates=5)
    print('PySCF solver    TDDFT energies', TDDFT_obj.e*Hartree_to_eV)

    print()
    print()

    print('############### TDA using home made solver starts ###############')
    '''TDA, use Davidson diagonalization'''
    TDA_mv = gen_TDA_mv(TDA_vind)
    energies, X = eigen_solver.Davidson(TDA_mv, hdiag,
                                        N_states = 5,
                                        conv_tol = 1e-5,
                                        max_iter = 30)
    energies = energies*Hartree_to_eV
    print('home made solver TDA energies', energies)

    print('############### TDA using PySCF solver starts ###############')
    '''compare with pyscf TDA solver'''
    TDA_obj.nstates = 5
    TDA_obj.kernel()
    print('PySCF solver    TDA energies', TDA_obj.e*Hartree_to_eV)
if __name__ == '__main__':
    main()

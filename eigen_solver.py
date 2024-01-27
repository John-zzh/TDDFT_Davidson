
import numpy as np
import math_helper
import time

def Davidson(matrix_vector_product,
                    hdiag,
                    N_states=20,
                    conv_tol=1e-5,
                    max_iter=25,
                    single=False ):
    '''
    AX = XΩ
    Davidson frame, can use different initial guess and preconditioner
    initial_guess is a function, takes the number of initial guess as input
    preconditioner is a function, takes the residual as the input
    '''
    print('====== Davidson Diagonalization Starts ======')

    D_start = time.time()

    A_size = hdiag.shape[0]
    print('size of A matrix =', A_size)
    size_old = 0
    size_new = min([N_states+8, 2*N_states, A_size])

    max_N_mv = max_iter*N_states + size_new
    V_holder = np.zeros((A_size, max_N_mv),dtype=np.float32 if single else np.float64)
    W_holder = np.zeros_like(V_holder)
    sub_A_holder = np.zeros((max_N_mv,max_N_mv),dtype=np.float32 if single else np.float64)
    '''
    generate the initial guesss and put into the basis holder V_holder
    '''
    V_holder = math_helper.TDA_diag_initial_guess(V_holder=V_holder, N_states=size_new, hdiag=hdiag)

    # V_holder[:,:size_new] = initial_vectors[:,:]

    subcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    full_cost = 0
    print('step max||r||   sub_A.shape')
    for ii in range(max_iter):
    
        MV_start = time.time()
        W_holder[:, size_old:size_new] = matrix_vector_product(V_holder[:,size_old:size_new])
        MV_end = time.time()
        iMVcost = MV_end - MV_start
        MVcost += iMVcost

        subgencost_start = time.time()
        sub_A_holder = math_helper.gen_VW(sub_A_holder, V_holder, W_holder, size_old, size_new)
        sub_A = sub_A_holder[:size_new,:size_new]
        subgencost_end = time.time()
        subgencost += subgencost_end - subgencost_start
        sub_A = math_helper.utriangle_symmetrize(sub_A)
        # sub_A = math_helper.symmetrize(sub_A)

        '''
        Diagonalize the subspace Hamiltonian, and sorted.
        sub_eigenvalue[:N_states] are smallest N_states eigenvalues
        '''

        subcost_start = time.time()
        sub_eigenvalue, sub_eigenket = np.linalg.eigh(sub_A)
        sub_eigenvalue = sub_eigenvalue[:N_states]
        sub_eigenket = sub_eigenket[:,:N_states]
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        full_cost_start = time.time()
        full_guess = np.dot(V_holder[:,:size_new], sub_eigenket)
        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        AV = np.dot(W_holder[:,:size_new], sub_eigenket)
        residual = AV - full_guess * sub_eigenvalue

        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        print('{:<3d}  {:<10.4e} {:<5d}'.format(ii+1, max_norm, sub_A.shape[0]))
        if max_norm < conv_tol or ii == (max_iter-1):
            break

        index = [r_norms.index(i) for i in r_norms if i>conv_tol]

        new_guess = math_helper.TDA_diag_preconditioner(residual = residual[:,index],
                                            sub_eigenvalue = sub_eigenvalue[index],
                                            hdiag = hdiag)

        GScost_start = time.time()
        size_old = size_new
        V_holder, size_new = math_helper.Gram_Schmidt_fill_holder(V_holder, size_old, new_guess, double=True)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

    # energies = sub_eigenvalue*parameter.Hartree_to_eV

    D_end = time.time()
    Dcost = D_end - D_start

    if ii == max_iter-1:
        print('=== TDA Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)
        
    # print('energies:')
    # print(energies)
    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, Dcost))
    print('Maximum residual norm = {:.2e}'.format(max_norm))
    print('Final subspace size = {:d}'.format(sub_A.shape[0]))
    for enrty in ['MVcost','GScost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/Dcost))
    print('========== Davidson Diagonalization Done ==========')
    return sub_eigenvalue, full_guess

def Davidson_Casida(matrix_vector_product,
                        hdiag,
                        N_states=20,
                        conv_tol=1e-5,
                        max_iter=25,
                        single=False ):
    '''
    [ A B ] X - [1   0] Y Ω = 0
    [ B A ] Y   [0  -1] X   = 0

    '''
    print('======= TDDFT Eiegn Solver Statrs =======')

    TD_start = time.time()
    A_size = hdiag.shape[0]
    print('size of A matrix =', A_size)
    size_old = 0
    size_new = min([N_states+8, 2*N_states, A_size])

    max_N_mv = (max_iter+1)*N_states
    

    V_holder = np.zeros((A_size, max_N_mv),dtype=np.float32 if single else np.float64)
    W_holder = np.zeros_like(V_holder)

    U1_holder = np.zeros_like(V_holder)
    U2_holder = np.zeros_like(V_holder)

    VU1_holder = np.zeros((max_N_mv,max_N_mv),dtype=np.float32 if single else np.float64)
    VU2_holder = np.zeros_like(VU1_holder)
    WU1_holder = np.zeros_like(VU1_holder)
    WU2_holder = np.zeros_like(VU1_holder)

    VV_holder = np.zeros_like(VU1_holder)
    VW_holder = np.zeros_like(VU1_holder)
    WW_holder = np.zeros_like(VU1_holder)

    '''
    set up initial guess V= TDA initila guess, W=0
    '''

    V_holder = math_helper.TDA_diag_initial_guess(V_holder = V_holder,
                                         N_states = size_new,
                                         hdiag = hdiag)
    subcost = 0
    MVcost = 0
    GScost = 0
    subgencost = 0
    full_cost = 0
    # math_helper.show_memory_info('After Davidson initial guess set up')
    print('step maximum residual norm')
    for ii in range(max_iter):

        V = V_holder[:,:size_new]
        W = W_holder[:,:size_new]

        '''
        U1 = AV + BW
        U2 = AW + BV
        '''
        # print('size_old =', size_old)
        # print('size_new =', size_new)
        MV_start = time.time()
        U1_holder[:, size_old:size_new], U2_holder[:, size_old:size_new] = matrix_vector_product(
                                                            X=V[:, size_old:size_new],
                                                            Y=W[:, size_old:size_new])
        MV_end = time.time()
        MVcost += MV_end - MV_start

        U1 = U1_holder[:,:size_new]
        U2 = U2_holder[:,:size_new]

        subgenstart = time.time()

        '''
        [U1] = [A B][V]
        [U2]   [B A][W]

        a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                     [B A][W]            [U2]
        '''

        (sub_A, sub_B, sigma, pi,
        VU1_holder, WU2_holder, VU2_holder, WU1_holder,
        VV_holder, WW_holder, VW_holder) = math_helper.gen_sub_ab(
                      V_holder, W_holder, U1_holder, U2_holder,
                      VU1_holder, WU2_holder, VU2_holder, WU1_holder,
                      VV_holder, WW_holder, VW_holder,
                      size_old, size_new)
        # print('sub_A size =', sub_A.shape)
        subgenend = time.time()
        subgencost += subgenend - subgenstart

        '''
        solve the eigenvalue omega in the subspace
        '''
        subcost_start = time.time()
        omega, x, y = math_helper.TDDFT_subspace_eigen_solver(sub_A, sub_B, sigma, pi, N_states)
        subcost_end = time.time()
        subcost += subcost_end - subcost_start

        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        full_cost_start = time.time()
        X_full = np.dot(V,x)
        X_full += np.dot(W,y)

        Y_full = np.dot(W,x)
        Y_full += np.dot(V,y)

        R_x = np.dot(U1,x)
        R_x += np.dot(U2,y)
        R_x -= X_full*omega

        R_y = np.dot(U2,x)
        R_y += np.dot(U1,y)
        R_y += Y_full*omega

        full_cost_end = time.time()
        full_cost += full_cost_end - full_cost_start

        residual = np.vstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=0).tolist()
        max_norm = np.max(r_norms)
        print('{:<3d}  {:<10.4e}'.format(ii+1, max_norm))
        if max_norm < conv_tol or ii == (max_iter -1):
            # math_helper.show_memory_info('After last Davidson iteration')
            break

        index = [r_norms.index(i) for i in r_norms if i > conv_tol]

        '''
        preconditioning step
        '''
        X_new, Y_new = math_helper.TDDFT_diag_preconditioner(R_x = R_x[:,index],
                                                   R_y = R_y[:,index],
                                                 omega = omega[index],
                                                 hdiag = hdiag)
        '''
        GS and symmetric orthonormalization
        '''
        size_old = size_new
        GScost_start = time.time()
        V_holder, W_holder, size_new = math_helper.VW_Gram_Schmidt_fill_holder(V_holder=V_holder,
                                                                                W_holder=W_holder,
                                                                                X_new=X_new,
                                                                                Y_new=Y_new,
                                                                                m=size_old,
                                                                                double=False)
        GScost_end = time.time()
        GScost += GScost_end - GScost_start

        if size_new == size_old:
            print('All new guesses kicked out during GS orthonormalization')
            break

    TD_end = time.time()

    TD_cost = TD_end - TD_start

    if ii == (max_iter -1):
        print('=== TDDFT eigen solver Failed Due to Iteration Limit ===')
        print('current residual norms', r_norms)

    print('Finished in {:d} steps, {:.2f} seconds'.format(ii+1, TD_cost))
    print('final subspace', sub_A.shape[0])
    print('max_norm = {:.2e}'.format(max_norm))
    for enrty in ['MVcost','GScost','subgencost','subcost','full_cost']:
        cost = locals()[enrty]
        print("{:<10} {:<5.4f}s {:<5.2%}".format(enrty, cost, cost/TD_cost))
    # math_helper.show_memory_info('After Davidson Done')
    print('======= TDDFT Eigen Solver Done =======' )
    # energies = omega*parameter.Hartree_to_eV

    return omega, X_full, Y_full


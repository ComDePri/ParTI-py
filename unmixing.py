import numpy as np

def hinge(y):
    return np.maximum(-y, 0)

def soft_neg(y, tau):
    z = np.maximum(np.abs(y + (tau / 2)) - tau / 2, 0)
    return (z / (z + (tau / 2))) * (y + (tau / 2))

def svds(M, d):
    U,S,Vh = np.linalg.svd(M)
    Ud = U[:,:d]
    Vd = Vh[:,:d]
    Sd = S[:d]
    return Ud, Sd, Vd

def estimate_snr(R, r_m, x):
    L,N = R.shape
    p,_ = x.shape
    P_y = np.sum(R ** 2) / N
    P_x = (np.sum(x ** 2) / N) + (np.inner(r_m, r_m))
    if (P_y - P_x) < 1e-8:
        return 0
    return 10 * np.log10((P_x - (p / L) * P_y) / (P_y - P_x))


def VCA(R,p, verbose=True):
    snr_input = False
    L, N = R.shape
    assert p <= L
    r_m = np.mean(R, axis=1)
    R_m = np.tile(r_m[:,None], [1, N])
    R_o = R - R_m
    R_o_nn = R_o / np.sqrt(N)
    Ud, Sd, Vd = svds(R_o_nn @ R_o_nn.T, p)
    x_p = Ud.T @ R_o
    SNR = estimate_snr(R, r_m, x_p)
    SNR_th = 15 + 10 * np.log10(p)

    if SNR < SNR_th:
        d = p-1
        Ud = Ud[:,:d]
        Rp = Ud @ x_p[:d,:] + R_m
        x = x_p[:d,:]
        c = np.sqrt(np.max(np.sum(x ** 2, axis=0)))
        y = np.concatenate([x, np.ones((1,N)) * c])
    else:
        d = p
        R_nn = R / np.sqrt(N)
        Ud, Sd, Vd = svds(R_nn @ R_nn.T, d)
        x_p = Ud.T @ R
        Rp = Ud @ x_p

        x = Ud.T @ R
        u = np.mean(x, axis=1, keepdims=True)
        y = x / np.sum(x * u, keepdims=True)


    indice = np.zeros((p,), dtype=int)
    A = np.zeros((p,p))
    A[-1,0] = 1

    for i in range(p):
        w = np.random.rand(p,1)
        f = w - (A @ np.linalg.pinv(A) @ w)
        f = f / np.linalg.norm(f.flatten())
        v = f.T @ y
        ind_max = np.argmax(np.abs(v))
        indice[i] = ind_max
        A[:,i] = y[:, ind_max]

    Ae = Rp[:, indice]
    return Ae, indice, Rp


def SISAL(Y, p):
    #
    # Translated from original Matlab code with ParTI fix for non-improved target function
    #
    L,N = Y.shape
    MMiters = 80
    spherize = True
    tau = 1
    mu = (p*1000) / N
    M = 0
    tol_f = 1e-2

    slack = 1e-3
    energy_decreasing = False

    f_val_back = np.inf

    lam_sphe = 1e-8

    lam_quad = 1e-6

    AL_iters = 4

    flaged = False


    my = np.mean(Y, axis=1, keepdims=True)
    Y_orig = Y
    Y = Y - my
    Y_nn = Y / np.sqrt(N)
    Ud, D, _ = svds(Y_nn @ Y_nn.T, p-1)
    Y = Ud @ (Ud.T @ Y)
    Y = Y + my
    my_ortho = my - Ud @ (Ud.T  @ my)
    my_ortho /= np.linalg.norm(my_ortho)
    Ud = np.concatenate([Ud, my_ortho.reshape((L, 1))], axis=1)
    Y = Ud.T @ Y

    if spherize:
        Y = Ud @ Y
        Y -= my
        c_vec = 1. / np.sqrt(D + lam_sphe)
        ic_vec = np.sqrt(D + lam_sphe)
        Y = c_vec[:, None] * (Ud[:, :p-1].T @ Y)
        Y = np.concatenate([Y, np.ones((1,N))], axis=0)
        Y = Y / np.sqrt(p)


    Mvca, _, _ = VCA(Y, p)

    M = Mvca
    Ym = np.mean(M, axis=1, keepdims=True)
    dQ = M - Ym
    M = M + p * dQ

    Q0 = np.linalg.inv(M)
    Q = Q0

    # Build constant matrixes

    AAT = np.kron(Y @ Y.T, np.eye(p))
    B = np.kron(np.eye(p), np.ones((1,p)))
    qm = np.sum(np.linalg.inv(Y @ Y.T) @ Y, 1, keepdims=True)


    H = lam_quad * np.eye(p ** 2)
    F = H + mu * AAT
    IF = np.linalg.inv(F)

    G = IF @ B.T @ np.linalg.inv(B @ IF @ B.T)
    qm_aux = G @ qm
    G = IF - G @ B @ IF

    # Main body - sequence of quadratic-hinge subproblems

    # init

    Z = Q @ Y
    Bk = np.zeros_like(Z)

    for k in range(MMiters):
        IQ = np.linalg.inv(Q)
        g = - IQ.T
        g = g.flatten(order='F')[:,None]
        baux = H @ Q.flatten(order='F')[:,None] - g
        q0 = Q.flatten(order='F')[:,None]
        Q0 = Q

        if k == (MMiters - 1):
            AL_iters = 100

        while True:
            q = Q.flatten(order='F')[:,None]
            f0_val = - np.log(np.abs(np.linalg.det(Q))) + tau * np.sum(hinge(Q @ Y))

            f0_quad = (q - q0).T @ g + (1/2) * (q - q0).T @ H @ (q- q0) + tau * np.sum(hinge(Q @ Y))

            for i in range(1, AL_iters):
                dq_aux = Z + Bk
                dtz_b = dq_aux @ Y.T
                dtz_b = dtz_b.flatten(order='F')[:,None]
                b = baux + mu * dtz_b
                q = G @ b + qm_aux
                Q = np.reshape(q, (p,p), order='F')

                # solve hinge
                Z = soft_neg(Q @ Y - Bk, tau / mu)
                Bk = Bk - (Q @ Y - Z)

            f_quad = (q - q0).T @ g + (1 / 2) * (q - q0).T @ H @ (q - q0) + tau * np.sum(hinge(Q @ Y))
            f_val = - np.log(np.abs(np.linalg.det(Q))) + tau * np.sum(hinge(Q @ Y))

            if f0_quad >= f_quad:
                no_diff = 0
                while f0_val < f_val:
                    Q = (Q + Q0) / 2
                    f_valprev = f_val

                    f_val = - np.log(np.abs(np.linalg.det(Q))) + tau * np.sum(hinge(Q @ Y))
                    dif = f_valprev - f_val
                    if dif == 0:
                        no_diff += 1
                    if no_diff == 3:
                        print("Fail")
                        return None, None, None, None
                break
        if spherize:
            M = np.linalg.inv(Q)
            M = M * np.sqrt(p)
            M = M[:p - 1, :]
            M = Ud[:, :p - 1] @ (np.diag(ic_vec) @ M)
            M = M + my.reshape((p,1))
        else:
            M = Ud @ np.linalg.inv(Q)


    if spherize:
        M = np.linalg.inv(Q)
        M = M * np.sqrt(p)
        M = M[:p-1,:]
        M = Ud[:, :p - 1] @ (np.diag(ic_vec) @ M)
        M = M + my.reshape((p,1))
    else:
        M = Ud @ np.linalg.inv(Q)
    return M, Ud, my, D

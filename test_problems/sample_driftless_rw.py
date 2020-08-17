import numpy as np


def sample_single_exact_v0(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    - exact so no weighting needed
    :param v: k, 2 | time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return: n_samples, k, 2
    """
    k = v.shape[0]
    v_mask = ~np.isnan(v).any(axis=1)
    v_inds = np.arange(k)[v_mask]
    m = v_inds.size

    # S \in 2m, 2k - observations
    S = np.zeros((m, k))
    S[np.arange(m), v_inds] = 1.
    S = np.kron(S, np.eye(2))

    # C \in 2(k-1), 2k - differences
    C = np.zeros((k-1, k))
    inds = np.arange(0, (k-1)*k, k+1)
    C.flat[inds] = -1.
    C.flat[inds+1] = 1
    C = np.kron(C, np.eye(2))

    var = np.linalg.inv(S.T.dot(S) / sigma**2 + C.T.dot(C) / sigma_v**2)
    m_vec = S.T.dot(v[v_mask].reshape(-1)) / sigma**2
    m = var.dot(m_vec)
    L = np.linalg.cholesky(var)

    # n_samples, 2k
    v_s = m + (L.dot(np.random.randn(2*k, n_samples))).T
    return v_s.reshape(n_samples, k, 2)


def sample_single_exact_v1(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    - exact so no weighting needed
    :param v: k, 2 | time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return: n_samples, k, 2
    """
    k = v.shape[0]
    v_mask = ~np.isnan(v).any(axis=1)
    v_inds = np.arange(k)[v_mask]
    m = v_inds.size

    # S \in 2m, 2k - observations
    S = np.zeros((m, k))
    S[np.arange(m), v_inds] = 1.
    StS = np.kron(S.T.dot(S), np.eye(2))
    Sty = np.kron(S.T, np.eye(2)).dot(v[v_mask].reshape(-1))  #

    # C \in 2(k-1), 2k - differences
    C = np.zeros((k-1, k))
    inds = np.arange(0, (k-1)*k, k+1)
    C.flat[inds] = -1.
    C.flat[inds+1] = 1
    CtC = np.kron(C.T.dot(C), np.eye(2))

    var = np.linalg.inv(StS / sigma**2 + CtC / sigma_v**2)
    m_vec = Sty / sigma**2
    m = var.dot(m_vec)
    L = np.linalg.cholesky(var)

    # n_samples, 2k
    v_s = m + (L.dot(np.random.randn(2*k, n_samples))).T
    return v_s.reshape(n_samples, k, 2)


def sample_single_exact_v2(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    - exact so no weighting needed
    :param v: k, 2 | time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return: n_samples, k, 2
    """
    k = v.shape[0]
    v_mask = ~np.isnan(v).any(axis=1)
    v_inds = np.arange(k)[v_mask]
    m = v_inds.size

    # S \in 2m, 2k - observations
    S = np.zeros((m, k))
    S[np.arange(m), v_inds] = 1.
    StS0 = S.T.dot(S)
    Sty = np.kron(S.T, np.eye(2)).dot(v[v_mask].reshape(-1))

    # C \in 2(k-1), 2k - differences
    # C = np.zeros((k-1, k))
    # inds = np.arange(0, (k-1)*k, k+1)
    # C.flat[inds] = -1.
    # C.flat[inds+1] = 1
    # CtC0 = C.T.dot(C)
    # -
    CtC0 = np.zeros((k, k))
    inds = np.arange(1, k * k, k+1)
    CtC0.flat[inds] = -1
    CtC0.flat[np.arange(k, k * k, k+1)] = -1
    CtC0.flat[inds-1] = 2
    CtC0[0, 0] = 1
    CtC0[-1, -1] = 1

    inv0 = np.linalg.inv(StS0 / sigma**2 + CtC0 / sigma_v**2)
    var = np.kron(inv0, np.eye(2))
    m_vec = Sty / sigma**2
    m = var.dot(m_vec)
    L = np.linalg.cholesky(var)
    # L = np.kron(np.linalg.cholesky(inv0), np.eye(2))
    # -
    # inv0 = np.linalg.inv(StS0 / sigma**2 + CtC0 / sigma_v**2)
    # var = np.kron(inv0, np.eye(2))
    # m_vec = Sty / sigma**2
    # m = var.dot(m_vec)
    # L = np.kron(np.linalg.cholesky(inv0), np.eye(2))
    # -
    # spc = StS0 / sigma ** 2 + CtC0 / sigma_v ** 2
    # Linv = np.linalg.cholesky(spc)
    # m_vec = Sty / sigma ** 2
    # m = np.linalg.solve(np.kron(spc, np.eye(2)), m_vec)
    # L = np.kron(np.linalg.inv(Linv), np.eye(2))

    # n_samples, 2k
    v_s = m + (L.dot(np.random.randn(2*k, n_samples))).T
    return v_s.reshape(n_samples, k, 2)


def sample_n_exact_v0(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    :param v: n, k, 2 | n independent time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value in each series
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return: n, n_samples, k, 2
    """
    n, k = v.shape[:2]
    v_s = np.empty((n, n_samples, k, 2))
    for i in range(n):
        v_s[i, ...] = sample_single_exact_v0(
            v[i, ...], n_samples, sigma, sigma_v)
    return v_s


def sample_single_v0(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    - calculate nll for v_s set consecutively from v since
      they are related by sigma_v
    :param v: k, 2 | time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return:
        v_s: n_samples, k, 2
        nll: n_samples,
    """
    k = v.shape[0]
    v_s = np.empty((n_samples, k, 2))
    v_mask = ~np.isnan(v).any(axis=1)
    v_inds = np.arange(k)[v_mask]
    v_s[:, v_mask, :] = v[v_mask]
    v_s[:, v_mask, :] += np.random.randn(n_samples, v_inds.size, 2) * sigma
    c_inds = v_inds[:-1][(v_inds[1:] - v_inds[:-1]) == 1]
    nll = ((v_s[:, c_inds, :] -
            v_s[:, c_inds+1, :])**2).sum(axis=(1, 2)) * 0.5 / sigma_v
    # back
    rv = np.random.randn(n_samples, v_inds[0], 2) * sigma_v
    v_s[:, :v_inds[0], :] = rv.cumsum(axis=1)[:, ::-1, :] + v[v_inds[0]]
    # forward
    rv = np.random.randn(n_samples, k - v_inds[-1]-1, 2) * sigma_v
    v_s[:, v_inds[-1]+1:, :] = rv.cumsum(axis=1) + v[v_inds[-1]]
    # interpolate
    for i in range(1, v_inds.size):
        k_i = v_inds[i] - v_inds[i-1]
        from_l = np.arange(k_i-1) + 1
        from_r = from_l[::-1]
        mean = (v[[v_inds[i-1]]] * from_r[:, np.newaxis] +
                v[[v_inds[i]]] * from_l[:, np.newaxis]) / k_i
        sig = np.sqrt(from_r * from_l / k_i) * sigma_v
        rv = mean + np.random.randn(n_samples, k_i - 1, 2) *\
             sig[np.newaxis, :, np.newaxis]
        v_s[:, v_inds[i-1]+1:v_inds[i], :] = rv
    return v_s, nll


def sample_single_v1(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    - calculate nll for v_s set consecutively from v since
      they are related by sigma_v
    :param v: k, 2 | time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return:
        v_s: n_samples, k, 2
        nll: n_samples,
    """
    k = v.shape[0]
    v_s = np.empty((n_samples, k, 2))
    v_mask = ~np.isnan(v).any(axis=1)
    v_inds = np.arange(k)[v_mask]
    v_s[:, v_mask, :] = v[v_mask]
    v_s[:, v_mask, :] += np.random.randn(n_samples, v_inds.size, 2) * sigma
    c_inds = v_inds[:-1][(v_inds[1:] - v_inds[:-1]) == 1]
    nll = ((v_s[:, c_inds, :] -
            v_s[:, c_inds+1, :])**2).sum(axis=(1, 2)) * 0.5 / sigma_v
    # back
    rv = np.random.randn(n_samples, v_inds[0], 2) * sigma_v
    v_s[:, :v_inds[0], :] = rv.cumsum(axis=1)[:, ::-1, :] + v[v_inds[0]]
    # forward
    rv = np.random.randn(n_samples, k - v_inds[-1] - 1, 2) * sigma_v
    v_s[:, v_inds[-1] + 1:, :] = rv.cumsum(axis=1) + v[v_inds[-1]]
    # interpolate
    from_l_inds = v_mask.cumsum() - 1
    m = from_l_inds > -1
    from_l_inds[m] = v_inds[from_l_inds[m]]
    # :v_inds[0] values are undefined
    from_l = 1 - v_mask
    from_l[v_inds] = -np.hstack((v_inds[0], np.diff(v_inds)-1))
    from_l = from_l.cumsum()

    from_r_inds = v_mask[::-1].cumsum()[::-1] - 1
    m = from_r_inds > -1
    from_r_inds[m] = v_inds[v_inds.size - from_r_inds[m]-1]
    # v_inds[-1]: values are undefined
    from_r = 1 - v_mask
    from_r[v_inds] = 1-np.hstack((np.diff(v_inds), k - v_inds[-1]))
    from_r = from_r[::-1].cumsum()[::-1]

    im = (v_inds[0] < np.arange(k)) & (np.arange(k) < v_inds[-1]) &\
         (from_l > 0) & (from_r > 0)
    k_interp = im.sum()
    from_l_inds = from_l_inds[im]
    from_l = from_l[im]
    from_r_inds = from_r_inds[im]
    from_r = from_r[im]
    mean = (v[from_l_inds].reshape(-1, 2) * from_r[:, np.newaxis] +
            v[from_r_inds].reshape(-1, 2) * from_l[:, np.newaxis])
    gaps = from_r_inds - from_l_inds
    mean /= gaps[:, np.newaxis]
    sig = np.sqrt(from_r * from_l / gaps) * sigma_v
    rv = mean + np.random.randn(n_samples, k_interp, 2) * \
        sig[np.newaxis, :, np.newaxis]
    v_s[:, im, :] = rv
    return v_s, nll


def sample_n_v0(v, n_samples, sigma, sigma_v):
    """
    Sample random walk values from partial observations
    :param v: n, k, 2 | n independent time series of velocities
    - nan in timesteps to sample by bridge/walking
      valid values in timesteps to use
    - exist at least 1 valid value in each series
    :param n_samples:
    :param sigma: sd of valid v, eg. sqrt(2)sigma_x/dt for differences
    :param sigma_v:
    :return:
        v_s: n, n_samples, k, 2
        nll: n, n_samples
    """
    n, k = v.shape[:2]
    v_s = np.empty((n, n_samples, k, 2))
    nll = np.empty((n, n_samples))
    for i in range(n):
        v_s[i, ...], nll[i, :] = sample_single_v1(
            v[i, ...], n_samples, sigma, sigma_v)
    return v_s, nll


def main_sample_single_exact():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    k = 30
    v = np.random.randn(k, 2)
    is_nan = np.random.randn(k) > 0
    if np.all(is_nan):
        is_nan[1] = False
    v[is_nan] = np.nan
    # print(v)
    n_samples = 100
    sigma = 0.1
    sigma_v = 0.05
    args = (v, n_samples, sigma, sigma_v)
    print('---------------')

    np.random.seed(seed)
    x_true = sample_single_exact_v0(*args)
    np.random.seed(seed)
    x_hat = sample_single_exact_v2(*args)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))
    # print(x_true)
    # print(x_hat)

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_single_exact_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_single_exact_v2, args=args))/n_tries)


def main_sample_single():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    k = 30
    v = np.random.randn(k, 2)
    is_nan = np.random.randn(k) > 0
    if np.all(is_nan):
        is_nan[1] = False
    v[is_nan] = np.nan
    # print(v)
    n_samples = 100
    sigma = 0.
    sigma_v = 0.1
    args = (v, n_samples, sigma, sigma_v)
    print('---------------')

    np.random.seed(seed)
    x_true = sample_single_v0(*args)
    np.random.seed(seed)
    x_hat = sample_single_v1(*args)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))
    # print(x_true)
    # print(x_hat)

    n_tries = 2
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_single_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_single_v1, args=args))/n_tries)


if __name__ == '__main__':
    # main_sample_single()
    main_sample_single_exact()

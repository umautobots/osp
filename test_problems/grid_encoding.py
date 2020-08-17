import numpy as np

EPS = 1e-4


def clip2grid(rt, grid):
    # modifies rt
    np.clip(rt[..., 0], grid[0] + EPS, grid[0] + grid[2] * (grid[-2] - 1) - EPS, out=rt[..., 0])
    np.clip(rt[..., 1], grid[1] + EPS, grid[1] + grid[3] * (grid[-1] - 1) - EPS, out=rt[..., 1])


def rt2enc_v0(rt, grid):
    """

    :param rt: n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n, k, m | m = nx*ny, encoding for each ped to each vic
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n, k = rt.shape[:2]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n, k, m), dtype=np.float)
    clip2grid(rt, grid)
    for i in range(n):
        for j in range(k):
            a_x, r_x = np.divmod(rt[i, j, 0] - grid[0], grid[2])
            th_x = 1 - r_x / grid[2]
            a_y, r_y = np.divmod(rt[i, j, 1] - grid[1], grid[3])
            th_y = 1 - r_y / grid[3]
            inds = np.ravel_multi_index(np.array(
                [[a_x, a_x + 1, a_x, a_x + 1], [a_y, a_y, a_y + 1, a_y + 1]], dtype=np.int), (nx, ny))
            Z[i, j, inds] += np.array([th_x * th_y, (1 - th_x) * th_y, (1 - th_y) * th_x, (1 - th_x) * (1 - th_y)])
    return Z


def rt2enc_v1(rt, grid):
    """

    :param rt: n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n, k, m | m = nx*ny, encoding for each ped to each vic
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n, k = rt.shape[:2]
    nx, ny = np.array(grid[-2:]).astype(np.int)
    m = nx * ny
    Z = np.zeros((n, k, m), dtype=np.float32)
    clip2grid(rt, grid)

    # n, k
    a_x = np.empty((n, k))
    r_x = np.empty((n, k), dtype=np.float32)
    np.divmod(rt[..., 0] - grid[0], grid[2], a_x, r_x)
    th_x = 1 - r_x / grid[2]
    a_y = np.empty((n, k))
    r_y = np.empty((n, k), dtype=np.float32)
    np.divmod(rt[..., 1] - grid[1], grid[3], a_y, r_y)
    th_y = 1 - r_y / grid[3]

    # 1d inds for m, | n, k
    m_inds = (ny * a_x + a_y).astype(np.int)
    assert np.all(m_inds >= 0), 'ny {} ax {} ay {}'.format(ny, a_x.min(), a_y.min())
    offsets = np.array([0, ny, 1, ny+1], dtype=np.int)

    nk_flat_inds = np.arange(0, n*k*m, m, dtype=np.int)
    nk_flat_inds += m_inds.reshape(-1)
    Z.flat[nk_flat_inds + offsets[0]] = (th_x * th_y).reshape(-1)
    Z.flat[nk_flat_inds + offsets[1]] = ((1-th_x) * th_y).reshape(-1)
    Z.flat[nk_flat_inds + offsets[2]] = (th_x * (1-th_y)).reshape(-1)
    Z.flat[nk_flat_inds + offsets[3]] = ((1-th_x) * (1-th_y)).reshape(-1)
    return Z


def rt2enc_particles_v0(rt, grid):
    """

    :param rt: n_p, n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
        for n_p particles
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n_p, n, k, m | m = nx*ny, encoding for each ped to each vic
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n_p, n, k = rt.shape[:3]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n_p, n, k, m), dtype=np.float)
    clip2grid(rt, grid)
    for i in range(n_p):
        Z[i, ...] = rt2enc_v1(rt[i, ...], grid)
    return Z


def rt2enc_particles_v1(rt, grid):
    """

    :param rt: n_p, n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
        for n_p particles
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n_p, n, k, m | m = nx*ny, encoding for each ped to each vic
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n_p, n, k = rt.shape[:3]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n_p, n, k, m), dtype=np.float)
    clip2grid(rt, grid)

    # n_p, n, k
    a_x = np.empty((n_p, n, k), dtype=np.int32)
    r_x = np.empty((n_p, n, k), dtype=np.float32)
    np.divmod(rt[..., 0] - grid[0], grid[2], a_x, r_x, casting='unsafe')
    th_x = 1 - r_x / grid[2]
    a_y = np.empty((n_p, n, k), dtype=np.int32)
    r_y = np.empty((n_p, n, k), dtype=np.float32)
    np.divmod(rt[..., 1] - grid[1], grid[3], a_y, r_y, casting='unsafe')
    th_y = 1 - r_y / grid[3]

    # 1d inds for m, | n_p, n, k
    m_inds = ny * a_x + a_y
    offsets = np.array([0, ny, 1, ny + 1], dtype=np.int32)
    nk_flat_inds = np.arange(0, n_p*n*k*m, m, dtype=np.int32)
    nk_flat_inds = nk_flat_inds + m_inds.ravel()
    Z.flat[nk_flat_inds + offsets[0]] = (th_x * th_y).ravel()
    Z.flat[nk_flat_inds + offsets[1]] = ((1-th_x) * th_y).ravel()
    Z.flat[nk_flat_inds + offsets[2]] = (th_x * (1-th_y)).ravel()
    Z.flat[nk_flat_inds + offsets[3]] = ((1-th_x) * (1-th_y)).ravel()
    return Z


def rt2add_enc_v0(rt, grid):
    """
    
    :param rt: n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n, m | m = nx*ny, encoding for each ped
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n, k = rt.shape[:2]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n, m), dtype=np.float)
    clip2grid(rt, grid)
    for i in range(n):
        for j in range(k):
            a_x, r_x = np.divmod(rt[i, j, 0] - grid[0], grid[2])
            # th_x = 1 + a_x - (rt[i, j, 0] - grid[0])/grid[2]  # 0)
            # th_x = 1 - np.remainder(rt[i, j, 0] - grid[0], grid[2])/grid[2]
            th_x = 1 - r_x/grid[2]  # 1)
            a_y, r_y = np.divmod(rt[i, j, 1] - grid[1], grid[3])
            # th_y = 1 + a_y - (rt[i, j, 1] - grid[1]) / grid[3]  # 0)
            # th_y = 1 - np.remainder(rt[i, j, 1] - grid[1], grid[3])/grid[3]
            th_y = 1 - r_y/grid[3]  # 1)
            inds = np.ravel_multi_index(np.array(
                [[a_x, a_x+1, a_x, a_x+1], [a_y, a_y, a_y+1, a_y+1]], dtype=np.int), (nx, ny))
            Z[i, inds] += np.array([th_x*th_y, (1-th_x)*th_y, (1-th_y)*th_x, (1-th_x)*(1-th_y)])
    return Z


def rt2add_enc_v1(rt, grid):
    """

    :param rt: n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n, m | m = nx*ny, encoding for each ped
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n, k = rt.shape[:2]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n, m), dtype=np.float32)
    clip2grid(rt, grid)

    # n, k
    a_x = np.empty((n, k), dtype=np.int32)
    r_x = np.empty((n, k), dtype=np.float32)
    np.divmod(rt[..., 0] - grid[0], grid[2], a_x, r_x, casting='unsafe')
    th_x = 1 - r_x / grid[2]
    a_y = np.empty((n, k), dtype=np.int32)
    r_y = np.empty((n, k), dtype=np.float32)
    np.divmod(rt[..., 1] - grid[1], grid[3], a_y, r_y, casting='unsafe')
    th_y = 1 - r_y / grid[3]

    # 1d inds for m, | n, k
    c_x = ny * a_x + a_y
    offsets = np.array([0, ny, 1, ny+1], dtype=np.int32)

    # n, k, 4
    inds = c_x[..., np.newaxis] + offsets[np.newaxis, :]
    vals = np.dstack((th_x*th_y, (1-th_x)*th_y, th_x*(1-th_y), (1-th_x)*(1-th_y)))

    row_inds = np.repeat(np.arange(n, dtype=np.int32), 4*k)
    np.add.at(Z, (row_inds, inds.ravel()), vals.ravel())
    return Z


def rt2add_enc_particles_v0(rt, grid):
    """

    :param rt: n_p, n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
        for n_p particles
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n_p, n, m | m = nx*ny, encoding for each ped
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n_p, n, k = rt.shape[:3]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n_p, n, m), dtype=np.float)
    clip2grid(rt, grid)
    for i in range(n_p):
        Z[i, ...] = rt2add_enc_v1(rt[i, ...], grid)
    return Z


def rt2add_enc_particles_v1(rt, grid):
    """

    :param rt: n_p, n, k, 2 | log[d, tau] for each ped (n,) to each vic (k,)
        modifies rt during clipping to grid
        for n_p particles
    :param grid: (lx, ly, dx, dy, nx, ny)
        lx, ly | lower bounds for x and y coordinates of the n*k (2,) in rt
        dx, dy | step sizes of the regular grid
        nx, ny | number of grid points in each coordinate (so nx*ny total)
    :return: n_p, n, m | m = nx*ny, encoding for each ped
        uses row-major indexing for the flattened (2d) indices
        for nx 'rows' and ny 'columns'
    """
    n_p, n, k = rt.shape[:3]
    nx, ny = np.array(grid[-2:]).astype(np.int32)
    m = nx * ny
    Z = np.zeros((n_p, n, m), dtype=np.float)
    clip2grid(rt, grid)

    # n_p, n, k
    a_x = np.empty((n_p, n, k), dtype=np.int32)
    r_x = np.empty((n_p, n, k), dtype=np.float32)
    np.divmod(rt[..., 0] - grid[0], grid[2], a_x, r_x, casting='unsafe')
    th_x = 1 - r_x / grid[2]
    a_y = np.empty((n_p, n, k), dtype=np.int32)
    r_y = np.empty((n_p, n, k), dtype=np.float32)
    np.divmod(rt[..., 1] - grid[1], grid[3], a_y, r_y, casting='unsafe')
    th_y = 1 - r_y / grid[3]

    # 1d inds for m, | n_p, n, k
    c_x = ny * a_x + a_y
    offsets = np.array([0, ny, 1, ny + 1], dtype=np.int32)

    # n_p, n, k, 4
    inds = c_x[..., np.newaxis] + offsets[np.newaxis, :]
    vals = np.stack((th_x*th_y, (1-th_x)*th_y,
                     th_x*(1-th_y), (1-th_x)*(1-th_y)), axis=-1)

    particle_inds = np.repeat(np.arange(n_p, dtype=np.int32), n * k * 4)
    row_inds = np.repeat(np.arange(n, dtype=np.int32), 4 * k)
    row_inds = np.tile(row_inds, n_p)
    np.add.at(Z, (particle_inds, row_inds, inds.ravel()), vals.ravel())
    return Z


def main_enc_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 30
    k = 2
    rt = np.random.rand(n_p, n, k, 2) * 5
    grid = np.array([0, 0, 0.5, 0.5, 10, 10])  # [0, ..., 4.5)^2
    print('---------------')

    x_true = rt2enc_particles_v0(rt.copy(), grid)
    x_hat = rt2enc_particles_v1(rt.copy(), grid)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2enc_particles_v0, a=rt.copy(), b=grid)) / n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2enc_particles_v1, a=rt.copy(), b=grid)) / n_tries)


def main_enc():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 30
    k = 2
    rt = np.random.rand(n, k, 2) * 5
    grid = np.array([0, 0, 0.5, 0.5, 10, 10])  # [0, ..., 4.5)^2
    print('---------------')

    x_true = rt2enc_v0(rt.copy(), grid)
    x_hat = rt2enc_v1(rt.copy(), grid)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_v0, a=rt.copy(), b=grid)) / n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_v1, a=rt.copy(), b=grid)) / n_tries)


def main_enc_add_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 30
    k = 2
    rt = np.random.rand(n_p, n, k, 2) * 5
    grid = np.array([0, 0, 0.5, 0.5, 10, 10])  # [0, ..., 4.5)^2
    print('---------------')

    x_true = rt2add_enc_particles_v0(rt.copy(), grid)
    x_hat = rt2add_enc_particles_v1(rt.copy(), grid)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))

    n_tries = 2
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_particles_v0, a=rt.copy(), b=grid)) / n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_particles_v1, a=rt.copy(), b=grid)) / n_tries)


def main_enc_add():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 30
    k = 2
    rt = np.random.rand(n, k, 2) * 5
    grid = np.array([0, 0, 0.5, 0.5, 10, 10])  # [0, ..., 4.5)^2
    print('---------------')

    x_true = rt2add_enc_v0(rt.copy(), grid)
    x_hat = rt2add_enc_v1(rt.copy(), grid)
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))
    # print(rt)
    # print(x_true)

    n_tries = 2
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_v0, a=rt.copy(), b=grid)) / n_tries)
    print(timeit('f(a, b)', number=n_tries, globals=dict(f=rt2add_enc_v1, a=rt.copy(), b=grid)) / n_tries)


if __name__ == '__main__':
    # main_enc_add()
    # main_enc_add_particles()
    main_enc()
    # main_enc_particles()


import numpy as np
from scipy import special as sp


def sample_softmax_rq_v0(x, is_ignore=()):
    """
    Sample indices of k [= r] with probability:

        softmax(x) = exp(x)/sum(exp(x))

    for each of n agents, and corresponding q such that

        q ~ bernoulli( 1 / 1+exp{x[r]} ) = logistic(-x[r])

    :param x: n, k
    :param is_ignore: n, k
    :return:
        r: n,
        q: n,
    """
    k = x.shape[-1]
    xmk_shape = x.shape[:-1]
    if len(is_ignore) > 0:
        b_mask = (~is_ignore).astype(np.int)
        p = np.exp(x - sp.logsumexp(x, axis=1, keepdims=True, b=b_mask))
        p[is_ignore] = 0.
    else:
        p = sp.softmax(x, axis=1)
    cp = np.concatenate((0*p[..., [0]], p.cumsum(axis=-1)), axis=-1)
    cp[..., -1] = 1.
    t1 = np.random.rand(*xmk_shape)
    inds = np.broadcast_to(np.arange(k), x.shape)
    mask = ((cp[..., :-1].T <= t1) & (t1 < cp[..., 1:].T)).T
    r = inds[mask]
    p_logistic = np.exp(-np.logaddexp(np.zeros(xmk_shape), x[mask]))
    t2 = np.random.rand(*xmk_shape)
    q = t2 < p_logistic
    return r, q


def sample_softmax_rq_particles_v0(x):
    """
    Sample indices of k [= r] with probability:

        softmax(x) = exp(x)/sum(exp(x))

    for each of n agents, and corresponding q such that

        q ~ bernoulli( 1 / 1+exp{x[r]} ) = logistic(-x[r])

    :param x: n_p, n, k
    :return:
        r: n_p, n,
        q: n_p, n,
    """
    xmk_shape = x.shape[:-1]
    r = np.empty(xmk_shape, dtype=np.int)
    q = np.empty(xmk_shape, dtype=np.int)
    for i in range(xmk_shape[0]):
        r[i], q[i] = sample_softmax_rq_v0(x[i])
    return r, q


def sample_softmax_rq_particles_v1(x):
    """
    Sample indices of k [= r] with probability:

        softmax(x) = exp(x)/sum(exp(x))

    for each of n agents, and corresponding q such that

        q ~ bernoulli( 1 / 1+exp{x[r]} ) = logistic(-x[r])

    :param x: n_p, n, k
    :return:
        r: n_p, n,
        q: n_p, n,
    """
    k = x.shape[-1]
    xmk_shape = x.shape[:-1]
    p = sp.softmax(x, axis=-1)
    cp = np.concatenate((0*p[..., [0]], p.cumsum(axis=-1)), axis=-1)
    cp[..., -1] = 1.
    t1 = np.random.rand(*xmk_shape)[..., np.newaxis]
    inds = np.broadcast_to(np.arange(k), x.shape)
    mask = (cp[..., :-1] <= t1) & (t1 < cp[..., 1:])
    r = inds[mask].reshape(xmk_shape)
    p_logistic = np.exp(-np.logaddexp(np.zeros(xmk_shape), x[mask].reshape(xmk_shape)))
    t2 = np.random.rand(*xmk_shape)
    q = t2 < p_logistic
    return r, q


def main_sample_softmax_rq_particles():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    # seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n_p = 100
    n = 3
    k = 2
    x = np.random.randn(n_p, n, k)
    print('---------------')

    np.random.seed(seed)
    x_true = sample_softmax_rq_particles_v0(x)
    np.random.seed(seed)
    x_hat = sample_softmax_rq_particles_v1(x)
    # random - but should be equal most of the time for small sizes
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[0] - x_hat[0])))
    print('diff: {:0.4f}'.format(np.linalg.norm(x_true[1] - x_hat[1])))
    # print(x_true[0])
    # print(x_hat[0])

    n_tries = 2
    args = (x,)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_softmax_rq_particles_v0, args=args))/n_tries)
    print(timeit('f(*args)', number=n_tries, globals=dict(f=sample_softmax_rq_particles_v1, args=args))/n_tries)


def main_sample_softmax_rq():
    from timeit import timeit
    seed = np.random.randint(0, 1000)
    seed = 0
    np.random.seed(seed)
    print('seed: {}'.format(seed))

    n = 3
    k = 2
    x = np.random.randn(n, k)
    print('---------------')

    np.random.seed(seed)
    x_true = sample_softmax_rq_v0(x)
    # np.random.seed(seed)
    # x_hat = sample_softmax_rq_v1(x)
    # print('diff: {:0.4f}'.format(np.linalg.norm(x_true - x_hat)))


if __name__ == '__main__':
    # main_sample_softmax_rq()
    main_sample_softmax_rq_particles()

import torch
from torch.utils.data import DataLoader, TensorDataset


#region mini-batch K means
class MiniBatchKMeans:
    def __init__(self, k:int, batch_size:int, max_epoch:int=1000,
                 tol:float=1e-6, radius:float=1, eps:float=1e-3,
                 sparse:bool=False):
        assert k > 0 and batch_size > 0, f'Parameters should be positive, but got: {k} and {batch_size}\n'
        self.k = k
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.center = None
        self.tol = tol
        self.radius = radius
        self.eps = eps
        self.count = 0
        self.sparse = sparse

    def fit(self, X:torch.Tensor):
        # initialize parameters
        (n, p) = X.shape
        v = torch.zeros(self.k, dtype=torch.float64)

        # initialze k centers randomly
        center_init = torch.randint(0, n, (self.k,))
        center = X[center_init, :]
        center_prev = center.clone()

        epoch = 0
        while epoch < self.max_epoch:
            # shuffle every epoch
            dataloader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)

            # loop minibatch
            for i, M in enumerate(dataloader, 0):
                M = M[0]
                cache = torch.zeros(M.shape[0])

                # cache centers
                for j in range(M.shape[0]):
                    dist = torch.norm(M[j, :].repeat(self.k, 1) - center, dim=1)
                    (val, ind) = torch.min(dist, dim=0)
                    cache[j] = ind

                # per center update
                for j in range(M.shape[0]):
                    c = cache[j].long()
                    v[c] += 1
                    eta = 1/v[c]
                    center[c, :] = (1 - eta)*center[c, :] + eta*M[j, :]

                # sparse projection
                if self.sparse:
                    for k in range(self.k):
                        center[k, :] = self.sparse_projection(center[k, :])

                # check convergence
                if torch.sum(torch.square(center - center_prev)) < self.tol:
                    print(f'Convergence achieved. Iteration: {epoch}\n')
                    self.center = center
                    return

            # report iterations
            if epoch % 100 == 0:
                print(f'Iteration: {epoch}')

            # go next epoch
            center_prev = center.clone()
            epoch += 1

        print(f'Maximum epoch reached: {epoch}')
        self.center = center

    def update_center(self, X:torch.Tensor):
        (n, p) = X.shape
        labels = torch.zeros(n, dtype=torch.long)

        for i in range(n):
            x = X[i, :]
            dist = torch.norm(x.repeat(self.k, 1) - self.center, dim=1)
            (val, ind) = torch.min(dist, dim=0)
            labels[i] = ind

        return labels

    def sparse_projection(self, c:torch.Tensor):
        c_1norm = torch.sum(torch.abs(c))
        if c_1norm <= self.radius + self.eps:
            return c

        upper = torch.max(torch.abs(c))
        lower = 0
        current = c_1norm
        theta = 0

        while current > self.radius*(1+self.eps) or current < self.radius:
            theta = 0.5*(upper+lower)
            current = torch.sum(torch.clamp(torch.abs(c) - theta, 0))

            if current <= self.radius:
                upper = theta
            else:
                lower = theta

        c = torch.sign(c)*torch.clamp(torch.abs(c)-theta, 0)
        return c
#endregion


#region K means with Lloyd's algorithm
class LloydKMeans:
    def __init__(self, k:int, max_iter:int=1000, tol:float=1e-6):
        assert k > 1, f'There must be at least two clusters, but got: {k}\n'
        assert max_iter > 0, f'max_iter must be positive, but got :{max_iter}\n'
        assert tol > 0, f'tol must be positive, but got: {tol}\n'
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.center = None
        self.label = None
        self.inertia = None

    def fit(self, X:torch.Tensor):
        # initialize parameters
        (n, p) = X.shape
        cache = torch.zeros(n, dtype=torch.long)

        # initialze k centers randomly
        center_init = torch.randint(0, n, (self.k,))
        center = X[center_init, :]
        inertia_prev = torch.zeros(1)
        cache_prev = cache.clone()

        # algorithm
        iter = 0
        while iter < self.max_iter:
            self.inertia = torch.zeros(1)

            # assign to clusters
            for i in range(n):
                dist = torch.norm(X[i, :].repeat(self.k, 1) - center, dim=1)
                (val, ind) = torch.min(dist, dim=0)
                cache[i] = ind

            # update clusters and inertia
            for j in range(self.k):
                mask = cache == j
                center[j, :] = torch.mean(X[mask, :], dim=0)
                self.inertia += torch.sum(torch.square(X[mask, :] - center[j, :]))

            # convergence
            if torch.max(torch.abs(inertia_prev - self.inertia)) < self.tol:
                print(f'Convergence achieved: {iter}\n')
                self.center = center
                self.label = cache
                return

            inertia_prev = self.inertia
            iter += 1

        print(f'Maximum iteration reached: {iter}')
        self.center = center
        self.label = cache
#endregion


#region K means with MacQueen's algorithm
class MacQueenKMeans:
    def __init__(self, k:int, max_iter:int=1000, tol:float=1e-4, shuffle:bool=True):
        assert k > 1, f'There must be at least two clusters, but got: {k}\n'
        assert max_iter > 0, f'max_iter must be positive, but got :{max_iter}\n'
        assert tol > 0, f'tol must be positive, but got: {tol}\n'
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.center = None
        self.label = None
        self.inertia = None

    def fit(self, X:torch.Tensor):
        # initialize parameters
        (n, p) = X.shape
        cache = None
        self.label = torch.zeros(n, dtype=torch.long)

        # initialze k centers randomly
        center_init = torch.randint(0, n, (self.k,))
        center = X[center_init, :]
        center_prev = center.clone()

        # shuffle
        idx_shuffle = torch.arange(n)
        if self.shuffle:
            idx_shuffle = torch.randperm(n)
            X = X[idx_shuffle, :]

        # algorithm
        iter = 0
        while iter < self.max_iter:
            # reinitialize parameters
            n_center = torch.zeros(self.k)
            cache = torch.zeros(n, dtype=torch.long)

            for i in range(n):
                # assign to clusters
                dist = torch.sum(torch.square(X[i, :].repeat(self.k, 1) - center), dim=1)
                (val, ind) = torch.min(dist, dim=0)
                cache[i] = ind

                # update clusters
                center[ind, :] = (n_center[ind]*center[ind, :] + X[i, :])/ (n_center[ind] + 1)
                n_center[ind] += 1

            # check convergence
            if torch.max(torch.abs(center - center_prev)) < self.tol:
                print(f'Convergence achieved: {iter}\n')
                self.center = center
                self.label[idx_shuffle] = cache
                return

            center_prev = center.clone()
            iter += 1

        print(f'Maximum iteration reached: {iter}')
        self.center = center
        self.label[idx_shuffle] = cache
#endregion


#region K means with Hartigan-Wong algorithm
class HartiganWongKMeans:
    def __init__(self, k:int, max_iter:int=1000, tol:float=1e-4, M:int=5):
        assert k > 1, f'There must be at least two clusters, but got: {k}\n'
        assert max_iter > 0, f'max_iter must be positive, but got :{max_iter}\n'
        assert tol > 0, f'tol must be positive, but got: {tol}\n'
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.M = M
        self.center = None
        self.label = None
        self.inertia = None

    def fit(self, X:torch.Tensor):
        # initialize parameters
        (n, p) = X.shape
        self.label = torch.zeros(n, dtype=torch.long)

        ic1 = torch.zeros(n, dtype=torch.long) # ic1[i] the closest cluster center for i-th obs
        ic2 = torch.zeros(n, dtype=torch.long) # ic2[i] the second closest cluster center for i-th obs

        # initialize k centers randomly
        center_init = torch.randint(0, n, (self.k,))
        center = X[center_init, :]

        # update IC(1) and IC(2)
        for i in range(n):
            dist = torch.sum(torch.square(X[i, :].repeat(self.k, 1) - center), dim=1)
            (val, ind) = torch.topk(dist, 2, dim=0, largest=False)
            ic1[i], ic2[i] = ind[0], ind[1]

        # update center
        for j in range(self.k):
            mask = ic1 == j
            center[j, :] = torch.mean(X[mask, :], dim=0)

        # define live set
        live_set = torch.zeros(self.k) == 0

        # precompute R1
        R1 = torch.zeros(n)
        for i in range(n):
            nc = torch.sum(ic1 == ic1[i])
            R1[i] = torch.sum(torch.square(X[i, :] - center[ic1[i], :]))
            R1[i] = nc*R1[i]/(nc - 1)

        iter = 0
        while iter < self.max_iter:
            print(f'iter: {iter}')
            print(f'Stage: OPTRA')
            # OPTRA stage
            for i in range(n):
                min_R2 = torch.inf
                min_idx = None

                # if the i-th obs is in cluster which belonging to the live set
                if live_set[ic1[i]]:
                    # loop through L2
                    for j in range(self.k):
                        if j != ic1[i]:
                            nc2 = torch.sum(ic1 == j)
                            R2 = torch.sum(torch.square(X[i, :] - center[j, :]))
                            R2 = nc2*R2/(nc2+1)

                            if R2 < min_R2:
                                min_R2 = R2
                                min_idx = j
                else:
                    # loop through L2
                    for j in range(self.k):
                        if j != ic1[i] and live_set[j] is False:
                            nc2 = torch.sum(ic1 == j)
                            R2 = torch.sum(torch.square(X[i, :] - center[j, :]))
                            R2 = nc2*R2/(nc2+1)

                            if R2 < min_R2:
                                min_R2 = R2
                                min_idx = j

                # compare loss
                if min_R2 >= R1[i]:
                    # no reallocation
                    ic2[i] = min_idx

                else:
                    # update R1
                    R1[i] = min_R2
                    # swap L1 and L2 sets
                    ic2[i] = ic1[i]
                    ic1[i] = min_idx
                    # update live set
                    live_set[ic1[i]] = True
                    live_set[ic2[i]] = True
                    # update centers
                    center[ic2[i], :] = torch.mean(X[ic1 == ic2[i], :], dim=0)
                    center[ic1[i], :] = torch.mean(X[ic1 == ic1[i], :], dim=0)

            # terminate
            if torch.all(live_set) is False:
                self.center = center
                self.label = ic1
                print('Convergence achieved.')
                return

            # define live set
            live_set = torch.zeros(self.k) == 0

            print(f'Stage: QTRAN')
            # QTRAN stage
            not_update_cluster = torch.zeros(self.k, dtype=torch.long) == 0
            not_update_cluster_prev = not_update_cluster.clone()
            count_transfer = 0

            bool_cont = True
            while bool_cont:
                for i in range(n):
                    # compute R2
                    nc2 = torch.sum(ic1 == ic2[i])
                    R2 = torch.sum(torch.square(X[i, :] - center[ic2[i], :]))
                    R2 = nc2 * R2 / (nc2 + 1)

                    if R1[i] >= R2:
                        # reallocation
                        ic2[i], ic1[i] = ic1[i], ic2[i]
                        R1[i] = R2
                        # update centers
                        center[ic2[i], :] = torch.mean(X[ic1 == ic2[i], :], dim=0)
                        center[ic1[i], :] = torch.mean(X[ic1 == ic1[i], :], dim=0)
                        # update transfer
                        live_set[ic1[i]] = True
                        live_set[ic2[i]] = True
                        # reset not update
                        not_update_cluster[ic1[i]] = True
                        not_update_cluster[ic2[i]] = True

                # count transfer
                if torch.all(not_update_cluster == not_update_cluster_prev):
                    count_transfer += 1
                else:
                    count_transfer = 0
                not_update_cluster_prev = not_update_cluster.clone()
                bool_cont = False
                # no transfer in last M steps
                if count_transfer == self.M:
                    bool_cont = False

import torch
from torch.utils.data import DataLoader, TensorDataset

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














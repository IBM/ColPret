import numpy as np
import torch
from tqdm import tqdm


def chinchilla_loss(inp, params):
    a, b, e, alpha, beta = params[0], params[1], params[2], params[3], params[4]
    pre_lse = torch.stack([a - alpha * torch.log(inp[:, 0]), b - beta * torch.log(inp[:, 1]), e.expand((inp.shape[0]))])
    post_lse = torch.logsumexp(pre_lse, dim=0)
    huber_loss = torch.nn.functional.huber_loss(post_lse, torch.log(inp[:, 2]), delta=1e-3, reduction='none')
    return huber_loss.sum()


def minimize_loss(inp, init_params, steps=50, loss_func=chinchilla_loss):
    params = torch.nn.Parameter(data=torch.Tensor(init_params))

    lbfgs = torch.optim.LBFGS([params],
                              lr=1e-1,
                              history_size=10,
                              max_iter=20,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = loss_func(inp, params)
        l.backward()
        return l

    history_lbfgs = []
    for i in range(steps):
        l = lbfgs.step(closure)
    return l, params


def chinchilla_torch_fit(func, metadata, perf, p0=None, bounds=None, method=None):
    inp = torch.Tensor([[F, D, L] for F, D, L in
                        zip(metadata["flops"], metadata["tokens_seen"], perf)])
    inp.require_grad = True

    min_loss = 1e10
    grid_dim = 4
    if p0 is not None:
        l, params = minimize_loss(inp, p0, loss_func=chinchilla_loss)
        min_loss = l
        best_params = params.detach().numpy()
    else:
        for a in tqdm(np.linspace(12, 0, grid_dim, endpoint=False)[::-1]):
            for b in np.linspace(12, 0, grid_dim, endpoint=False)[::-1]:
                for e in np.linspace(1, -1, grid_dim, endpoint=False)[::-1]:
                    for alpha in np.linspace(1, 0, grid_dim, endpoint=False)[::-1]:
                        for beta in np.linspace(1, 0, grid_dim, endpoint=False)[::-1]:
                            p0 = [a, b, e, alpha, beta]
                            l, params = minimize_loss(inp, p0, loss_func=chinchilla_loss)
                            if l < min_loss:
                                min_loss = l
                                best_params = params.detach().numpy()
    min_loss = min_loss.detach().numpy()
    return [float(x) for x in best_params], float(min_loss)


if __name__ == '__main__':

    from util.fit_utils import get_perf_df, get_perf_path, LossType, get_data_path, get_per_model_metadata
    from util.read_data import get_data

    # acquire data
    force = True
    force = False
    cache_dir = '/Users/lc/PycharmProjects/CLPR/cache/'
    data_path = get_data_path(cache_dir)

    loss_types = (LossType.PERP, LossType.LOSS)
    perf_path = get_perf_path(cache_dir, loss_types)
    all_df = get_data(save_in=data_path, force=force)
    df = get_perf_df(all_df, loss_types, save_in=perf_path, force=force)
    metadata = get_per_model_metadata(df)

    opt_df = df[df["scaled_set"] == "OPT"]
    inp = torch.Tensor([[F, D, L] for F, D, L in
                        zip(opt_df["flops"], opt_df["tokens_seen"], opt_df["perf"])])
    inp.require_grad = True

    losses = []
    all_pred_params = []
    min_loss = 1e10
    for a in tqdm(np.linspace(0, 12, 4)):
        for b in np.linspace(0, 12, 4):
            for e in np.linspace(-1, 1, 4):
                for alpha in np.linspace(0, 1, 4):
                    for beta in np.linspace(0, 1, 4):
                        l, params = minimize_loss(inp, [a, b, e, alpha, beta])
                        losses.append(l.detach().numpy())
                        all_pred_params.append(params.detach().numpy())
                        if l < min_loss:
                            min_loss = l
                            best_params = params.detach().numpy()

    print(np.array(losses))
    print(min_loss)
    print(best_params)
    guess = np.array([6.255414, 7.3049974, 0.6254804, 0.3526596, 0.3526596])
    print("Guess, min loss:", minimize_loss(inp, guess)[0], minimize_loss(inp, guess + 0.1)[0],
          minimize_loss(inp, guess - 0.1)[0])

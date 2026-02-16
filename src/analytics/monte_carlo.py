import numpy as np

def monte_carlo_paths(series, n_days, n_sims):
log_returns = np.log(series / series.shift(1)).dropna()
mu = log_returns.mean()
sigma = log_returns.std()

paths = np.zeros((n_days, n_sims))
paths[0] = series.iloc[-1]

for t in range(1, n_days):
rand = np.random.normal(mu, sigma, n_sims)
paths[t] = paths[t - 1] * np.exp(rand)

return paths

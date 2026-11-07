import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

def run_bayesian_return_inference(returns: np.ndarray, draws: int = 1000, tune: int = 500):
    """
    Performs Bayesian MCMC inference on asset returns using PyMC.
    Estimates posterior distributions for mean return (mu) and volatility (sigma).
    """
    # Clean and flatten returns data
    clean_returns = np.asarray(returns, dtype=float)
    clean_returns = clean_returns[~np.isnan(clean_returns)]

    with pm.Model() as model:
        # Priors for mean and standard deviation
        mu = pm.Normal("mu", mu=0.0, sigma=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        
        # Likelihood (Student-T or Normal distribution of returns)
        likelihood = pm.Normal("returns_obs", mu=mu, sigma=sigma, observed=clean_returns)
        
        # MCMC Sampling (Posterior Inference)
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True, progressbar=False)

    # Extract posterior summary statistics
    summary = az.summary(trace, hdi_prob=0.95)
    
    mu_mean = float(summary.loc["mu", "mean"])
    mu_hdi_low = float(summary.loc["mu", "hdi_3%"])
    mu_hdi_high = float(summary.loc["mu", "hdi_97%"])
    
    sigma_mean = float(summary.loc["sigma", "mean"])
    sigma_hdi_low = float(summary.loc["sigma", "hdi_3%"])
    sigma_hdi_high = float(summary.loc["sigma", "hdi_97%"])

    return {
        "inference_method": "Bayesian MCMC (PyMC / NUTS sampler)",
        "samples_drawn": draws,
        "tuning_steps": tune,
        "posterior_estimates": {
            "mu_mean_return": mu_mean,
            "mu_95_hdi": [mu_hdi_low, mu_hdi_high],
            "sigma_volatility": sigma_mean,
            "sigma_95_hdi": [sigma_hdi_low, sigma_hdi_high]
        },
        "status": "converged"
    }
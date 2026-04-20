import pandas as pd
import numpy as np
import jax.numpy as jnp

def fetch_and_clean_data(cutoff_date="2/15/21"):
    url_c = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
    url_d = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"

    # 391 days between jan 2020 to feb 2021
    df_c = pd.read_csv(url_c)
    df_d = pd.read_csv(url_d)
    
    cols_to_drop = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 
                    'Country_Region', 'Lat', 'Long_', 'Combined_Key']
    
    C_obs = df_c.drop(columns=cols_to_drop).sum(axis=0).loc[:cutoff_date].values
    D_obs = df_d.drop(columns=cols_to_drop + ['Population']).sum(axis=0).loc[:cutoff_date].values
    
    return C_obs, D_obs

def derive_latent_states(C_obs, D_obs, N=331449281):
    # Parameters based on research paper
    lamb, gamma, sigma, alpha = 0.2, 0.1, 0.2, 0.0065
    
    dC_dt = np.gradient(C_obs)
    I_t = dC_dt / (lamb * gamma)
    dI_dt = np.gradient(I_t)
    
    E_t = (dI_dt + gamma * I_t) / sigma
    dE_dt = np.gradient(E_t)
    
    dR_dt = (1 - alpha) * gamma * I_t
    R_t = np.cumsum(dR_dt)
    
    # Solve for S(t)
    dS_dt = -(dE_dt + sigma * E_t)
    S_t = np.zeros_like(dS_dt)
    S_t[0] = N - E_t[0] - I_t[0] - R_t[0] - D_obs[0]
    
    for i in range(1, len(S_t)):
        S_t[i] = S_t[i-1] + dS_dt[i]
        
    obs_matrix = jnp.stack([S_t, E_t, I_t, R_t, D_obs, C_obs])
    t_data = jnp.linspace(0, len(C_obs)-1, len(C_obs)).reshape(-1, 1)
    true_data = obs_matrix.T / N
    
    return t_data, true_data, N
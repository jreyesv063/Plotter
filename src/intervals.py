from scipy.stats import poisson


def poisson_interval_v2(values, conf_level=0.95):
    # Calcular los percentiles inferior y superior del intervalo de confianza para cada valor en la lista
    alpha = 1 - conf_level
    down = [poisson.ppf(alpha / 2, k) for k in values]
    up = [poisson.ppf(1 - alpha / 2, k) for k in values]
    return down, up

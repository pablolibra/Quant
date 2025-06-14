import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from scipy.optimize import minimize
import os

plt.style.use('seaborn-v0_8-darkgrid')

API_KEY = os.getenv('API_KEY_FMP')
BASE_URL = 'https://financialmodelingprep.com/api/v3'
rf = 0.03

def safe_float(value):
    return float(value) if value not in [None, ''] else 0

def max_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

def get_ratios(ticker):
    url = f'{BASE_URL}/ratios-ttm/{ticker}?apikey={API_KEY}'
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return r.json()[0]
    return None

def get_zscore(ticker):
    url = f'{BASE_URL}/ratios/{ticker}?limit=1&apikey={API_KEY}'
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return r.json()[0].get("altmanZScore", None)
    return None

def get_peers(ticker):
    url = f'https://financialmodelingprep.com/api/v4/stock_peers?symbol={ticker}&apikey={API_KEY}'
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return r.json()[0].get('peersList', [])
    return []

def get_metric_from_tickers(tickers, metric_name):
    values = []
    for t in tickers:
        r = get_ratios(t)
        if r:
            val = safe_float(r.get(metric_name))
            if val > 0:
                values.append(val)
    return values

def calcular_percentil(value, peers_values):
    if not peers_values:
        return None
    all_values = np.array(peers_values + [value])
    return round((np.sum(all_values <= value) / len(all_values)) * 100, 2)

def analizar_fundamentales(ticker):
    data = get_ratios(ticker)
    if not data:
        return None

    try:
        roic = safe_float(data.get("returnOnCapitalEmployedTTM"))
        ebit_margin = safe_float(data.get("operatingProfitMarginTTM"))
        zscore = safe_float(get_zscore(ticker))
        quick_ratio = safe_float(data.get("quickRatioTTM"))
        interest_coverage = safe_float(data.get("interestCoverageTTM"))
        debt_to_equity = safe_float(data.get("debtEquityRatioTTM"))

        peers = get_peers(ticker)

        peers_roic = get_metric_from_tickers(peers, "returnOnCapitalEmployedTTM")
        peers_ebit = get_metric_from_tickers(peers, "operatingProfitMarginTTM")
        peers_quick = get_metric_from_tickers(peers, "quickRatioTTM")
        peers_coverage = get_metric_from_tickers(peers, "interestCoverageTTM")
        peers_debt = get_metric_from_tickers(peers, "debtEquityRatioTTM")

        roic_percentil = calcular_percentil(roic, peers_roic)
        ebit_percentil = calcular_percentil(ebit_margin, peers_ebit)
        quick_percentil = calcular_percentil(quick_ratio, peers_quick)
        coverage_percentil = calcular_percentil(interest_coverage, peers_coverage)
        debt_percentil = calcular_percentil(debt_to_equity, peers_debt)

        roic_ok = roic_percentil >= 75
        ebit_ok = ebit_percentil >= 75
        quick_ok = quick_percentil >= 75
        coverage_ok = coverage_percentil >= 75
        debt_ok = debt_percentil <= 25

    except:
        return None

    criterios = {
        'ROIC en top 25% del sector': roic_ok,
        'EBIT Margin en top 25% del sector': ebit_ok,
        'Altman Z > 3': zscore > 3,
        'Quick Ratio en top 25% del sector': quick_ok,
        'Interest Coverage en top 25% del sector': coverage_ok,
        'Debt to Equity en bottom 25% del sector': debt_ok
    }

    comparaciones = {
        'ROIC': {'value': roic, 'percentile': roic_percentil},
        'EBIT Margin': {'value': ebit_margin, 'percentile': ebit_percentil},
        'Quick Ratio': {'value': quick_ratio, 'percentile': quick_percentil},
        'Interest Coverage': {'value': interest_coverage, 'percentile': coverage_percentil},
        'Debt/Equity': {'value': debt_to_equity, 'percentile': debt_percentil},
    }

    score = sum(criterios.values())
    signal = 'BUY' if score >= 5 else 'HOLD' if score >= 3 else 'SELL'

    return {
        'ticker': ticker,
        'ROIC': roic,
        'EBIT Margin': ebit_margin,
        'Z-Score': zscore,
        'Quick Ratio': quick_ratio,
        'Interest Coverage': interest_coverage,
        'Debt/Equity': debt_to_equity,
        'Score': score,
        'Signal': signal,
        'Criterios': criterios,
        'Comparaciones': comparaciones
    }

def run_analysis(tickers_str: str, pesos_str: str):
    tickers = [t.strip().upper() for t in tickers_str.split(',')]
    pesos = np.array([float(p) for p in pesos_str.split(',')]) / 100

    benchmark = "^GSPC"
    data = yf.download(tickers + [benchmark], start="2020-01-01")["Close"].dropna()
    returns = data.pct_change().dropna()
    returns_annual = returns.mean() * 252
    vol_annual = returns.std() * np.sqrt(252)
    sharpe_ratios = (returns_annual - rf) / vol_annual
    drawdowns = {ticker: max_drawdown(data[ticker]) for ticker in tickers}

    benchmark_returns = returns[benchmark]
    betas = {
        ticker: np.cov(returns[ticker], benchmark_returns)[0][1] / np.var(benchmark_returns)
        for ticker in tickers
    }

    port_return = np.dot(returns_annual[tickers], pesos)
    port_vol = np.sqrt(np.dot(pesos.T, np.dot(returns[tickers].cov() * 252, pesos)))
    port_sharpe = (port_return - rf) / port_vol

    def negative_sharpe(weights, mean_returns, cov_matrix):
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(ret - rf) / vol

    bounds = tuple((0, 1) for _ in range(len(tickers)))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    opt_result = minimize(
        negative_sharpe,
        initial_weights,
        args=(returns_annual[tickers], returns[tickers].cov() * 252),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = opt_result.x

    fundamentales = [analizar_fundamentales(t) for t in tickers]
    resumen = pd.DataFrame({
        "Ticker": tickers,
        "Ret Anual %": returns_annual[tickers] * 100,
        "Vol Anual %": vol_annual[tickers] * 100,
        "Sharpe Ratio": sharpe_ratios[tickers],
        "Max Drawdown %": [drawdowns[t] * 100 for t in tickers],
        "Beta vs S&P500": [betas[t] for t in tickers],
        "Peso Óptimo": optimal_weights,
        "Señal": [f['Signal'] if f else 'N/A' for f in fundamentales]
    })

    return resumen, fundamentales
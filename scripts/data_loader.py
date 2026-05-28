import pandas as pd
from statsmodels.tsa.stattools import adfuller , kpss

class MacroDataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.stationarity_results =None
        self.feature_matrix = None

    def load(self):
        self.df = pd.read_csv(self.filename, parse_dates= ['date'], index_col ='date')
        return self.df  

    def run_stationarity_tests(self):

        original_cols = ['cpi_inflation', 'repo_rate', 'm3', 'crude_oil_brent',
                    'inr_usd', 'iip', 'tbill_91d', 'gsec_10yr', 
                    'term_spread', 'fed_funds_rate']
        results =[]
        for col in original_cols:
            adf_stat, adf_p, _, _, _, _, = adfuller(self.df[col].dropna())
            kpss_stat, kpss_p, _, _,  = kpss(self.df[col].dropna(), regression ='c')
            results.append({
                'variable': col,
                'adf_pvalue': round(adf_p,4),
                'kpss_pvalue':round(kpss_p,4),
                'adf_stationary': adf_p<0.05,
                'kpss_staionary': kpss_p>0.05
            })
        self.stationarity_results =pd.DataFrame(results)
        return self.stationarity_results
    
    def build_lag_features(self):
        df = self.df.copy()
        df['cpi_lag1'] = df['cpi_inflation'].shift(1)
        df['cpi_lag2'] = df['cpi_inflation'].shift(2)
        df['cpi_lag3'] = df['cpi_inflation'].shift(3)
        df['crude_lag1'] = df['crude_oil_brent'].shift(1)
        df['inr_lag1'] = df['inr_usd'].shift(1)
        df['m3_lag12'] = df['m3'].shift(12)
        df.dropna(inplace=True)
        self.feature_matrix = df
        return self.feature_matrix

    def get_feature_matrix(self):
        X = self.feature_matrix[['cpi_lag1', 'cpi_lag2', 'cpi_lag3',
                                'crude_lag1', 'inr_lag1', 'm3_lag12']]
        y = self.feature_matrix['cpi_inflation']
        return X, y
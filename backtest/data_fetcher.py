#!/usr/bin/env python3
"""
🌸 Historical Data Fetcher for Nanpin Backtesting
Fetches BTC price data and simulates macro indicators for comprehensive analysis
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    🌸 Historical Data Fetcher for Backtesting
    
    Features:
    - Multi-source BTC price data
    - Macro indicator simulation
    - Data validation and cleaning
    - Caching for performance
    - Export capabilities
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data fetcher
        
        Args:
            cache_dir: Directory for data caching
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data sources configuration
        self.data_sources = {
            'yahoo_finance': True,
            'coindesk_api': True,
            'simulated_macro': True
        }
        
        logger.info("🌸 Historical Data Fetcher initialized")
        logger.info(f"   Cache directory: {cache_dir}")
    
    async def fetch_complete_dataset(self, start_date: str, end_date: str) -> Dict:
        """
        Fetch complete dataset for backtesting
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Complete dataset with BTC prices and macro indicators
        """
        try:
            logger.info(f"📊 Fetching complete dataset from {start_date} to {end_date}")
            
            # Fetch BTC price data
            btc_data = await self._fetch_btc_data(start_date, end_date)
            
            # Fetch/simulate macro data
            macro_data = await self._fetch_macro_data(start_date, end_date)
            
            # Combine datasets
            combined_data = self._combine_datasets(btc_data, macro_data)
            
            # Validate data quality
            validation_results = self._validate_data(combined_data)
            
            # Cache the results
            await self._cache_data(combined_data, start_date, end_date)
            
            result = {
                'btc_data': btc_data,
                'macro_data': macro_data,
                'combined_data': combined_data,
                'validation': validation_results,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'fetched_at': datetime.now().isoformat(),
                    'sources': self.data_sources
                }
            }
            
            logger.info("✅ Complete dataset fetched successfully")
            logger.info(f"   BTC data points: {len(btc_data)}")
            logger.info(f"   Macro data points: {len(macro_data)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch complete dataset: {e}")
            raise
    
    async def _fetch_btc_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC price data from multiple sources"""
        try:
            logger.info("₿ Fetching BTC price data...")
            
            # Check cache first
            cache_key = f"btc_data_{start_date}_{end_date}"
            cached_data = await self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info("📦 Using cached BTC data")
                return cached_data
            
            # Primary source: Yahoo Finance
            btc_data = await self._fetch_yahoo_btc(start_date, end_date)
            
            # Fallback: CoinDesk API for missing data
            if btc_data.empty:
                logger.warning("Yahoo Finance failed, trying CoinDesk API...")
                btc_data = await self._fetch_coindesk_btc(start_date, end_date)
            
            # Enhance with additional metrics
            btc_data = self._enhance_btc_data(btc_data)
            
            # Cache the result
            await self._save_to_cache(cache_key, btc_data)
            
            logger.info(f"✅ BTC data fetched: {len(btc_data)} days")
            return btc_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch BTC data: {e}")
            raise
    
    async def _fetch_yahoo_btc(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC data from Yahoo Finance"""
        try:
            # Add buffer for technical indicators
            buffer_start = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=200)
            
            btc = yf.Ticker("BTC-USD")
            data = btc.history(
                start=buffer_start.strftime("%Y-%m-%d"),
                end=end_date,
                interval="1d"
            )
            
            if data.empty:
                raise Exception("No data returned from Yahoo Finance")
            
            # Clean column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Remove timezone info and ensure datetime index
            data.index = pd.to_datetime(data.index.date)
            
            return data.dropna()
            
        except Exception as e:
            logger.warning(f"Yahoo Finance BTC fetch failed: {e}")
            return pd.DataFrame()
    
    async def _fetch_coindesk_btc(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC data from CoinDesk API (fallback)"""
        try:
            # CoinDesk API for historical prices
            url = "https://api.coindesk.com/v1/bpi/historical/close.json"
            params = {
                'start': start_date,
                'end': end_date
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'bpi' not in data:
                raise Exception("Invalid CoinDesk API response")
            
            # Convert to DataFrame
            prices = data['bpi']
            df = pd.DataFrame.from_dict(prices, orient='index', columns=['Close'])
            df.index = pd.to_datetime(df.index)
            df['Close'] = df['Close'].astype(float)
            
            # Create OHLV data (simplified)
            df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            df['High'] = df['Close'] * (1 + np.random.uniform(0.001, 0.02, len(df)))
            df['Low'] = df['Close'] * (1 - np.random.uniform(0.001, 0.02, len(df)))
            df['Volume'] = 1000000  # Placeholder volume
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.warning(f"CoinDesk API fetch failed: {e}")
            return pd.DataFrame()
    
    def _enhance_btc_data(self, btc_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance BTC data with technical indicators"""
        try:
            if btc_data.empty:
                return btc_data
            
            # Price-based indicators
            btc_data['Returns'] = btc_data['Close'].pct_change()
            btc_data['Log_Returns'] = np.log(btc_data['Close'] / btc_data['Close'].shift(1))
            
            # Moving averages
            for period in [7, 14, 21, 50, 100, 200]:
                btc_data[f'MA_{period}'] = btc_data['Close'].rolling(period).mean()
                btc_data[f'MA_{period}_Distance'] = (btc_data['Close'] - btc_data[f'MA_{period}']) / btc_data[f'MA_{period}']
            
            # Volatility measures
            btc_data['Volatility_7d'] = btc_data['Returns'].rolling(7).std() * np.sqrt(365)
            btc_data['Volatility_30d'] = btc_data['Returns'].rolling(30).std() * np.sqrt(365)
            btc_data['Volatility_90d'] = btc_data['Returns'].rolling(90).std() * np.sqrt(365)
            
            # Price momentum
            btc_data['Momentum_7d'] = btc_data['Close'] / btc_data['Close'].shift(7) - 1
            btc_data['Momentum_30d'] = btc_data['Close'] / btc_data['Close'].shift(30) - 1
            btc_data['Momentum_90d'] = btc_data['Close'] / btc_data['Close'].shift(90) - 1
            
            # Support and resistance levels
            btc_data['Rolling_Max_90d'] = btc_data['High'].rolling(90).max()
            btc_data['Rolling_Min_90d'] = btc_data['Low'].rolling(90).min()
            btc_data['Price_Position'] = (btc_data['Close'] - btc_data['Rolling_Min_90d']) / (btc_data['Rolling_Max_90d'] - btc_data['Rolling_Min_90d'])
            
            # Drawdown from ATH
            btc_data['ATH'] = btc_data['High'].expanding().max()
            btc_data['Drawdown_from_ATH'] = (btc_data['Close'] - btc_data['ATH']) / btc_data['ATH']
            
            logger.debug("✅ BTC data enhanced with technical indicators")
            return btc_data
            
        except Exception as e:
            logger.warning(f"Failed to enhance BTC data: {e}")
            return btc_data
    
    async def _fetch_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch/simulate macro economic data"""
        try:
            logger.info("🔮 Generating macro economic data...")
            
            # Check cache first
            cache_key = f"macro_data_{start_date}_{end_date}"
            cached_data = await self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info("📦 Using cached macro data")
                return cached_data
            
            # Generate realistic macro data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            macro_data = pd.DataFrame(index=dates)
            
            # Simulate VIX (volatility index)
            macro_data['VIX'] = self._simulate_vix(len(dates))
            
            # Simulate Fear & Greed Index
            macro_data['Fear_Greed'] = self._simulate_fear_greed(macro_data['VIX'])
            
            # Simulate Fed Funds Rate
            macro_data['Fed_Rate'] = self._simulate_fed_rate(dates)
            
            # Simulate inflation indicators
            macro_data['CPI'] = self._simulate_inflation(dates)
            macro_data['Core_CPI'] = macro_data['CPI'] * np.random.uniform(0.8, 1.2, len(dates))
            
            # Simulate employment
            macro_data['Unemployment'] = self._simulate_unemployment(dates)
            
            # Simulate market indicators
            macro_data['DXY'] = self._simulate_dollar_index(dates)
            macro_data['Gold_Price'] = self._simulate_gold_price(dates)
            
            # Economic regime classification
            macro_data['Regime'] = self._classify_economic_regime(macro_data)
            
            # Position scaling factors
            macro_data['Position_Scaling'] = self._calculate_position_scaling(macro_data)
            
            # Prediction market indicators (simulated)
            macro_data['Recession_Probability'] = self._simulate_recession_prob(macro_data)
            macro_data['Rate_Cut_Probability'] = self._simulate_rate_cut_prob(macro_data)
            macro_data['Bitcoin_Sentiment'] = self._simulate_btc_sentiment(macro_data)
            
            # Cache the result
            await self._save_to_cache(cache_key, macro_data)
            
            logger.info(f"✅ Macro data generated: {len(macro_data)} days")
            return macro_data
            
        except Exception as e:
            logger.error(f"❌ Failed to generate macro data: {e}")
            raise
    
    def _simulate_vix(self, length: int) -> np.ndarray:
        """Simulate VIX-like volatility index"""
        # Base VIX around 20, with spikes during stress
        base_vix = 20
        noise = np.random.normal(0, 3, length)
        trend = np.random.normal(0, 0.5, length).cumsum() * 0.1
        
        # Add occasional spikes (crisis periods)
        spikes = np.random.choice([0, 1], size=length, p=[0.95, 0.05])
        spike_magnitude = np.random.exponential(15, length) * spikes
        
        vix = base_vix + noise + trend + spike_magnitude
        return np.clip(vix, 5, 80)
    
    def _simulate_fear_greed(self, vix: np.ndarray) -> np.ndarray:
        """Simulate Fear & Greed Index based on VIX"""
        # Inverse relationship with VIX
        base_fg = 100 - (vix - 10) * 1.5
        noise = np.random.normal(0, 8, len(vix))
        return np.clip(base_fg + noise, 0, 100)
    
    def _simulate_fed_rate(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Simulate Federal Funds Rate evolution"""
        rates = []
        current_rate = 1.5  # Starting rate
        
        for i, date in enumerate(dates):
            # Policy changes based on year
            year = date.year
            
            if year <= 2020:
                # Pre-COVID: gradual increases
                change = np.random.choice([0, 0.01, -0.01], p=[0.8, 0.15, 0.05])
            elif year == 2021:
                # COVID response: low rates
                change = np.random.choice([0, -0.01], p=[0.9, 0.1])
                current_rate = max(current_rate, 0.25)  # Floor at 0.25%
            elif year >= 2022:
                # Post-COVID: rate hikes
                change = np.random.choice([0, 0.01, 0.02], p=[0.7, 0.2, 0.1])
            else:
                change = 0
            
            current_rate = np.clip(current_rate + change, 0, 6)
            rates.append(current_rate)
        
        return np.array(rates)
    
    def _simulate_inflation(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Simulate CPI inflation"""
        inflation = []
        base_inflation = 2.0  # Target inflation
        
        for date in dates:
            year = date.year
            
            if year <= 2020:
                # Pre-COVID: stable inflation
                current = base_inflation + np.random.normal(0, 0.5)
            elif year == 2021:
                # COVID: deflationary pressure then recovery
                current = base_inflation + np.random.normal(-1, 1)
            elif year >= 2022:
                # Post-COVID: inflationary pressure
                current = base_inflation + np.random.normal(2, 1.5)
            else:
                current = base_inflation
            
            inflation.append(max(current, -2))  # Floor at -2%
        
        return np.array(inflation)
    
    def _simulate_unemployment(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Simulate unemployment rate"""
        unemployment = []
        base_rate = 4.0
        
        for date in dates:
            year = date.year
            
            if year <= 2019:
                current = base_rate + np.random.normal(0, 0.3)
            elif year == 2020:
                # COVID spike
                current = base_rate + np.random.normal(10, 3)
            elif year >= 2021:
                # Recovery
                current = base_rate + np.random.normal(1, 1)
            else:
                current = base_rate
            
            unemployment.append(max(current, 2))
        
        return np.array(unemployment)
    
    def _simulate_dollar_index(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Simulate DXY Dollar Index"""
        base_dxy = 100
        dxy = [base_dxy]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 0.5)
            new_value = dxy[-1] + change
            dxy.append(np.clip(new_value, 80, 120))
        
        return np.array(dxy)
    
    def _simulate_gold_price(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """Simulate gold price"""
        base_gold = 1800
        gold = [base_gold]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0, 20)
            new_price = gold[-1] + change
            gold.append(max(new_price, 1000))
        
        return np.array(gold)
    
    def _classify_economic_regime(self, macro_data: pd.DataFrame) -> np.ndarray:
        """Classify economic regime based on macro indicators"""
        regimes = []
        
        for i in range(len(macro_data)):
            vix = macro_data['VIX'].iloc[i]
            fed_rate = macro_data['Fed_Rate'].iloc[i]
            unemployment = macro_data['Unemployment'].iloc[i]
            fear_greed = macro_data['Fear_Greed'].iloc[i]
            
            if vix > 40 or unemployment > 8 or fear_greed < 20:
                regime = 'crisis'
            elif vix > 30 or unemployment > 6 or fear_greed < 35:
                regime = 'recession'
            elif fed_rate < 1 and unemployment < 5:
                regime = 'recovery'
            elif fear_greed > 80 and vix < 15:
                regime = 'bubble'
            elif macro_data['CPI'].iloc[i] > 4 and fed_rate > 3:
                regime = 'stagflation'
            else:
                regime = 'expansion'
            
            regimes.append(regime)
        
        return np.array(regimes)
    
    def _calculate_position_scaling(self, macro_data: pd.DataFrame) -> np.ndarray:
        """Calculate position scaling factors"""
        scaling = []
        
        regime_multipliers = {
            'crisis': 2.5,
            'recession': 2.0,
            'recovery': 1.2,
            'expansion': 1.0,
            'stagflation': 1.5,
            'bubble': 0.7
        }
        
        for i in range(len(macro_data)):
            regime = macro_data['Regime'].iloc[i]
            fear_greed = macro_data['Fear_Greed'].iloc[i]
            
            base_scaling = regime_multipliers[regime]
            
            # Fear/Greed adjustments
            if fear_greed < 20:
                fg_mult = 1.8
            elif fear_greed < 35:
                fg_mult = 1.4
            elif fear_greed > 80:
                fg_mult = 0.5
            else:
                fg_mult = 1.0
            
            total_scaling = base_scaling * fg_mult
            scaling.append(np.clip(total_scaling, 0.3, 3.0))
        
        return np.array(scaling)
    
    def _simulate_recession_prob(self, macro_data: pd.DataFrame) -> np.ndarray:
        """Simulate recession probability from prediction markets"""
        prob = []
        
        for i in range(len(macro_data)):
            vix = macro_data['VIX'].iloc[i]
            unemployment = macro_data['Unemployment'].iloc[i]
            
            base_prob = min(50, (vix - 20) * 2 + (unemployment - 4) * 5)
            noise = np.random.normal(0, 10)
            recession_prob = np.clip(base_prob + noise, 0, 100)
            prob.append(recession_prob)
        
        return np.array(prob)
    
    def _simulate_rate_cut_prob(self, macro_data: pd.DataFrame) -> np.ndarray:
        """Simulate rate cut probability"""
        prob = []
        
        for i in range(len(macro_data)):
            fed_rate = macro_data['Fed_Rate'].iloc[i]
            vix = macro_data['VIX'].iloc[i]
            
            # Higher probability when rates are high or stress is high
            base_prob = fed_rate * 15 + max(0, vix - 25) * 2
            noise = np.random.normal(0, 15)
            cut_prob = np.clip(base_prob + noise, 0, 100)
            prob.append(cut_prob)
        
        return np.array(prob)
    
    def _simulate_btc_sentiment(self, macro_data: pd.DataFrame) -> np.ndarray:
        """Simulate Bitcoin sentiment from prediction markets"""
        sentiment = []
        
        for i in range(len(macro_data)):
            fear_greed = macro_data['Fear_Greed'].iloc[i]
            fed_rate = macro_data['Fed_Rate'].iloc[i]
            
            # Bitcoin sentiment influenced by fear/greed and monetary policy
            base_sentiment = fear_greed * 0.7 + (6 - fed_rate) * 8
            noise = np.random.normal(0, 10)
            btc_sentiment = np.clip(base_sentiment + noise, 0, 100)
            sentiment.append(btc_sentiment)
        
        return np.array(sentiment)
    
    def _combine_datasets(self, btc_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Combine BTC and macro datasets"""
        try:
            # Align dates
            common_dates = btc_data.index.intersection(macro_data.index)
            
            if len(common_dates) == 0:
                raise Exception("No overlapping dates between BTC and macro data")
            
            # Combine data
            combined = btc_data.loc[common_dates].copy()
            
            # Add macro indicators
            for col in macro_data.columns:
                combined[f'Macro_{col}'] = macro_data.loc[common_dates, col]
            
            logger.info(f"✅ Datasets combined: {len(combined)} overlapping days")
            return combined
            
        except Exception as e:
            logger.error(f"❌ Failed to combine datasets: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> Dict:
        """Validate data quality"""
        try:
            validation = {
                'total_records': len(data),
                'date_range': {
                    'start': data.index.min().strftime('%Y-%m-%d'),
                    'end': data.index.max().strftime('%Y-%m-%d')
                },
                'missing_data': {},
                'data_quality': {},
                'warnings': []
            }
            
            # Check for missing data
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                missing_pct = missing_count / len(data) * 100
                validation['missing_data'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                if missing_pct > 5:
                    validation['warnings'].append(f"High missing data in {col}: {missing_pct:.1f}%")
            
            # Check data ranges
            if 'Close' in data.columns:
                btc_range = data['Close'].max() / data['Close'].min()
                validation['data_quality']['btc_price_range'] = round(btc_range, 2)
                
                if btc_range > 50:  # More than 50x range
                    validation['warnings'].append(f"Extreme BTC price range: {btc_range:.1f}x")
            
            # Check for sufficient data
            if len(data) < 365:
                validation['warnings'].append(f"Limited data: only {len(data)} days")
            
            logger.info(f"✅ Data validation completed: {len(validation['warnings'])} warnings")
            return validation
            
        except Exception as e:
            logger.warning(f"Data validation failed: {e}")
            return {'error': str(e)}
    
    async def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_file):
                # Check if cache is recent (less than 1 day old)
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                
                if cache_age < timedelta(days=1):
                    return pd.read_pickle(cache_file)
            
            return None
            
        except Exception as e:
            logger.debug(f"Cache load failed for {cache_key}: {e}")
            return None
    
    async def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            data.to_pickle(cache_file)
            logger.debug(f"Data cached: {cache_key}")
            
        except Exception as e:
            logger.debug(f"Cache save failed for {cache_key}: {e}")
    
    async def _cache_data(self, data: pd.DataFrame, start_date: str, end_date: str):
        """Cache complete dataset"""
        try:
            cache_key = f"complete_dataset_{start_date}_{end_date}"
            await self._save_to_cache(cache_key, data)
            
        except Exception as e:
            logger.debug(f"Failed to cache complete dataset: {e}")


async def test_data_fetcher():
    """Test the data fetcher"""
    print("🧪 Testing Historical Data Fetcher")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize fetcher
    fetcher = HistoricalDataFetcher()
    
    # Test data fetching
    try:
        dataset = await fetcher.fetch_complete_dataset("2022-01-01", "2022-12-31")
        
        print(f"\n✅ Dataset fetched successfully:")
        print(f"   BTC data points: {len(dataset['btc_data'])}")
        print(f"   Macro data points: {len(dataset['macro_data'])}")
        print(f"   Combined data points: {len(dataset['combined_data'])}")
        print(f"   Validation warnings: {len(dataset['validation'].get('warnings', []))}")
        
        # Display sample data
        if not dataset['combined_data'].empty:
            print(f"\n📊 Sample data (last 5 days):")
            sample = dataset['combined_data'].tail()
            print(sample[['Close', 'Macro_VIX', 'Macro_Fear_Greed', 'Macro_Regime']].to_string())
        
        print(f"\n🏁 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_data_fetcher())
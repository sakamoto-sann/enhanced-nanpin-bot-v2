#!/usr/bin/env python3
"""
🌸 FRED (Federal Reserve Economic Data) API Client
Clean wrapper for accessing Federal Reserve economic indicators
for macro-informed Nanpin trading strategy
"""

import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class FREDSeries:
    """FRED economic data series"""
    id: str
    title: str
    units: str
    frequency: str
    last_updated: datetime
    observations: List[Dict]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert observations to pandas DataFrame"""
        try:
            df = pd.DataFrame(self.observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df[['value']].rename(columns={'value': self.id})
        except Exception as e:
            logger.error(f"Failed to convert {self.id} to DataFrame: {e}")
            return pd.DataFrame()

@dataclass
class EconomicIndicator:
    """Processed economic indicator with analysis"""
    series_id: str
    name: str
    current_value: float
    previous_value: float
    change_absolute: float
    change_percent: float
    percentile_1y: float
    percentile_5y: float
    trend: str  # 'rising', 'falling', 'stable'
    signal: str  # 'bullish', 'bearish', 'neutral'
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'series_id': self.series_id,
            'name': self.name,
            'current_value': self.current_value,
            'previous_value': self.previous_value,
            'change_absolute': self.change_absolute,
            'change_percent': self.change_percent,
            'percentile_1y': self.percentile_1y,
            'percentile_5y': self.percentile_5y,
            'trend': self.trend,
            'signal': self.signal,
            'last_updated': self.last_updated.isoformat()
        }

class FREDClient:
    """
    🌸 Federal Reserve Economic Data (FRED) API Client
    
    Features:
    - Async data fetching for high performance
    - Automatic data analysis and signal generation
    - Historical percentile calculations
    - Bitcoin-relevant economic indicators
    - Rate limiting and error handling
    - Data caching for efficiency
    """
    
    def __init__(self, api_key: str):
        """
        Initialize FRED API Client
        
        Args:
            api_key: FRED API key
        """
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        self.session = None
        
        # Cache for avoiding redundant API calls
        self.cache = {}
        self.cache_duration = 3600  # 1 hour
        
        # Key economic indicators for Bitcoin trading
        self.key_indicators = {
            # Monetary Policy
            'FEDFUNDS': {
                'name': 'Federal Funds Rate',
                'category': 'monetary_policy',
                'bitcoin_impact': 'inverse',  # Lower rates = bullish Bitcoin
                'weight': 0.3
            },
            'GS10': {
                'name': '10-Year Treasury Rate', 
                'category': 'monetary_policy',
                'bitcoin_impact': 'inverse',
                'weight': 0.25
            },
            'GS2': {
                'name': '2-Year Treasury Rate',
                'category': 'monetary_policy', 
                'bitcoin_impact': 'inverse',
                'weight': 0.2
            },
            
            # Inflation
            'CPILFESL': {
                'name': 'Core CPI',
                'category': 'inflation',
                'bitcoin_impact': 'positive',  # Inflation hedge
                'weight': 0.25
            },
            'CPIAUCSL': {
                'name': 'Consumer Price Index',
                'category': 'inflation',
                'bitcoin_impact': 'positive',
                'weight': 0.2
            },
            'DFEDTARU': {
                'name': 'Fed Target Rate Upper Bound',
                'category': 'inflation',
                'bitcoin_impact': 'inverse',
                'weight': 0.15
            },
            
            # Economic Growth
            'GDP': {
                'name': 'Gross Domestic Product',
                'category': 'growth',
                'bitcoin_impact': 'positive',  # Growth supports risk assets
                'weight': 0.2
            },
            'UNRATE': {
                'name': 'Unemployment Rate',
                'category': 'growth',
                'bitcoin_impact': 'inverse',  # High unemployment = economic stress
                'weight': 0.25
            },
            'PAYEMS': {
                'name': 'Non-Farm Payrolls',
                'category': 'growth', 
                'bitcoin_impact': 'positive',
                'weight': 0.15
            },
            
            # Money Supply & Liquidity
            'M2SL': {
                'name': 'M2 Money Supply',
                'category': 'liquidity',
                'bitcoin_impact': 'positive',  # Money printing = Bitcoin bullish
                'weight': 0.3
            },
            'WALCL': {
                'name': 'Fed Balance Sheet',
                'category': 'liquidity',
                'bitcoin_impact': 'positive',
                'weight': 0.25
            },
            
            # Market Indicators
            'VIXCLS': {
                'name': 'VIX Volatility Index',
                'category': 'market_stress',
                'bitcoin_impact': 'complex',  # High VIX can be bullish (crisis hedge) or bearish (risk-off)
                'weight': 0.3
            },
            'DEXUSEU': {
                'name': 'US Dollar Index',
                'category': 'currency',
                'bitcoin_impact': 'inverse',  # Strong dollar typically bearish for Bitcoin
                'weight': 0.25
            },
            
            # Real Estate & Asset Bubbles
            'CSUSHPISA': {
                'name': 'Case-Shiller Home Price Index',
                'category': 'assets',
                'bitcoin_impact': 'positive',  # Asset inflation
                'weight': 0.1
            }
        }
        
        logger.info("🌸 FRED Client initialized")
        logger.info(f"   API Key: {api_key[:8]}...")
        logger.info(f"   Tracking {len(self.key_indicators)} indicators")
    
    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()
        
        # Test API connection
        await self._test_connection()
        logger.info("✅ FRED Client ready")
    
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self):
        """Test FRED API connection"""
        try:
            # Test with a simple request
            url = f"{self.base_url}/series"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'seriess' in data and len(data['seriess']) > 0:
                        logger.info("✅ FRED API connection successful")
                        return True
                
                raise Exception(f"FRED API test failed: {response.status}")
                
        except Exception as e:
            logger.error(f"❌ FRED API connection failed: {e}")
            raise
    
    def _get_cache_key(self, series_id: str, params: Dict) -> str:
        """Generate cache key"""
        return f"{series_id}_{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (datetime.now().timestamp() - cache_time) < self.cache_duration
    
    async def fetch_series(self, series_id: str, limit: int = 120, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Optional[FREDSeries]:
        """
        Fetch FRED data series
        
        Args:
            series_id: FRED series ID (e.g., 'FEDFUNDS')
            limit: Number of observations to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            FREDSeries object or None if failed
        """
        try:
            # Check cache first
            params = {'limit': limit, 'start_date': start_date, 'end_date': end_date}
            cache_key = self._get_cache_key(series_id, params)
            
            if self._is_cache_valid(cache_key):
                logger.debug(f"📦 Using cached data for {series_id}")
                return self.cache[cache_key]['data']
            
            # Fetch series metadata
            series_url = f"{self.base_url}/series"
            series_params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            async with self.session.get(series_url, params=series_params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch series metadata for {series_id}: {response.status}")
                    return None
                
                series_data = await response.json()
                if 'seriess' not in series_data or len(series_data['seriess']) == 0:
                    logger.error(f"No series data found for {series_id}")
                    return None
                
                series_info = series_data['seriess'][0]
            
            # Fetch observations
            obs_url = f"{self.base_url}/series/observations"
            obs_params = {
                'series_id': series_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
                'sort_order': 'desc'
            }
            
            if start_date:
                obs_params['observation_start'] = start_date
            if end_date:
                obs_params['observation_end'] = end_date
            
            async with self.session.get(obs_url, params=obs_params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch observations for {series_id}: {response.status}")
                    return None
                
                obs_data = await response.json()
                observations = obs_data.get('observations', [])
                
                if not observations:
                    logger.warning(f"No observations found for {series_id}")
                    return None
            
            # Create FREDSeries object
            fred_series = FREDSeries(
                id=series_id,
                title=series_info.get('title', series_id),
                units=series_info.get('units', ''),
                frequency=series_info.get('frequency', ''),
                last_updated=datetime.now(),
                observations=observations
            )
            
            # Cache the result
            self.cache[cache_key] = {
                'data': fred_series,
                'timestamp': datetime.now().timestamp()
            }
            
            logger.debug(f"📊 Fetched {len(observations)} observations for {series_id}")
            return fred_series
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch series {series_id}: {e}")
            return None
    
    async def fetch_multiple_series(self, series_ids: List[str], 
                                  limit: int = 120) -> Dict[str, FREDSeries]:
        """
        Fetch multiple FRED series concurrently
        
        Args:
            series_ids: List of FRED series IDs
            limit: Number of observations per series
            
        Returns:
            Dictionary mapping series_id to FREDSeries
        """
        try:
            logger.info(f"📊 Fetching {len(series_ids)} FRED series...")
            
            # Create tasks for concurrent fetching
            tasks = []
            for series_id in series_ids:
                task = self.fetch_series(series_id, limit)
                tasks.append(task)
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            series_data = {}
            for series_id, result in zip(series_ids, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch {series_id}: {result}")
                    continue
                
                if result is not None:
                    series_data[series_id] = result
                    
                # Rate limiting
                await asyncio.sleep(0.1)
            
            logger.info(f"✅ Successfully fetched {len(series_data)} series")
            return series_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch multiple series: {e}")
            return {}
    
    def analyze_indicator(self, series: FREDSeries, config: Dict) -> Optional[EconomicIndicator]:
        """
        Analyze economic indicator and generate trading signal
        
        Args:
            series: FRED series data
            config: Indicator configuration
            
        Returns:
            EconomicIndicator with analysis
        """
        try:
            df = series.to_dataframe()
            if df.empty or len(df) < 2:
                logger.warning(f"Insufficient data for {series.id}")
                return None
            
            # Get current and previous values
            current_value = df.iloc[-1, 0]
            previous_value = df.iloc[-2, 0] if len(df) >= 2 else current_value
            
            # Skip if values are NaN
            if pd.isna(current_value) or pd.isna(previous_value):
                logger.warning(f"NaN values in {series.id}")
                return None
            
            # Calculate changes
            change_absolute = current_value - previous_value
            change_percent = (change_absolute / previous_value * 100) if previous_value != 0 else 0
            
            # Calculate percentiles
            values_1y = df.tail(12).iloc[:, 0].dropna()  # Last 12 observations (roughly 1 year for monthly data)
            values_5y = df.tail(60).iloc[:, 0].dropna()  # Last 60 observations (roughly 5 years)
            
            percentile_1y = (values_1y < current_value).mean() * 100 if len(values_1y) > 0 else 50
            percentile_5y = (values_5y < current_value).mean() * 100 if len(values_5y) > 0 else 50
            
            # Determine trend
            if abs(change_percent) < 0.5:
                trend = 'stable'
            elif change_percent > 0:
                trend = 'rising'
            else:
                trend = 'falling'
            
            # Generate Bitcoin trading signal
            signal = self._generate_bitcoin_signal(series.id, current_value, change_percent, 
                                                 percentile_1y, percentile_5y, config)
            
            indicator = EconomicIndicator(
                series_id=series.id,
                name=config['name'],
                current_value=current_value,
                previous_value=previous_value,
                change_absolute=change_absolute,
                change_percent=change_percent,
                percentile_1y=percentile_1y,
                percentile_5y=percentile_5y,
                trend=trend,
                signal=signal,
                last_updated=datetime.now()
            )
            
            logger.debug(f"📈 {config['name']}: {current_value:.2f} ({signal}, {trend})")
            return indicator
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {series.id}: {e}")
            return None
    
    def _generate_bitcoin_signal(self, series_id: str, current_value: float, 
                               change_percent: float, percentile_1y: float, 
                               percentile_5y: float, config: Dict) -> str:
        """Generate Bitcoin trading signal from economic indicator"""
        try:
            bitcoin_impact = config.get('bitcoin_impact', 'neutral')
            
            # Base signal from trend and percentiles
            if bitcoin_impact == 'inverse':
                # Lower values are bullish for Bitcoin (e.g., interest rates)
                if percentile_1y < 25:  # In bottom quartile
                    base_signal = 'bullish'
                elif percentile_1y > 75:  # In top quartile  
                    base_signal = 'bearish'
                else:
                    base_signal = 'neutral'
                    
                # Strengthen signal if trend supports it
                if base_signal == 'bullish' and change_percent < -2:
                    return 'bullish'
                elif base_signal == 'bearish' and change_percent > 2:
                    return 'bearish'
                    
            elif bitcoin_impact == 'positive':
                # Higher values are bullish for Bitcoin (e.g., inflation, money supply)
                if percentile_1y > 75:
                    base_signal = 'bullish'
                elif percentile_1y < 25:
                    base_signal = 'bearish'
                else:
                    base_signal = 'neutral'
                    
                if base_signal == 'bullish' and change_percent > 2:
                    return 'bullish'
                elif base_signal == 'bearish' and change_percent < -2:
                    return 'bearish'
                    
            elif bitcoin_impact == 'complex':
                # Special logic for complex indicators like VIX
                if series_id == 'VIXCLS':
                    if current_value > 30:  # High fear
                        return 'bullish'  # Bitcoin as crisis hedge
                    elif current_value < 15:  # Complacency
                        return 'neutral'  # Risk assets generally stable
                    else:
                        return 'neutral'
            
            return 'neutral'
            
        except Exception as e:
            logger.warning(f"Failed to generate signal for {series_id}: {e}")
            return 'neutral'
    
    async def get_key_indicators(self) -> Dict[str, EconomicIndicator]:
        """
        Fetch and analyze all key economic indicators
        
        Returns:
            Dictionary of analyzed economic indicators
        """
        try:
            logger.info("📊 Fetching key economic indicators...")
            
            # Fetch all series
            series_ids = list(self.key_indicators.keys())
            series_data = await self.fetch_multiple_series(series_ids)
            
            # Analyze each indicator
            indicators = {}
            for series_id, series in series_data.items():
                config = self.key_indicators[series_id]
                indicator = self.analyze_indicator(series, config)
                
                if indicator:
                    indicators[series_id] = indicator
            
            logger.info(f"✅ Analyzed {len(indicators)} economic indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to get key indicators: {e}")
            return {}
    
    async def get_inflation_outlook(self) -> Dict[str, Union[float, str]]:
        """Get comprehensive inflation outlook"""
        try:
            inflation_series = ['CPILFESL', 'CPIAUCSL', 'DFEDTARU']
            series_data = await self.fetch_multiple_series(inflation_series)
            
            outlook = {
                'current_core_cpi': 0.0,
                'cpi_trend': 'neutral',
                'fed_target': 0.0,
                'inflation_signal': 'neutral'
            }
            
            if 'CPILFESL' in series_data:
                df = series_data['CPILFESL'].to_dataframe()
                if not df.empty:
                    outlook['current_core_cpi'] = df.iloc[-1, 0]
                    if len(df) >= 12:
                        yoy_change = ((df.iloc[-1, 0] - df.iloc[-13, 0]) / df.iloc[-13, 0]) * 100
                        outlook['cpi_trend'] = 'rising' if yoy_change > 3 else 'falling' if yoy_change < 2 else 'stable'
            
            if 'DFEDTARU' in series_data:
                df = series_data['DFEDTARU'].to_dataframe()
                if not df.empty:
                    outlook['fed_target'] = df.iloc[-1, 0]
            
            # Generate overall inflation signal
            if outlook['cpi_trend'] == 'rising' and outlook['current_core_cpi'] > 3:
                outlook['inflation_signal'] = 'bullish'  # High inflation bullish for Bitcoin
            elif outlook['cpi_trend'] == 'falling' and outlook['current_core_cpi'] < 2:
                outlook['inflation_signal'] = 'bearish'
            
            return outlook
            
        except Exception as e:
            logger.error(f"❌ Failed to get inflation outlook: {e}")
            return {}
    
    async def get_fed_policy_outlook(self) -> Dict[str, Union[float, str]]:
        """Get Federal Reserve policy outlook"""
        try:
            fed_series = ['FEDFUNDS', 'GS10', 'GS2']
            series_data = await self.fetch_multiple_series(fed_series)
            
            outlook = {
                'current_fed_rate': 0.0,
                'yield_curve_slope': 0.0,
                'rate_trend': 'neutral',
                'policy_signal': 'neutral'
            }
            
            if 'FEDFUNDS' in series_data:
                df = series_data['FEDFUNDS'].to_dataframe()
                if not df.empty:
                    outlook['current_fed_rate'] = df.iloc[-1, 0]
                    if len(df) >= 3:
                        recent_change = df.iloc[-1, 0] - df.iloc[-4, 0]  # 3-month change
                        outlook['rate_trend'] = 'rising' if recent_change > 0.25 else 'falling' if recent_change < -0.25 else 'stable'
            
            # Calculate yield curve slope (10Y - 2Y)
            if 'GS10' in series_data and 'GS2' in series_data:
                df_10y = series_data['GS10'].to_dataframe()
                df_2y = series_data['GS2'].to_dataframe()
                if not df_10y.empty and not df_2y.empty:
                    outlook['yield_curve_slope'] = df_10y.iloc[-1, 0] - df_2y.iloc[-1, 0]
            
            # Generate policy signal
            if outlook['rate_trend'] == 'falling' or outlook['current_fed_rate'] < 2:
                outlook['policy_signal'] = 'bullish'  # Easing policy bullish for Bitcoin
            elif outlook['rate_trend'] == 'rising' and outlook['current_fed_rate'] > 4:
                outlook['policy_signal'] = 'bearish'  # Tight policy bearish for Bitcoin
            
            return outlook
            
        except Exception as e:
            logger.error(f"❌ Failed to get Fed policy outlook: {e}")
            return {}
    
    def calculate_macro_score(self, indicators: Dict[str, EconomicIndicator]) -> Dict[str, float]:
        """
        Calculate overall macro score for Bitcoin
        
        Args:
            indicators: Dictionary of analyzed indicators
            
        Returns:
            Macro score breakdown
        """
        try:
            category_scores = {
                'monetary_policy': 0.0,
                'inflation': 0.0,
                'growth': 0.0,
                'liquidity': 0.0,
                'market_stress': 0.0,
                'currency': 0.0
            }
            
            category_weights = {
                'monetary_policy': 0.0,
                'inflation': 0.0,
                'growth': 0.0,
                'liquidity': 0.0,
                'market_stress': 0.0,
                'currency': 0.0
            }
            
            # Calculate weighted scores by category
            for series_id, indicator in indicators.items():
                config = self.key_indicators.get(series_id, {})
                category = config.get('category', 'other')
                weight = config.get('weight', 0.1)
                
                if category in category_scores:
                    # Convert signal to numeric score
                    signal_score = {'bullish': 1.0, 'neutral': 0.0, 'bearish': -1.0}.get(indicator.signal, 0.0)
                    
                    category_scores[category] += signal_score * weight
                    category_weights[category] += weight
            
            # Normalize scores
            for category in category_scores:
                if category_weights[category] > 0:
                    category_scores[category] /= category_weights[category]
            
            # Calculate overall score
            overall_score = sum(category_scores.values()) / len(category_scores)
            
            return {
                'overall_score': overall_score,
                'monetary_policy': category_scores['monetary_policy'],
                'inflation': category_scores['inflation'],
                'growth': category_scores['growth'],
                'liquidity': category_scores['liquidity'],
                'market_stress': category_scores['market_stress'],
                'currency': category_scores['currency']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate macro score: {e}")
            return {'overall_score': 0.0}
    
    def export_indicators(self, indicators: Dict[str, EconomicIndicator]) -> Dict:
        """Export indicators to dictionary format"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'indicator_count': len(indicators),
                'indicators': {},
                'summary': {}
            }
            
            # Export individual indicators
            for series_id, indicator in indicators.items():
                export_data['indicators'][series_id] = indicator.to_dict()
            
            # Generate summary statistics
            signals = [ind.signal for ind in indicators.values()]
            export_data['summary'] = {
                'bullish_count': signals.count('bullish'),
                'bearish_count': signals.count('bearish'),
                'neutral_count': signals.count('neutral'),
                'bullish_percentage': (signals.count('bullish') / len(signals) * 100) if signals else 0
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export indicators: {e}")
            return {}


# ===========================
# UTILITY FUNCTIONS
# ===========================

async def test_fred_client():
    """Test FRED Client functionality"""
    print("🧪 Testing FRED Client")
    print("=" * 50)
    
    # Initialize client
    api_key = "7aa42875026454682d22f3e02afff1b2"  # Using provided key
    client = FREDClient(api_key)
    
    try:
        await client.initialize()
        
        # Test single series fetch
        print("\n📊 Testing single series fetch...")
        fed_funds = await client.fetch_series('FEDFUNDS', limit=12)
        if fed_funds:
            print(f"   ✅ Fetched {len(fed_funds.observations)} Fed Funds observations")
            print(f"   Latest value: {fed_funds.observations[0]['value']}%")
        
        # Test multiple series fetch
        print("\n📊 Testing multiple series fetch...")
        indicators = await client.get_key_indicators()
        print(f"   ✅ Fetched {len(indicators)} key indicators")
        
        for series_id, indicator in list(indicators.items())[:5]:  # Show first 5
            print(f"   {indicator.name}: {indicator.current_value:.2f} ({indicator.signal})")
        
        # Test macro score calculation
        print("\n📊 Testing macro score calculation...")
        macro_scores = client.calculate_macro_score(indicators)
        print(f"   Overall Bitcoin macro score: {macro_scores['overall_score']:.2f}")
        print(f"   Monetary policy: {macro_scores['monetary_policy']:.2f}")
        print(f"   Inflation: {macro_scores['inflation']:.2f}")
        
        # Test specialized outlooks
        print("\n📊 Testing specialized outlooks...")
        inflation_outlook = await client.get_inflation_outlook()
        fed_outlook = await client.get_fed_policy_outlook()
        
        print(f"   Inflation outlook: {inflation_outlook.get('inflation_signal', 'unknown')}")
        print(f"   Fed policy outlook: {fed_outlook.get('policy_signal', 'unknown')}")
        
    finally:
        await client.close()
    
    print("\n🏁 FRED Client test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_fred_client())
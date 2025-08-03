#!/usr/bin/env python3
"""
🌸 Liquidation Data Aggregator for Nanpin Strategy
Multi-exchange liquidation intelligence optimized for 永久ナンピン DCA entry timing
"""

import asyncio
import aiohttp
import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64

logger = logging.getLogger(__name__)

class LiquidationSource(Enum):
    """Supported liquidation data sources"""
    HYPERLIQUID = "hyperliquid"
    BINANCE = "binance" 
    COINGLASS = "coinglass"
    FLIPSIDE = "flipside"
    COINMARKETCAP = "coinmarketcap"
    COINGECKO = "coingecko"

@dataclass
class LiquidationCluster:
    """Liquidation cluster at specific price level"""
    price: float
    volume: float  # Total liquidation volume at this price
    long_volume: float  # Long liquidation volume
    short_volume: float  # Short liquidation volume
    exchange_count: int  # Number of exchanges reporting this level
    confidence: float  # Confidence score (0-1)
    significance: str  # 'low', 'medium', 'high', 'critical'
    sources: List[str]  # Data sources confirming this level

@dataclass
class LiquidationHeatmap:
    """Complete liquidation heatmap analysis"""
    symbol: str
    timestamp: datetime
    current_price: float
    clusters: List[LiquidationCluster]
    major_support_levels: List[float]  # Strong liquidation support below price
    major_resistance_levels: List[float]  # Strong liquidation resistance above price
    nanpin_opportunities: List[Dict]  # Recommended DCA entry levels
    risk_assessment: Dict  # Overall risk metrics
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'clusters': [asdict(cluster) for cluster in self.clusters],
            'major_support_levels': self.major_support_levels,
            'major_resistance_levels': self.major_resistance_levels,
            'nanpin_opportunities': self.nanpin_opportunities,
            'risk_assessment': self.risk_assessment
        }

class LiquidationAggregator:
    """
    🌸 Multi-Source Liquidation Data Aggregator for Nanpin Strategy
    
    Features:
    - Aggregates liquidation data from multiple free-tier APIs
    - Identifies liquidation clusters for DCA entry timing
    - Provides Fibonacci-aligned liquidation analysis
    - Risk assessment based on liquidation proximity
    - Optimized for permanent accumulation strategy
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Liquidation Aggregator
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or self._get_default_config()
        self.session = None
        
        # API configurations
        self.api_configs = {
            LiquidationSource.COINGLASS: {
                "base_url": "https://open-api-v4.coinglass.com",
                "api_key": self.config.get('api_keys', {}).get('coinglass'),
                "rate_limit": 60,  # 60 requests per minute (free tier)
                "endpoints": {
                    "liquidation_map": "/api/futures/liquidation_map",
                    "open_interest": "/api/futures/open_interest",
                    "liquidation_history": "/api/futures/liquidation"
                }
            },
            LiquidationSource.COINMARKETCAP: {
                "base_url": "https://pro-api.coinmarketcap.com",
                "api_key": self.config.get('api_keys', {}).get('coinmarketcap'),
                "rate_limit": 15,  # Conservative for 333/day limit
                "endpoints": {
                    "quotes": "/v1/cryptocurrency/quotes/latest",
                    "global": "/v1/global-metrics/quotes/latest"
                }
            },
            LiquidationSource.COINGECKO: {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.config.get('api_keys', {}).get('coingecko'),  # Optional for free tier
                "rate_limit": 30,  # 30 requests per minute (demo key)
                "endpoints": {
                    "price": "/simple/price",
                    "market_data": "/coins/bitcoin/market_chart",
                    "derivatives": "/derivatives"
                }
            },
            LiquidationSource.FLIPSIDE: {
                "base_url": "https://api.flipsidecrypto.com/api/v2",
                "api_key": self.config.get('api_keys', {}).get('flipside'),
                "rate_limit": 10,  # Conservative for free tier
                "endpoints": {
                    "query": "/queries",
                    "results": "/query_results"
                }
            }
        }
        
        # Rate limiting tracking
        self.last_request_times = {source: 0 for source in LiquidationSource}
        
        # Cache for liquidation data
        self.cache = {}
        self.cache_ttl = {
            'liquidation_map': 180,    # 3 minutes
            'price_data': 60,         # 1 minute
            'market_data': 300        # 5 minutes
        }
        
        # Analysis parameters
        self.clustering_threshold = 0.01  # 1% price clustering
        self.min_cluster_volume = 1000000  # $1M minimum for significant cluster
        self.significance_levels = {
            'low': 1000000,      # $1M
            'medium': 5000000,   # $5M
            'high': 20000000,    # $20M
            'critical': 50000000 # $50M
        }
        
        logger.info("🌸 Liquidation Aggregator initialized for Nanpin strategy")
        logger.info(f"   Configured sources: {list(self.api_configs.keys())}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'api_keys': {},
            'symbol': 'BTC',
            'clustering_threshold': 0.01,
            'min_cluster_volume': 1000000
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'NanpinBot/1.0 LiquidationAggregator',
                    'Accept': 'application/json'
                }
            )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _rate_limit_check(self, source: LiquidationSource):
        """Check and enforce rate limiting"""
        if source not in self.api_configs:
            return
        
        config = self.api_configs[source]
        rate_limit = config.get('rate_limit', 10)
        min_interval = 60 / rate_limit  # seconds between requests
        
        now = time.time()
        last_request = self.last_request_times[source]
        time_since_last = now - last_request
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"⏱️ Rate limiting {source.value}: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        self.last_request_times[source] = time.time()
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_data = self.cache[cache_key]
        cache_age = time.time() - cached_data['timestamp']
        ttl = self.cache_ttl.get(cache_type, 300)
        
        return cache_age < ttl
    
    async def _make_request(self, source: LiquidationSource, endpoint: str, 
                          params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make authenticated request to API"""
        try:
            await self._rate_limit_check(source)
            
            if source not in self.api_configs:
                logger.warning(f"⚠️ Source {source.value} not configured")
                return None
            
            config = self.api_configs[source]
            base_url = config['base_url']
            api_key = config.get('api_key')
            
            url = f"{base_url}{endpoint}"
            request_headers = headers or {}
            request_params = params or {}
            
            # Add authentication based on source
            if api_key:
                if source == LiquidationSource.COINGLASS:
                    request_headers['X-API-KEY'] = api_key
                elif source == LiquidationSource.COINMARKETCAP:
                    request_headers['X-CMC_PRO_API_KEY'] = api_key
                elif source == LiquidationSource.COINGECKO:
                    request_headers['x-cg-demo-api-key'] = api_key
                elif source == LiquidationSource.FLIPSIDE:
                    request_headers['X-API-Key'] = api_key
            
            async with self.session.get(url, params=request_params, headers=request_headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"✅ {source.value} request successful")\n                    return data\n                elif response.status == 429:\n                    logger.warning(f\"⏱️ Rate limited by {source.value}\")\n                    await asyncio.sleep(5)\n                    return None\n                else:\n                    error_text = await response.text()\n                    logger.warning(f\"⚠️ {source.value} request failed ({response.status}): {error_text}\")\n                    return None\n                    \n        except Exception as e:\n            logger.warning(f\"⚠️ {source.value} request error: {e}\")\n            return None\n    \n    async def get_coinglass_liquidation_map(self, symbol: str = 'BTC') -> Optional[Dict]:\n        \"\"\"Get liquidation heatmap from CoinGlass (free tier)\"\"\"\n        try:\n            cache_key = f\"coinglass_liq_map_{symbol}\"\n            if self._is_cache_valid(cache_key, 'liquidation_map'):\n                return self.cache[cache_key]['data']\n            \n            endpoint = self.api_configs[LiquidationSource.COINGLASS]['endpoints']['liquidation_map']\n            params = {\n                'symbol': symbol,\n                'type': 'liquidation_map'\n            }\n            \n            data = await self._make_request(LiquidationSource.COINGLASS, endpoint, params)\n            \n            if data:\n                # Cache the result\n                self.cache[cache_key] = {\n                    'data': data,\n                    'timestamp': time.time()\n                }\n                logger.info(f\"📊 Retrieved CoinGlass liquidation map for {symbol}\")\n            \n            return data\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get CoinGlass liquidation map: {e}\")\n            return None\n    \n    async def get_coingecko_market_data(self, symbol: str = 'bitcoin') -> Optional[Dict]:\n        \"\"\"Get market data from CoinGecko (free tier)\"\"\"\n        try:\n            cache_key = f\"coingecko_market_{symbol}\"\n            if self._is_cache_valid(cache_key, 'market_data'):\n                return self.cache[cache_key]['data']\n            \n            # Get current price and basic data\n            price_endpoint = self.api_configs[LiquidationSource.COINGECKO]['endpoints']['price']\n            price_params = {\n                'ids': symbol,\n                'vs_currencies': 'usd',\n                'include_market_cap': 'true',\n                'include_24hr_vol': 'true',\n                'include_24hr_change': 'true'\n            }\n            \n            price_data = await self._make_request(LiquidationSource.COINGECKO, price_endpoint, price_params)\n            \n            # Get derivatives data for liquidation insights\n            derivatives_endpoint = self.api_configs[LiquidationSource.COINGECKO]['endpoints']['derivatives']\n            derivatives_data = await self._make_request(LiquidationSource.COINGECKO, derivatives_endpoint)\n            \n            if price_data and derivatives_data:\n                combined_data = {\n                    'price_data': price_data,\n                    'derivatives_data': derivatives_data\n                }\n                \n                # Cache the result\n                self.cache[cache_key] = {\n                    'data': combined_data,\n                    'timestamp': time.time()\n                }\n                \n                logger.info(f\"📊 Retrieved CoinGecko market data for {symbol}\")\n                return combined_data\n            \n            return None\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get CoinGecko market data: {e}\")\n            return None\n    \n    async def get_coinmarketcap_data(self, symbol: str = 'BTC') -> Optional[Dict]:\n        \"\"\"Get market data from CoinMarketCap (free tier)\"\"\"\n        try:\n            cache_key = f\"cmc_data_{symbol}\"\n            if self._is_cache_valid(cache_key, 'market_data'):\n                return self.cache[cache_key]['data']\n            \n            endpoint = self.api_configs[LiquidationSource.COINMARKETCAP]['endpoints']['quotes']\n            params = {\n                'symbol': symbol,\n                'convert': 'USD'\n            }\n            \n            data = await self._make_request(LiquidationSource.COINMARKETCAP, endpoint, params)\n            \n            if data:\n                # Cache the result\n                self.cache[cache_key] = {\n                    'data': data,\n                    'timestamp': time.time()\n                }\n                logger.info(f\"📊 Retrieved CoinMarketCap data for {symbol}\")\n            \n            return data\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get CoinMarketCap data: {e}\")\n            return None\n    \n    def _estimate_liquidation_levels_from_oi(self, price: float, open_interest_data: Dict) -> List[LiquidationCluster]:\n        \"\"\"Estimate liquidation levels from open interest data\"\"\"\n        try:\n            clusters = []\n            \n            # Common leverage levels and their approximate liquidation distances\n            leverage_levels = [3, 5, 10, 20, 25, 50, 100]\n            \n            for leverage in leverage_levels:\n                # For long positions: liquidation occurs when price drops by ~(1/leverage)\n                long_liq_distance = 1 / leverage * 0.9  # 90% to account for margin requirements\n                long_liq_price = price * (1 - long_liq_distance)\n                \n                # For short positions: liquidation occurs when price rises by ~(1/leverage)\n                short_liq_distance = 1 / leverage * 0.9\n                short_liq_price = price * (1 + short_liq_distance)\n                \n                # Estimate volume based on leverage popularity\n                popularity_weights = {\n                    3: 0.05, 5: 0.15, 10: 0.25, 20: 0.20, \n                    25: 0.15, 50: 0.15, 100: 0.05\n                }\n                \n                base_volume = 10000000  # $10M base\n                estimated_volume = base_volume * popularity_weights.get(leverage, 0.1)\n                \n                # Create liquidation clusters\n                if long_liq_price > price * 0.5:  # Don't go below 50% of current price\n                    long_cluster = LiquidationCluster(\n                        price=long_liq_price,\n                        volume=estimated_volume,\n                        long_volume=estimated_volume,\n                        short_volume=0,\n                        exchange_count=1,\n                        confidence=0.5,  # Estimated data\n                        significance=self._calculate_significance(estimated_volume),\n                        sources=['estimated_from_oi']\n                    )\n                    clusters.append(long_cluster)\n                \n                if short_liq_price < price * 2.0:  # Don't go above 200% of current price\n                    short_cluster = LiquidationCluster(\n                        price=short_liq_price,\n                        volume=estimated_volume,\n                        long_volume=0,\n                        short_volume=estimated_volume,\n                        exchange_count=1,\n                        confidence=0.5,  # Estimated data\n                        significance=self._calculate_significance(estimated_volume),\n                        sources=['estimated_from_oi']\n                    )\n                    clusters.append(short_cluster)\n            \n            return clusters\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to estimate liquidation levels: {e}\")\n            return []\n    \n    def _calculate_significance(self, volume: float) -> str:\n        \"\"\"Calculate significance level based on volume\"\"\"\n        for level, threshold in sorted(self.significance_levels.items(), \n                                     key=lambda x: x[1], reverse=True):\n            if volume >= threshold:\n                return level\n        return 'low'\n    \n    def _cluster_liquidation_levels(self, all_levels: List[LiquidationCluster]) -> List[LiquidationCluster]:\n        \"\"\"Cluster nearby liquidation levels together\"\"\"\n        try:\n            if not all_levels:\n                return []\n            \n            # Sort by price\n            sorted_levels = sorted(all_levels, key=lambda x: x.price)\n            clustered = []\n            \n            i = 0\n            while i < len(sorted_levels):\n                current_cluster = sorted_levels[i]\n                cluster_levels = [current_cluster]\n                \n                # Find nearby levels to cluster\n                j = i + 1\n                while j < len(sorted_levels):\n                    next_level = sorted_levels[j]\n                    price_diff = abs(next_level.price - current_cluster.price) / current_cluster.price\n                    \n                    if price_diff <= self.clustering_threshold:\n                        cluster_levels.append(next_level)\n                        j += 1\n                    else:\n                        break\n                \n                # Merge cluster\n                if len(cluster_levels) > 1:\n                    merged_cluster = self._merge_clusters(cluster_levels)\n                    clustered.append(merged_cluster)\n                else:\n                    clustered.append(current_cluster)\n                \n                i = j\n            \n            return clustered\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to cluster liquidation levels: {e}\")\n            return all_levels\n    \n    def _merge_clusters(self, clusters: List[LiquidationCluster]) -> LiquidationCluster:\n        \"\"\"Merge multiple clusters into one\"\"\"\n        try:\n            # Volume-weighted average price\n            total_volume = sum(c.volume for c in clusters)\n            weighted_price = sum(c.price * c.volume for c in clusters) / total_volume if total_volume > 0 else clusters[0].price\n            \n            # Sum volumes\n            total_long_volume = sum(c.long_volume for c in clusters)\n            total_short_volume = sum(c.short_volume for c in clusters)\n            \n            # Combine sources\n            all_sources = []\n            for cluster in clusters:\n                all_sources.extend(cluster.sources)\n            unique_sources = list(set(all_sources))\n            \n            # Calculate new confidence (average weighted by volume)\n            weighted_confidence = sum(c.confidence * c.volume for c in clusters) / total_volume if total_volume > 0 else 0.5\n            \n            return LiquidationCluster(\n                price=weighted_price,\n                volume=total_volume,\n                long_volume=total_long_volume,\n                short_volume=total_short_volume,\n                exchange_count=len(unique_sources),\n                confidence=min(weighted_confidence + 0.1, 1.0),  # Boost confidence for merged data\n                significance=self._calculate_significance(total_volume),\n                sources=unique_sources\n            )\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to merge clusters: {e}\")\n            return clusters[0] if clusters else None\n    \n    async def generate_liquidation_heatmap(self, symbol: str = 'BTC') -> Optional[LiquidationHeatmap]:\n        \"\"\"Generate comprehensive liquidation heatmap for Nanpin strategy\"\"\"\n        try:\n            logger.info(f\"🔥 Generating liquidation heatmap for {symbol}\")\n            \n            # Get current price first\n            current_price = await self._get_current_price(symbol)\n            if not current_price:\n                logger.error(\"❌ Failed to get current price\")\n                return None\n            \n            # Collect liquidation data from all sources\n            all_clusters = []\n            \n            # 1. CoinGlass liquidation map (premium data)\n            coinglass_data = await self.get_coinglass_liquidation_map(symbol)\n            if coinglass_data:\n                clusters = self._parse_coinglass_data(coinglass_data, current_price)\n                all_clusters.extend(clusters)\n                logger.info(f\"   📊 CoinGlass: {len(clusters)} clusters\")\n            \n            # 2. CoinGecko derivatives data\n            coingecko_data = await self.get_coingecko_market_data('bitcoin' if symbol == 'BTC' else symbol.lower())\n            if coingecko_data:\n                clusters = self._parse_coingecko_data(coingecko_data, current_price)\n                all_clusters.extend(clusters)\n                logger.info(f\"   📊 CoinGecko: {len(clusters)} clusters\")\n            \n            # 3. Estimated levels from open interest patterns\n            estimated_clusters = self._estimate_liquidation_levels_from_oi(current_price, {})\n            all_clusters.extend(estimated_clusters)\n            logger.info(f\"   📊 Estimated: {len(estimated_clusters)} clusters\")\n            \n            # Cluster nearby levels\n            clustered_levels = self._cluster_liquidation_levels(all_clusters)\n            logger.info(f\"   🔗 Clustered into {len(clustered_levels)} levels\")\n            \n            # Filter for significant levels only\n            significant_clusters = [c for c in clustered_levels if c.volume >= self.min_cluster_volume]\n            logger.info(f\"   ⭐ {len(significant_clusters)} significant clusters\")\n            \n            # Identify support and resistance levels\n            support_levels = self._identify_support_levels(significant_clusters, current_price)\n            resistance_levels = self._identify_resistance_levels(significant_clusters, current_price)\n            \n            # Generate Nanpin opportunities\n            nanpin_opportunities = self._generate_nanpin_opportunities(significant_clusters, current_price)\n            \n            # Risk assessment\n            risk_assessment = self._assess_liquidation_risk(significant_clusters, current_price)\n            \n            heatmap = LiquidationHeatmap(\n                symbol=symbol,\n                timestamp=datetime.now(),\n                current_price=current_price,\n                clusters=significant_clusters,\n                major_support_levels=support_levels,\n                major_resistance_levels=resistance_levels,\n                nanpin_opportunities=nanpin_opportunities,\n                risk_assessment=risk_assessment\n            )\n            \n            logger.info(f\"✅ Generated liquidation heatmap with {len(significant_clusters)} clusters\")\n            logger.info(f\"   🎯 {len(nanpin_opportunities)} Nanpin opportunities identified\")\n            logger.info(f\"   ⚖️ Risk level: {risk_assessment.get('overall_risk', 'unknown')}\")\n            \n            return heatmap\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to generate liquidation heatmap: {e}\")\n            return None\n    \n    async def _get_current_price(self, symbol: str) -> Optional[float]:\n        \"\"\"Get current price from most reliable source\"\"\"\n        try:\n            # Try CoinGecko first (most reliable for price)\n            coingecko_data = await self.get_coingecko_market_data('bitcoin' if symbol == 'BTC' else symbol.lower())\n            if coingecko_data and 'price_data' in coingecko_data:\n                price_data = coingecko_data['price_data']\n                symbol_key = 'bitcoin' if symbol == 'BTC' else symbol.lower()\n                if symbol_key in price_data:\n                    return float(price_data[symbol_key]['usd'])\n            \n            # Fallback to CoinMarketCap\n            cmc_data = await self.get_coinmarketcap_data(symbol)\n            if cmc_data and 'data' in cmc_data:\n                if symbol in cmc_data['data']:\n                    return float(cmc_data['data'][symbol]['quote']['USD']['price'])\n            \n            logger.warning(f\"⚠️ Could not get current price for {symbol}\")\n            return None\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get current price: {e}\")\n            return None\n    \n    def _parse_coinglass_data(self, data: Dict, current_price: float) -> List[LiquidationCluster]:\n        \"\"\"Parse CoinGlass liquidation map data\"\"\"\n        try:\n            clusters = []\n            \n            # CoinGlass API response format may vary\n            # This is a generic parser - adjust based on actual API response\n            if 'data' in data:\n                liquidation_data = data['data']\n                \n                # Parse liquidation levels\n                if isinstance(liquidation_data, list):\n                    for item in liquidation_data:\n                        if isinstance(item, dict):\n                            price = float(item.get('price', 0))\n                            volume = float(item.get('volume', 0))\n                            \n                            if price > 0 and volume > 0:\n                                cluster = LiquidationCluster(\n                                    price=price,\n                                    volume=volume,\n                                    long_volume=volume * 0.6,  # Assume 60% long\n                                    short_volume=volume * 0.4,  # Assume 40% short\n                                    exchange_count=1,\n                                    confidence=0.8,  # High confidence for CoinGlass\n                                    significance=self._calculate_significance(volume),\n                                    sources=['coinglass']\n                                )\n                                clusters.append(cluster)\n            \n            return clusters\n            \n        except Exception as e:\n            logger.debug(f\"Failed to parse CoinGlass data: {e}\")\n            return []\n    \n    def _parse_coingecko_data(self, data: Dict, current_price: float) -> List[LiquidationCluster]:\n        \"\"\"Parse CoinGecko derivatives data for liquidation insights\"\"\"\n        try:\n            clusters = []\n            \n            if 'derivatives_data' in data:\n                derivatives = data['derivatives_data']\n                \n                # Look for futures/perpetual data\n                for item in derivatives:\n                    if isinstance(item, dict):\n                        # Extract open interest and funding rates\n                        oi_usd = item.get('open_interest_usd', 0)\n                        funding_rate = item.get('funding_rate', 0)\n                        \n                        if oi_usd > 1000000:  # Only consider significant OI\n                            # Estimate liquidation clusters based on OI\n                            estimated_clusters = self._estimate_liquidation_levels_from_oi(current_price, item)\n                            \n                            # Adjust confidence based on data quality\n                            for cluster in estimated_clusters:\n                                cluster.confidence = 0.6  # Medium confidence for estimated data\n                                cluster.sources = ['coingecko_derivatives']\n                            \n                            clusters.extend(estimated_clusters)\n            \n            return clusters\n            \n        except Exception as e:\n            logger.debug(f\"Failed to parse CoinGecko data: {e}\")\n            return []\n    \n    def _identify_support_levels(self, clusters: List[LiquidationCluster], current_price: float) -> List[float]:\n        \"\"\"Identify major support levels below current price\"\"\"\n        try:\n            support_clusters = [c for c in clusters if c.price < current_price and c.long_volume > c.short_volume]\n            \n            # Sort by volume (descending) and take top levels\n            support_clusters.sort(key=lambda x: x.volume, reverse=True)\n            \n            # Take top 5 support levels\n            support_levels = [c.price for c in support_clusters[:5]]\n            \n            return sorted(support_levels, reverse=True)  # Highest to lowest\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to identify support levels: {e}\")\n            return []\n    \n    def _identify_resistance_levels(self, clusters: List[LiquidationCluster], current_price: float) -> List[float]:\n        \"\"\"Identify major resistance levels above current price\"\"\"\n        try:\n            resistance_clusters = [c for c in clusters if c.price > current_price and c.short_volume > c.long_volume]\n            \n            # Sort by volume (descending) and take top levels\n            resistance_clusters.sort(key=lambda x: x.volume, reverse=True)\n            \n            # Take top 5 resistance levels\n            resistance_levels = [c.price for c in resistance_clusters[:5]]\n            \n            return sorted(resistance_levels)  # Lowest to highest\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to identify resistance levels: {e}\")\n            return []\n    \n    def _generate_nanpin_opportunities(self, clusters: List[LiquidationCluster], current_price: float) -> List[Dict]:\n        \"\"\"Generate Nanpin DCA opportunities based on liquidation levels\"\"\"\n        try:\n            opportunities = []\n            \n            # Focus on support levels below current price (buying opportunities)\n            support_clusters = [c for c in clusters if c.price < current_price and c.long_volume > 0]\n            \n            # Sort by proximity to current price and volume\n            support_clusters.sort(key=lambda x: (current_price - x.price, -x.volume))\n            \n            for i, cluster in enumerate(support_clusters[:10]):  # Top 10 opportunities\n                distance_pct = (current_price - cluster.price) / current_price * 100\n                \n                # Skip levels too close or too far\n                if distance_pct < 1 or distance_pct > 50:\n                    continue\n                \n                opportunity = {\n                    'price': cluster.price,\n                    'distance_from_current_pct': distance_pct,\n                    'liquidation_volume': cluster.long_volume,\n                    'confidence': cluster.confidence,\n                    'significance': cluster.significance,\n                    'recommended_allocation': self._calculate_allocation_size(cluster, distance_pct),\n                    'reasoning': self._generate_opportunity_reasoning(cluster, distance_pct),\n                    'urgency': self._calculate_urgency(cluster, distance_pct),\n                    'fibonacci_alignment': self._check_fibonacci_alignment(cluster.price, current_price)\n                }\n                \n                opportunities.append(opportunity)\n            \n            # Sort by recommended allocation (highest first)\n            opportunities.sort(key=lambda x: x['recommended_allocation'], reverse=True)\n            \n            return opportunities[:5]  # Return top 5 opportunities\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to generate Nanpin opportunities: {e}\")\n            return []\n    \n    def _calculate_allocation_size(self, cluster: LiquidationCluster, distance_pct: float) -> float:\n        \"\"\"Calculate recommended allocation size for this opportunity\"\"\"\n        try:\n            # Base allocation based on liquidation volume significance\n            significance_multipliers = {\n                'low': 1.0,\n                'medium': 2.0,\n                'high': 3.0,\n                'critical': 5.0\n            }\n            \n            base_allocation = significance_multipliers.get(cluster.significance, 1.0)\n            \n            # Adjust based on distance (further = larger allocation)\n            distance_multiplier = min(distance_pct / 10, 3.0)  # Cap at 3x\n            \n            # Adjust based on confidence\n            confidence_multiplier = cluster.confidence\n            \n            total_allocation = base_allocation * distance_multiplier * confidence_multiplier\n            \n            return round(total_allocation, 1)\n            \n        except Exception:\n            return 1.0\n    \n    def _generate_opportunity_reasoning(self, cluster: LiquidationCluster, distance_pct: float) -> str:\n        \"\"\"Generate human-readable reasoning for opportunity\"\"\"\n        try:\n            reasoning_parts = []\n            \n            # Distance reasoning\n            if distance_pct < 5:\n                reasoning_parts.append(\"nearby liquidation support\")\n            elif distance_pct < 15:\n                reasoning_parts.append(\"medium-term liquidation target\")\n            else:\n                reasoning_parts.append(\"long-term liquidation accumulation zone\")\n            \n            # Volume significance\n            reasoning_parts.append(f\"{cluster.significance} liquidation volume\")\n            \n            # Confidence level\n            if cluster.confidence > 0.8:\n                reasoning_parts.append(\"high data confidence\")\n            elif cluster.confidence > 0.6:\n                reasoning_parts.append(\"moderate data confidence\")\n            \n            # Sources\n            if len(cluster.sources) > 1:\n                reasoning_parts.append(\"multiple source confirmation\")\n            \n            return \", \".join(reasoning_parts)\n            \n        except Exception:\n            return \"liquidation-based opportunity\"\n    \n    def _calculate_urgency(self, cluster: LiquidationCluster, distance_pct: float) -> str:\n        \"\"\"Calculate urgency level for opportunity\"\"\"\n        try:\n            # Closer levels have higher urgency\n            if distance_pct < 5:\n                return \"HIGH\"\n            elif distance_pct < 15:\n                return \"MEDIUM\"\n            else:\n                return \"LOW\"\n                \n        except Exception:\n            return \"MEDIUM\"\n    \n    def _check_fibonacci_alignment(self, target_price: float, current_price: float) -> Optional[str]:\n        \"\"\"Check if liquidation level aligns with Fibonacci retracements\"\"\"\n        try:\n            # Simple Fibonacci level check\n            retracement = (current_price - target_price) / current_price\n            \n            fib_levels = {\n                0.236: \"23.6%\",\n                0.382: \"38.2%\",\n                0.500: \"50.0%\",\n                0.618: \"61.8%\",\n                0.786: \"78.6%\"\n            }\n            \n            # Check for alignment within 2%\n            for fib_ratio, fib_name in fib_levels.items():\n                if abs(retracement - fib_ratio) < 0.02:\n                    return fib_name\n            \n            return None\n            \n        except Exception:\n            return None\n    \n    def _assess_liquidation_risk(self, clusters: List[LiquidationCluster], current_price: float) -> Dict:\n        \"\"\"Assess overall liquidation risk\"\"\"\n        try:\n            # Find nearest liquidation clusters\n            nearby_clusters = [c for c in clusters if abs(c.price - current_price) / current_price < 0.1]  # Within 10%\n            \n            # Calculate risk metrics\n            total_nearby_volume = sum(c.volume for c in nearby_clusters)\n            nearest_support = max([c.price for c in clusters if c.price < current_price], default=0)\n            nearest_resistance = min([c.price for c in clusters if c.price > current_price], default=float('inf'))\n            \n            # Risk assessment\n            if total_nearby_volume > 50000000:  # $50M+ nearby\n                risk_level = \"HIGH\"\n            elif total_nearby_volume > 20000000:  # $20M+ nearby\n                risk_level = \"MEDIUM\"\n            else:\n                risk_level = \"LOW\"\n            \n            support_distance = (current_price - nearest_support) / current_price * 100 if nearest_support > 0 else 100\n            resistance_distance = (nearest_resistance - current_price) / current_price * 100 if nearest_resistance != float('inf') else 100\n            \n            return {\n                'overall_risk': risk_level,\n                'nearby_liquidation_volume': total_nearby_volume,\n                'nearest_support_distance_pct': support_distance,\n                'nearest_resistance_distance_pct': resistance_distance,\n                'total_clusters_analyzed': len(clusters),\n                'recommendation': self._get_risk_recommendation(risk_level, support_distance)\n            }\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to assess liquidation risk: {e}\")\n            return {\n                'overall_risk': 'UNKNOWN',\n                'error': str(e),\n                'recommendation': 'Manual analysis required'\n            }\n    \n    def _get_risk_recommendation(self, risk_level: str, support_distance: float) -> str:\n        \"\"\"Get risk-based recommendation\"\"\"\n        if risk_level == \"HIGH\":\n            return \"Exercise caution - high liquidation activity nearby\"\n        elif risk_level == \"MEDIUM\":\n            return \"Monitor closely - moderate liquidation risk\"\n        elif support_distance < 5:\n            return \"Strong support nearby - good entry zone\"\n        else:\n            return \"Normal market conditions - proceed with strategy\"\n    \n    def __str__(self) -> str:\n        \"\"\"String representation\"\"\"\n        return f\"LiquidationAggregator(sources={len(self.api_configs)}, cache_entries={len(self.cache)})\"\n    \n    def __repr__(self) -> str:\n        \"\"\"Detailed representation\"\"\"\n        return (f\"LiquidationAggregator(\"\n                f\"sources={list(self.api_configs.keys())}, \"\n                f\"cache_entries={len(self.cache)}, \"\n                f\"min_cluster_volume={self.min_cluster_volume})\")\n\n\n# ===========================\n# TESTING AND EXAMPLES\n# ===========================\n\nasync def test_liquidation_aggregator():\n    \"\"\"Test the liquidation aggregator\"\"\"\n    print(\"🧪 Testing Liquidation Aggregator\")\n    print(\"=\" * 50)\n    \n    # Initialize aggregator\n    config = {\n        'api_keys': {\n            'coinglass': '3ec7b948900e4bd2a407a26fd4c52135',  # Example key\n            # Add other API keys as needed\n        }\n    }\n    \n    async with LiquidationAggregator(config) as aggregator:\n        print(\"🌸 Liquidation Aggregator initialized\")\n        \n        # Generate heatmap\n        print(\"\\n🔥 Generating liquidation heatmap for BTC...\")\n        heatmap = await aggregator.generate_liquidation_heatmap('BTC')\n        \n        if heatmap:\n            print(f\"✅ Heatmap generated successfully\")\n            print(f\"   Current Price: ${heatmap.current_price:,.2f}\")\n            print(f\"   Liquidation Clusters: {len(heatmap.clusters)}\")\n            print(f\"   Support Levels: {len(heatmap.major_support_levels)}\")\n            print(f\"   Nanpin Opportunities: {len(heatmap.nanpin_opportunities)}\")\n            print(f\"   Risk Level: {heatmap.risk_assessment.get('overall_risk', 'unknown')}\")\n            \n            # Show top opportunities\n            if heatmap.nanpin_opportunities:\n                print(\"\\n🎯 Top Nanpin Opportunities:\")\n                for i, opp in enumerate(heatmap.nanpin_opportunities[:3], 1):\n                    print(f\"   {i}. ${opp['price']:,.2f} (-{opp['distance_from_current_pct']:.1f}%)\")\n                    print(f\"      Allocation: {opp['recommended_allocation']}x\")\n                    print(f\"      Reasoning: {opp['reasoning']}\")\n                    if opp['fibonacci_alignment']:\n                        print(f\"      Fibonacci: {opp['fibonacci_alignment']} level\")\n        else:\n            print(\"❌ Failed to generate heatmap\")\n    \n    print(\"\\n🏁 Test completed!\")\n\n\nif __name__ == \"__main__\":\n    # Configure logging\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n    )\n    \n    # Run test\n    asyncio.run(test_liquidation_aggregator())
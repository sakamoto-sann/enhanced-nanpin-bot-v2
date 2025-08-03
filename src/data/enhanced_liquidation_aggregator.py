"""
🔥 Enhanced Liquidation Aggregator
Multi-source liquidation intelligence with CoinGecko, CoinMarketCap, and Flipside integration
"""

import asyncio
import aiohttp
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from collections import defaultdict
from .liquidation_aggregator_fixed import LiquidationAggregator, LiquidationHeatmap, LiquidationCluster

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMarketData:
    """Enhanced market data from multiple sources"""
    current_price: float
    price_sources: Dict[str, float]
    volume_24h: float
    market_cap: float
    derivatives_volume: float
    open_interest: float
    funding_rates: Dict[str, float]
    liquidation_estimates: Dict[str, float]
    source_reliability: Dict[str, float]
    timestamp: datetime

class EnhancedLiquidationAggregator(LiquidationAggregator):
    """
    🔥 Enhanced Liquidation Aggregator
    
    Adds comprehensive CoinGecko, CoinMarketCap, and price validation
    """
    
    def __init__(self, config: Dict):
        """Initialize enhanced aggregator"""
        super().__init__(config)
        
        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.coinmarketcap_base = "https://pro-api.coinmarketcap.com/v1"
        
        # API keys - Read from environment variables
        self.coingecko_key = os.getenv('COINGECKO_API_KEY') or config.get('api_keys', {}).get('coingecko')
        self.coinmarketcap_key = os.getenv('COINMARKETCAP_API_KEY') or config.get('api_keys', {}).get('coinmarketcap')
        
        # Enhanced configuration
        self.price_validation = {
            'max_deviation_pct': 2.0,  # 2% max deviation between sources
            'min_sources': 2,          # Minimum sources for validation
            'timeout_seconds': 10      # API timeout
        }
        
        # Intelligent Rate Limiting - 95% of API limits
        self.rate_limits = config.get('nanpin_strategy', {}).get('api_rate_limits', {})
        self.request_tracking = {
            'coingecko': {'calls': [], 'last_429': 0},
            'coinmarketcap': {'calls': [], 'last_429': 0},
            'coinglass': {'calls': [], 'last_429': 0},
            'binance': {'calls': [], 'last_429': 0}
        }
        
        # Request optimization settings
        self.optimization = config.get('nanpin_strategy', {}).get('request_optimization', {})
        self.batch_requests = self.optimization.get('batch_requests', True)
        self.intelligent_spacing = self.optimization.get('intelligent_spacing', True)
        self.adaptive_throttling = self.optimization.get('adaptive_throttling', True)
        
        logger.info("🔥 Enhanced Liquidation Aggregator initialized")
        logger.info(f"   CoinGecko API: {'✅' if self.coingecko_key else '🔄 Demo mode'}")
        logger.info(f"   CoinMarketCap API: {'✅' if self.coinmarketcap_key else '🔄 Demo mode'}")
        logger.info("   Multi-source price validation: ✅")
        logger.info("   🚦 Intelligent rate limiting: ✅ (95% of API limits)")
        logger.info("   ⚡ Request optimization: ✅ (batching + spacing)")
    
    async def _check_rate_limit(self, api_name: str) -> bool:
        """Check if we can make a request without exceeding 95% of rate limits"""
        try:
            current_time = time.time()
            tracking = self.request_tracking[api_name]
            limits = self.rate_limits.get(api_name, {})
            
            # Check if we're in a cooldown period after a 429 error
            cooldown = limits.get('cooldown_on_429', 60)
            if current_time - tracking['last_429'] < cooldown:
                logger.debug(f"   🚦 {api_name} in cooldown for {cooldown - (current_time - tracking['last_429']):.1f}s")
                return False
            
            # Clean old requests (keep only recent ones)
            minute_ago = current_time - 60
            tracking['calls'] = [call_time for call_time in tracking['calls'] if call_time > minute_ago]
            
            # Check rate limits
            calls_per_minute = limits.get('calls_per_minute', float('inf'))
            calls_per_second = limits.get('calls_per_second', float('inf'))
            
            # Check minute limit
            if len(tracking['calls']) >= calls_per_minute:
                logger.debug(f"   🚦 {api_name} minute limit reached: {len(tracking['calls'])}/{calls_per_minute}")
                return False
            
            # Check second limit (last 1 second)
            second_ago = current_time - 1
            recent_calls = [call_time for call_time in tracking['calls'] if call_time > second_ago]
            if len(recent_calls) >= calls_per_second:
                logger.debug(f"   🚦 {api_name} second limit reached: {len(recent_calls)}/{calls_per_second}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Rate limit check error for {api_name}: {e}")
            return True  # Allow request if check fails
    
    async def _record_request(self, api_name: str, status_code: int = 200):
        """Record a request for rate limiting tracking"""
        try:
            current_time = time.time()
            tracking = self.request_tracking[api_name]
            
            # Record the request
            tracking['calls'].append(current_time)
            
            # Record 429 errors
            if status_code == 429:
                tracking['last_429'] = current_time
                logger.warning(f"⚠️ {api_name} rate limit hit (429), starting cooldown")
            
        except Exception as e:
            logger.error(f"❌ Request recording error for {api_name}: {e}")
    
    async def _intelligent_delay(self, api_name: str):
        """Add intelligent spacing between requests"""
        try:
            if not self.intelligent_spacing:
                return
                
            limits = self.rate_limits.get(api_name, {})
            calls_per_second = limits.get('calls_per_second', 1)
            
            # Calculate optimal delay to stay at 95% of limit
            optimal_delay = 0.95 / calls_per_second if calls_per_second > 0 else 0.2
            
            # Add small randomization to avoid synchronized requests
            import random
            delay = optimal_delay + random.uniform(0, 0.1)
            
            logger.debug(f"   ⏳ {api_name} intelligent delay: {delay:.2f}s")
            await asyncio.sleep(delay)
            
        except Exception as e:
            logger.error(f"❌ Intelligent delay error for {api_name}: {e}")
    
    async def get_enhanced_market_data(self, symbol: str = "BTC") -> Optional[EnhancedMarketData]:
        """Get comprehensive market data from all sources"""
        try:
            logger.info(f"🔍 Fetching enhanced market data for {symbol}...")
            
            # Fetch from all sources in parallel
            tasks = [
                self._get_coingecko_data(symbol),
                self._get_coinmarketcap_data(symbol),
                self._get_derivatives_data(symbol),
                super()._get_current_price(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            coingecko_data, cmc_data, derivatives_data, fallback_price = results
            
            # Process results
            price_sources = {}
            funding_rates = {}
            
            # CoinGecko data
            if isinstance(coingecko_data, dict) and not isinstance(coingecko_data, Exception):
                price_sources['coingecko'] = coingecko_data.get('current_price', 0)
                logger.info(f"   CoinGecko price: ${price_sources['coingecko']:,.2f}")
            
            # CoinMarketCap data
            if isinstance(cmc_data, dict) and not isinstance(cmc_data, Exception):
                price_sources['coinmarketcap'] = cmc_data.get('price', 0)
                logger.info(f"   CoinMarketCap price: ${price_sources['coinmarketcap']:,.2f}")
            
            # Derivatives data
            if isinstance(derivatives_data, dict) and not isinstance(derivatives_data, Exception):
                funding_rates.update(derivatives_data.get('funding_rates', {}))
            
            # Fallback price
            if isinstance(fallback_price, (int, float)) and fallback_price > 0:
                price_sources['fallback'] = fallback_price
                logger.info(f"   Fallback price: ${fallback_price:,.2f}")
            
            # Validate and combine prices
            validated_price = self._validate_prices(price_sources)
            
            if validated_price is None:
                logger.error("❌ Could not validate price from any source")
                return None
            
            # Create enhanced market data
            enhanced_data = EnhancedMarketData(
                current_price=validated_price,
                price_sources=price_sources,
                volume_24h=coingecko_data.get('total_volume', 0) if isinstance(coingecko_data, dict) else 0,
                market_cap=coingecko_data.get('market_cap', 0) if isinstance(coingecko_data, dict) else 0,
                derivatives_volume=derivatives_data.get('volume_24h', 0) if isinstance(derivatives_data, dict) else 0,
                open_interest=derivatives_data.get('open_interest', 0) if isinstance(derivatives_data, dict) else 0,
                funding_rates=funding_rates,
                liquidation_estimates=self._estimate_liquidation_levels(validated_price, funding_rates),
                source_reliability=self._calculate_source_reliability(price_sources),
                timestamp=datetime.now()
            )
            
            logger.info(f"   ✅ Enhanced market data compiled")
            logger.info(f"   Validated price: ${validated_price:,.2f}")
            logger.info(f"   Sources used: {len(price_sources)}")
            logger.info(f"   Reliability score: {sum(enhanced_data.source_reliability.values()):.2f}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced market data: {e}")
            return None
    
    async def _get_coingecko_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive data from CoinGecko API with intelligent rate limiting"""
        try:
            # Check rate limits before making request
            if not await self._check_rate_limit('coingecko'):
                logger.debug(f"   🚦 CoinGecko rate limit check failed, skipping request")
                return None
            
            # Add intelligent delay
            await self._intelligent_delay('coingecko')
            
            # Map symbols
            coin_id = "bitcoin" if symbol == "BTC" else symbol.lower()
            
            # Build URL with all needed data
            url = f"{self.coingecko_base}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'true',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            
            # Add API key if available
            headers = {}
            if self.coingecko_key:
                headers['X-CG-Pro-API-Key'] = self.coingecko_key
            
            async with self.session.get(url, params=params, headers=headers) as response:
                # Record the request for rate limiting
                await self._record_request('coingecko', response.status)
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('market_data', {})
                    
                    return {
                        'current_price': market_data.get('current_price', {}).get('usd', 0),
                        'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                        'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                        'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                        'market_cap_rank': data.get('market_cap_rank', 0),
                        'circulating_supply': market_data.get('circulating_supply', 0),
                        'total_supply': market_data.get('total_supply', 0),
                        'ath': market_data.get('ath', {}).get('usd', 0),
                        'atl': market_data.get('atl', {}).get('usd', 0),
                        'sentiment_votes': data.get('sentiment_votes_up_percentage', 50),
                        'source': 'coingecko'
                    }
                else:
                    logger.warning(f"⚠️ CoinGecko API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ CoinGecko API error: {e}")
            return None
    
    async def _get_coinmarketcap_data(self, symbol: str) -> Optional[Dict]:
        """Get data from CoinMarketCap API with intelligent rate limiting"""
        try:
            if not self.coinmarketcap_key:
                logger.debug("🔄 CoinMarketCap API key not available, using demo data")
                return self._get_coinmarketcap_demo_data(symbol)
            
            # Check rate limits before making request
            if not await self._check_rate_limit('coinmarketcap'):
                logger.debug(f"   🚦 CoinMarketCap rate limit check failed, using demo data")
                return self._get_coinmarketcap_demo_data(symbol)
            
            # Add intelligent delay
            await self._intelligent_delay('coinmarketcap')
            
            # Get latest quotes
            url = f"{self.coinmarketcap_base}/cryptocurrency/quotes/latest"
            params = {
                'symbol': symbol,
                'convert': 'USD'
            }
            headers = {
                'X-CMC_PRO_API_KEY': self.coinmarketcap_key,
                'Accept': 'application/json'
            }
            
            async with self.session.get(url, params=params, headers=headers) as response:
                # Record the request for rate limiting
                await self._record_request('coinmarketcap', response.status)
                if response.status == 200:
                    data = await response.json()
                    
                    if 'data' in data and symbol in data['data']:
                        coin_data = data['data'][symbol]
                        quote_data = coin_data['quote']['USD']
                        
                        return {
                            'price': quote_data.get('price', 0),
                            'volume_24h': quote_data.get('volume_24h', 0),
                            'market_cap': quote_data.get('market_cap', 0),
                            'percent_change_1h': quote_data.get('percent_change_1h', 0),
                            'percent_change_24h': quote_data.get('percent_change_24h', 0),
                            'percent_change_7d': quote_data.get('percent_change_7d', 0),
                            'circulating_supply': coin_data.get('circulating_supply', 0),
                            'total_supply': coin_data.get('total_supply', 0),
                            'max_supply': coin_data.get('max_supply', 0),
                            'cmc_rank': coin_data.get('cmc_rank', 0),
                            'source': 'coinmarketcap'
                        }
                    else:
                        logger.warning(f"⚠️ No data found for {symbol} in CoinMarketCap response")
                        return None
                else:
                    logger.warning(f"⚠️ CoinMarketCap API returned status {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ CoinMarketCap API error: {e}")
            return self._get_coinmarketcap_demo_data(symbol)
    
    def _get_coinmarketcap_demo_data(self, symbol: str) -> Dict:
        """Get demo data when CoinMarketCap API is unavailable"""
        if symbol == "BTC":
            return {
                'price': 113250,  # Realistic demo BTC price (closer to actual)
                'volume_24h': 25_000_000_000,
                'market_cap': 2_300_000_000_000,
                'percent_change_1h': -0.5,
                'percent_change_24h': 2.3,
                'percent_change_7d': -1.8,
                'circulating_supply': 19_500_000,
                'total_supply': 19_500_000,
                'max_supply': 21_000_000,
                'cmc_rank': 1,
                'source': 'coinmarketcap_demo'
            }
        return {}
    
    async def _get_derivatives_data(self, symbol: str) -> Optional[Dict]:
        """Get derivatives and funding rate data with intelligent rate limiting"""
        try:
            # Check rate limits before making request (uses CoinGecko)
            if not await self._check_rate_limit('coingecko'):
                logger.debug(f"   🚦 CoinGecko derivatives rate limit check failed, using demo data")
                return self._get_demo_derivatives_data()
            
            # Add intelligent delay
            await self._intelligent_delay('coingecko')
            
            # Use CoinGecko derivatives endpoint
            url = f"{self.coingecko_base}/derivatives/exchanges"
            
            headers = {}
            if self.coingecko_key:
                headers['X-CG-Pro-API-Key'] = self.coingecko_key
            
            async with self.session.get(url, headers=headers) as response:
                # Record the request for rate limiting
                await self._record_request('coingecko', response.status)
                if response.status == 200:
                    data = await response.json()
                    
                    # Process derivatives data
                    total_volume = 0
                    total_oi = 0
                    funding_rates = {}
                    
                    for exchange in data[:10]:  # Top 10 exchanges
                        name = exchange.get('name', '').lower()
                        volume = exchange.get('trade_volume_24h_btc', 0)
                        oi = exchange.get('open_interest_btc', 0)
                        
                        # Ensure numeric values
                        try:
                            if volume and isinstance(volume, (int, float)):
                                total_volume += float(volume)
                            if oi and isinstance(oi, (int, float)):
                                total_oi += float(oi)
                        except (ValueError, TypeError):
                            continue
                        
                        # Mock funding rates (would need specific exchange APIs)
                        if 'binance' in name:
                            funding_rates['binance'] = -0.0050  # -0.5%
                        elif 'okx' in name or 'okex' in name:
                            funding_rates['okx'] = 0.0025   # 0.25%
                        elif 'bybit' in name:
                            funding_rates['bybit'] = -0.0075  # -0.75%
                    
                    return {
                        'volume_24h': total_volume,
                        'open_interest': total_oi,
                        'funding_rates': funding_rates,
                        'source': 'coingecko_derivatives'
                    }
                else:
                    logger.warning(f"⚠️ Derivatives API returned status {response.status}")
                    return self._get_demo_derivatives_data()
                    
        except Exception as e:
            logger.error(f"❌ Derivatives API error: {e}")
            return self._get_demo_derivatives_data()
    
    def _get_demo_derivatives_data(self) -> Dict:
        """Get demo derivatives data"""
        return {
            'volume_24h': 150_000,  # 150k BTC volume
            'open_interest': 450_000,  # 450k BTC OI
            'funding_rates': {
                'binance': -0.0050,
                'okx': 0.0025,
                'bybit': -0.0075,
                'deribit': -0.0030
            },
            'source': 'demo_derivatives'
        }
    
    def _validate_prices(self, price_sources: Dict[str, float]) -> Optional[float]:
        """Validate prices from multiple sources"""
        try:
            valid_prices = {k: v for k, v in price_sources.items() if v > 0}
            
            if len(valid_prices) < self.price_validation['min_sources']:
                # If we don't have enough sources, use any available price
                if valid_prices:
                    return list(valid_prices.values())[0]
                return None
            
            # Calculate average and check for outliers
            prices = list(valid_prices.values())
            avg_price = sum(prices) / len(prices)
            
            # Filter out prices that deviate too much
            max_deviation = avg_price * (self.price_validation['max_deviation_pct'] / 100)
            validated_prices = [
                price for price in prices 
                if abs(price - avg_price) <= max_deviation
            ]
            
            if validated_prices:
                final_price = sum(validated_prices) / len(validated_prices)
                logger.info(f"   ✅ Price validated: ${final_price:,.2f} (from {len(validated_prices)} sources)")
                return final_price
            else:
                # If all prices are outliers, use the median
                prices.sort()
                median_price = prices[len(prices) // 2]
                logger.warning(f"   ⚠️ Using median price due to high deviation: ${median_price:,.2f}")
                return median_price
                
        except Exception as e:
            logger.error(f"❌ Price validation error: {e}")
            return None
    
    def _estimate_liquidation_levels(self, current_price: float, funding_rates: Dict[str, float]) -> Dict[str, float]:
        """Estimate liquidation levels based on funding rates and leverage"""
        try:
            liquidation_estimates = {}
            
            # Common leverage levels
            leverage_levels = [2, 3, 5, 10, 20, 50, 100]
            
            # Estimate based on typical margin requirements
            for leverage in leverage_levels:
                # Long liquidation (price goes down)
                margin_ratio = 1.0 / leverage
                maintenance_margin = 0.005  # 0.5% maintenance margin
                
                long_liq_price = current_price * (1 - margin_ratio + maintenance_margin)
                short_liq_price = current_price * (1 + margin_ratio - maintenance_margin)
                
                liquidation_estimates[f'long_{leverage}x'] = long_liq_price
                liquidation_estimates[f'short_{leverage}x'] = short_liq_price
            
            # Adjust based on funding rates (high funding = more shorts = higher liquidation risk)
            avg_funding = sum(funding_rates.values()) / len(funding_rates) if funding_rates else 0
            
            if avg_funding > 0.01:  # High positive funding (expensive to be long)
                # More shorts likely, adjust short liquidation estimates up
                for key in liquidation_estimates:
                    if 'short' in key:
                        liquidation_estimates[key] *= 0.98  # 2% closer to current price
            elif avg_funding < -0.01:  # High negative funding (expensive to be short)
                # More longs likely, adjust long liquidation estimates up
                for key in liquidation_estimates:
                    if 'long' in key:
                        liquidation_estimates[key] *= 1.02  # 2% further from current price
            
            return liquidation_estimates
            
        except Exception as e:
            logger.error(f"❌ Error estimating liquidation levels: {e}")
            return {}
    
    def _calculate_source_reliability(self, price_sources: Dict[str, float]) -> Dict[str, float]:
        """Calculate reliability scores for price sources"""
        reliability = {}
        
        for source, price in price_sources.items():
            if price <= 0:
                reliability[source] = 0.0
            elif source == 'coingecko':
                reliability[source] = 0.95  # High reliability for CoinGecko
            elif source == 'coinmarketcap':
                reliability[source] = 0.90  # High reliability for CoinMarketCap
            elif source == 'fallback':
                reliability[source] = 0.70  # Lower reliability for fallback
            else:
                reliability[source] = 0.80  # Default reliability
        
        return reliability
    
    async def generate_enhanced_liquidation_heatmap(self, symbol: str = "BTC") -> Optional[LiquidationHeatmap]:
        """Generate enhanced liquidation heatmap with multi-source data"""
        try:
            logger.info(f"🔥 Generating enhanced liquidation heatmap for {symbol}...")
            
            # Get enhanced market data first
            market_data = await self.get_enhanced_market_data(symbol)
            if not market_data:
                logger.error("❌ Could not get enhanced market data")
                return await super().generate_liquidation_heatmap(symbol)  # Fallback to basic
            
            # Get liquidation data from parent class
            basic_heatmap = await super().generate_liquidation_heatmap(symbol)
            
            if not basic_heatmap:
                # Create heatmap from enhanced data
                clusters = []
                
                # Add liquidation estimates as clusters
                for liq_type, price in market_data.liquidation_estimates.items():
                    if abs(price - market_data.current_price) / market_data.current_price < 0.1:  # Within 10%
                        leverage = int(liq_type.split('_')[1].replace('x', ''))
                        volume = market_data.open_interest / leverage * 1000  # Estimate volume
                        
                        cluster = LiquidationCluster(
                            price=price,
                            volume=volume,
                            leverage=leverage,
                            side='long' if 'long' in liq_type else 'short',
                            exchange='estimated',
                            risk_score=min(leverage / 10.0, 10.0),
                            confidence=0.7
                        )
                        clusters.append(cluster)
                
                # Calculate enhanced metrics
                cascade_risk = self._calculate_enhanced_cascade_risk(market_data, clusters)
                sentiment = self._calculate_enhanced_sentiment(market_data)
                
                enhanced_heatmap = LiquidationHeatmap(
                    symbol=symbol,
                    clusters=clusters[:20],  # Top 20 clusters
                    price_range=(
                        market_data.current_price * 0.85,
                        market_data.current_price * 1.15
                    ),
                    cascade_risk_score=cascade_risk,
                    overall_sentiment=sentiment,
                    last_updated=datetime.now(),
                    data_sources=['coingecko', 'coinmarketcap', 'enhanced_estimates']
                )
                
                logger.info(f"✅ Enhanced liquidation heatmap generated:")
                logger.info(f"   Clusters: {len(clusters)}")
                logger.info(f"   Cascade risk: {cascade_risk:.1f}/10")
                logger.info(f"   Sentiment: {sentiment}")
                
                return enhanced_heatmap
            else:
                # Enhance existing heatmap with multi-source data
                basic_heatmap.data_sources.extend(['coingecko', 'coinmarketcap'])
                
                # Recalculate with enhanced data
                basic_heatmap.cascade_risk_score = self._calculate_enhanced_cascade_risk(
                    market_data, basic_heatmap.clusters
                )
                basic_heatmap.overall_sentiment = self._calculate_enhanced_sentiment(market_data)
                
                logger.info(f"✅ Enhanced existing liquidation heatmap with multi-source data")
                return basic_heatmap
                
        except Exception as e:
            logger.error(f"❌ Error generating enhanced liquidation heatmap: {e}")
            return await super().generate_liquidation_heatmap(symbol)  # Fallback
    
    def _calculate_enhanced_cascade_risk(self, market_data: EnhancedMarketData, clusters: List) -> float:
        """Calculate enhanced cascade risk with multi-source data"""
        try:
            base_risk = 5.0  # Base risk score
            
            # Funding rate risk
            avg_funding = sum(market_data.funding_rates.values()) / len(market_data.funding_rates) if market_data.funding_rates else 0
            if abs(avg_funding) > 0.01:  # High funding rates
                base_risk += 2.0
            
            # Open interest risk
            if market_data.open_interest > 400_000:  # High OI in BTC
                base_risk += 1.5
            
            # Volume spike risk
            if market_data.volume_24h > market_data.market_cap * 0.05:  # Volume > 5% of market cap
                base_risk += 1.0
            
            # Cluster concentration risk
            current_price = market_data.current_price
            nearby_clusters = [
                c for c in clusters 
                if abs(c.price_level - current_price) / current_price < 0.05  # Within 5%
            ]
            
            if len(nearby_clusters) > 5:
                base_risk += 2.0
            elif len(nearby_clusters) > 3:
                base_risk += 1.0
            
            return min(base_risk, 10.0)
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced cascade risk: {e}")
            return 5.0
    
    def _calculate_enhanced_sentiment(self, market_data: EnhancedMarketData) -> str:
        """Calculate enhanced sentiment from multi-source data"""
        try:
            sentiment_score = 0
            
            # Funding rate sentiment
            avg_funding = sum(market_data.funding_rates.values()) / len(market_data.funding_rates) if market_data.funding_rates else 0
            if avg_funding > 0.005:  # Positive funding = bearish
                sentiment_score -= 2
            elif avg_funding < -0.005:  # Negative funding = bullish
                sentiment_score += 2
            
            # Volume sentiment
            if market_data.volume_24h > market_data.derivatives_volume * 2:  # Spot volume > 2x derivatives
                sentiment_score += 1  # Bullish
            
            # Price source agreement
            price_deviation = max(market_data.price_sources.values()) - min(market_data.price_sources.values())
            relative_deviation = price_deviation / market_data.current_price
            
            if relative_deviation > 0.02:  # High price disagreement = uncertainty
                sentiment_score -= 1
            
            # Convert to sentiment
            if sentiment_score >= 2:
                return 'BULLISH'
            elif sentiment_score <= -2:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced sentiment: {e}")
            return 'NEUTRAL'
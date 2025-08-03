#!/usr/bin/env python3
"""
🌸 Enhanced Macro Analyzer for Nanpin Strategy
Integrates FRED Federal Reserve data + Polymarket prediction markets
for macro-informed permanent DCA position scaling
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import aiohttp
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class MacroRegime(Enum):
    """Macro economic regime classification"""
    EXPANSION = "expansion"        # Growth + Low volatility
    RECESSION = "recession"        # Contraction + High fear
    CRISIS = "crisis"             # Extreme fear + Liquidity crunch
    RECOVERY = "recovery"         # Post-crisis growth resumption
    STAGFLATION = "stagflation"   # High inflation + Low growth
    BUBBLE = "bubble"             # Excessive optimism + Asset inflation

@dataclass
class MacroIndicator:
    """Single macro economic indicator"""
    name: str
    value: float
    previous_value: float
    change_pct: float
    percentile: float  # Historical percentile (0-100)
    signal: str       # 'bullish', 'bearish', 'neutral'
    weight: float     # Importance weight (0-1)
    last_updated: datetime
    source: str       # 'fred', 'polymarket', 'calculated'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'previous_value': self.previous_value,
            'change_pct': self.change_pct,
            'percentile': self.percentile,
            'signal': self.signal,
            'weight': self.weight,
            'last_updated': self.last_updated.isoformat(),
            'source': self.source
        }

@dataclass
class MacroAnalysis:
    """Complete macro analysis result"""
    regime: MacroRegime
    regime_confidence: float
    overall_signal: str
    fear_greed_index: float  # 0 = extreme fear, 100 = extreme greed
    bitcoin_sentiment: float  # 0 = very bearish, 100 = very bullish
    position_scaling_factor: float  # Multiplier for Fibonacci position sizes
    risk_adjustment: float   # Risk multiplier (0.5 = half risk, 2.0 = double risk)
    indicators: Dict[str, MacroIndicator]
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'regime': self.regime.value,
            'regime_confidence': self.regime_confidence,
            'overall_signal': self.overall_signal,
            'fear_greed_index': self.fear_greed_index,
            'bitcoin_sentiment': self.bitcoin_sentiment,
            'position_scaling_factor': self.position_scaling_factor,
            'risk_adjustment': self.risk_adjustment,
            'indicators': {name: indicator.to_dict() for name, indicator in self.indicators.items()},
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }

class MacroAnalyzer:
    """
    🌸 Enhanced Macro Analyzer for Nanpin Strategy
    
    Features:
    - FRED Federal Reserve economic data integration
    - Polymarket prediction market sentiment analysis
    - Macro regime classification
    - Bitcoin-specific sentiment tracking
    - Position scaling recommendations
    - Risk adjustment factors
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize Enhanced Macro Analyzer
        
        Args:
            config_path: Path to macro configuration file
        """
        self.config_path = config_path or "config/macro_config.yaml"
        self.config = self._load_config()
        
        # API configurations
        self.fred_api_key = self.config['api_keys']['fred']
        self.session = None
        
        # Data storage
        self.current_analysis = None
        self.historical_data = {}
        self.last_update = None
        
        # FRED indicators to track
        self.fred_indicators = {
            'FEDFUNDS': {'name': 'Federal Funds Rate', 'weight': 0.3, 'invert': True},
            'CPILFESL': {'name': 'Core CPI', 'weight': 0.25, 'invert': True},
            'GDP': {'name': 'GDP Growth', 'weight': 0.2, 'invert': False},
            'DEXUSEU': {'name': 'DXY Dollar Index', 'weight': 0.15, 'invert': True},
            'M2SL': {'name': 'M2 Money Supply', 'weight': 0.2, 'invert': False},
            'GS10': {'name': '10Y Treasury Yield', 'weight': 0.25, 'invert': True},
            'VIXCLS': {'name': 'VIX Volatility', 'weight': 0.3, 'invert': True}
        }
        
        # Polymarket prediction categories
        self.polymarket_categories = {
            'crypto': {'weight': 0.4, 'keywords': ['bitcoin', 'crypto', 'btc']},
            'fed_policy': {'weight': 0.35, 'keywords': ['fed', 'rate', 'inflation', 'powell']},
            'recession': {'weight': 0.3, 'keywords': ['recession', 'gdp', 'unemployment']},
            'politics': {'weight': 0.25, 'keywords': ['election', 'trump', 'biden', 'politics']},
            'banking': {'weight': 0.2, 'keywords': ['bank', 'financial', 'crisis']}
        }
        
        logger.info("🌸 Enhanced Macro Analyzer initialized")
        logger.info(f"   FRED indicators: {len(self.fred_indicators)}")
        logger.info(f"   Polymarket categories: {len(self.polymarket_categories)}")
        logger.info(f"   Update interval: {self.config['update_intervals']['macro_analysis']}s")
    
    def _load_config(self) -> Dict:
        """Load macro analyzer configuration"""
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load macro config: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'api_keys': {
                'fred': '7aa42875026454682d22f3e02afff1b2'  # Using existing key
            },
            'update_intervals': {
                'macro_analysis': 1800,  # 30 minutes
                'fred_data': 3600,       # 1 hour
                'polymarket_data': 900   # 15 minutes
            },
            'thresholds': {
                'extreme_fear': 20,      # Fear & Greed below 20
                'extreme_greed': 80,     # Fear & Greed above 80
                'high_recession_risk': 60, # Recession probability > 60%
                'high_rate_cut_prob': 70  # Rate cut probability > 70%
            },
            'position_scaling': {
                'base_multiplier': 1.0,
                'crisis_multiplier': 2.5,     # 2.5x during crisis
                'recession_multiplier': 2.0,   # 2x during recession risk
                'fed_easing_multiplier': 1.5,  # 1.5x when Fed likely to cut
                'extreme_fear_multiplier': 2.0 # 2x during extreme fear
            }
        }
    
    async def initialize(self):
        """Initialize async session and validate API access"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test FRED API access
            await self._test_fred_connection()
            
            # Test Polymarket API access
            await self._test_polymarket_connection()
            
            logger.info("✅ Macro Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Macro Analyzer: {e}")
            raise
    
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    async def _test_fred_connection(self):
        """Test FRED API connection"""
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'observations' in data:
                        logger.info("✅ FRED API connection successful")
                        return True
                else:
                    raise Exception(f"FRED API returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ FRED API connection failed: {e}")
            raise
    
    async def _test_polymarket_connection(self):
        """Test Polymarket API connection"""
        try:
            # Test Gamma Markets API (free access)
            url = "https://gamma-api.polymarket.com/markets"
            params = {'limit': 1}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        logger.info("✅ Polymarket API connection successful")
                        return True
                else:
                    raise Exception(f"Polymarket API returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ Polymarket API connection failed: {e}")
            raise
    
    async def update_macro_analysis(self) -> MacroAnalysis:
        """
        Update complete macro analysis
        
        Returns:
            MacroAnalysis object with current assessment
        """
        try:
            logger.info("📊 Updating macro analysis...")
            
            # Gather data from all sources
            fred_indicators = await self._fetch_fred_indicators()
            polymarket_sentiment = await self._fetch_polymarket_sentiment()
            
            # Calculate derived indicators
            fear_greed_index = self._calculate_fear_greed_index(fred_indicators, polymarket_sentiment)
            bitcoin_sentiment = self._calculate_bitcoin_sentiment(polymarket_sentiment)
            
            # Classify macro regime
            regime, regime_confidence = self._classify_macro_regime(fred_indicators, fear_greed_index)
            
            # Generate trading signals
            overall_signal = self._generate_overall_signal(regime, fear_greed_index, bitcoin_sentiment)
            position_scaling_factor = self._calculate_position_scaling_factor(regime, fear_greed_index, polymarket_sentiment)
            risk_adjustment = self._calculate_risk_adjustment(regime, fear_greed_index)
            
            # Combine all indicators
            all_indicators = {**fred_indicators}
            for category, data in polymarket_sentiment.items():
                indicator_name = f"polymarket_{category}"
                all_indicators[indicator_name] = MacroIndicator(
                    name=f"Polymarket {category.title()}",
                    value=data['probability'],
                    previous_value=data.get('previous_probability', data['probability']),
                    change_pct=data.get('change_pct', 0.0),
                    percentile=data['probability'],  # Use probability as percentile
                    signal=data['signal'],
                    weight=self.polymarket_categories[category]['weight'],
                    last_updated=datetime.now(),
                    source='polymarket'
                )
            
            # Create comprehensive analysis
            analysis = MacroAnalysis(
                regime=regime,
                regime_confidence=regime_confidence,
                overall_signal=overall_signal,
                fear_greed_index=fear_greed_index,
                bitcoin_sentiment=bitcoin_sentiment,
                position_scaling_factor=position_scaling_factor,
                risk_adjustment=risk_adjustment,
                indicators=all_indicators,
                analysis_timestamp=datetime.now()
            )
            
            self.current_analysis = analysis
            self.last_update = datetime.now()
            
            # Log key results
            logger.info(f"✅ Macro analysis updated:")
            logger.info(f"   Regime: {regime.value.upper()} (confidence: {regime_confidence:.1%})")
            logger.info(f"   Signal: {overall_signal.upper()}")
            logger.info(f"   Fear/Greed: {fear_greed_index:.1f}")
            logger.info(f"   BTC Sentiment: {bitcoin_sentiment:.1f}")
            logger.info(f"   Position Scaling: {position_scaling_factor:.2f}x")
            logger.info(f"   Risk Adjustment: {risk_adjustment:.2f}x")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to update macro analysis: {e}")
            raise
    
    async def _fetch_fred_indicators(self) -> Dict[str, MacroIndicator]:
        """Fetch indicators from FRED API"""
        try:
            indicators = {}
            
            for series_id, config in self.fred_indicators.items():
                try:
                    # Fetch recent data (last 12 months)
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 12,
                        'sort_order': 'desc'
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            if len(observations) >= 2:
                                # Get current and previous values
                                current_obs = observations[0]
                                previous_obs = observations[1]
                                
                                current_value = float(current_obs['value']) if current_obs['value'] != '.' else 0.0
                                previous_value = float(previous_obs['value']) if previous_obs['value'] != '.' else 0.0
                                
                                # Calculate change
                                change_pct = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0.0
                                
                                # Calculate historical percentile (simplified)
                                all_values = [float(obs['value']) for obs in observations if obs['value'] != '.']
                                percentile = (sorted(all_values).index(current_value) / len(all_values) * 100) if all_values else 50.0
                                
                                # Determine signal based on whether indicator is inverted
                                if config['invert']:
                                    signal = 'bullish' if current_value < previous_value else 'bearish'
                                else:
                                    signal = 'bullish' if current_value > previous_value else 'bearish'
                                
                                if abs(change_pct) < 1.0:  # Less than 1% change
                                    signal = 'neutral'
                                
                                indicator = MacroIndicator(
                                    name=config['name'],
                                    value=current_value,
                                    previous_value=previous_value,
                                    change_pct=change_pct,
                                    percentile=percentile,
                                    signal=signal,
                                    weight=config['weight'],
                                    last_updated=datetime.now(),
                                    source='fred'
                                )
                                
                                indicators[series_id] = indicator
                                logger.debug(f"   📈 {config['name']}: {current_value:.2f} ({signal})")
                        
                        await asyncio.sleep(0.1)  # Rate limiting
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {series_id}: {e}")
                    continue
            
            logger.info(f"📊 Fetched {len(indicators)} FRED indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch FRED indicators: {e}")
            return {}
    
    async def _fetch_polymarket_sentiment(self) -> Dict[str, Dict]:
        """Fetch sentiment data from Polymarket prediction markets"""
        try:
            sentiment_data = {}
            
            # Fetch markets from Gamma API
            url = "https://gamma-api.polymarket.com/markets"
            params = {
                'limit': 100,
                'closed': 'false'  # Only active markets
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Polymarket API returned status {response.status}")
                    return {}
                
                markets = await response.json()
                
                if not isinstance(markets, list):
                    logger.warning("Unexpected Polymarket API response format")
                    return {}
            
            # Analyze markets by category
            for category, config in self.polymarket_categories.items():
                relevant_markets = []
                
                for market in markets:
                    question = market.get('question', '').lower()
                    description = market.get('description', '').lower()
                    
                    # Check if market matches category keywords
                    for keyword in config['keywords']:
                        if keyword.lower() in question or keyword.lower() in description:
                            relevant_markets.append(market)
                            break
                
                if relevant_markets:
                    # Calculate average probability for category
                    probabilities = []
                    volumes = []
                    
                    for market in relevant_markets[:10]:  # Top 10 relevant markets
                        # Get outcome probabilities
                        outcomes = market.get('outcomes', [])
                        if outcomes:
                            # Assume first outcome is "Yes" probability
                            prob = float(outcomes[0].get('price', 0.5))
                            volume = float(market.get('volume', 0))
                            
                            probabilities.append(prob)
                            volumes.append(volume)
                    
                    if probabilities:
                        # Volume-weighted average probability
                        if sum(volumes) > 0:
                            avg_probability = sum(p * v for p, v in zip(probabilities, volumes)) / sum(volumes)
                        else:
                            avg_probability = sum(probabilities) / len(probabilities)
                        
                        avg_probability *= 100  # Convert to percentage
                        
                        # Determine signal based on category
                        if category in ['crypto', 'fed_policy']:
                            signal = 'bullish' if avg_probability > 60 else 'bearish' if avg_probability < 40 else 'neutral'
                        elif category in ['recession', 'banking']:
                            signal = 'bearish' if avg_probability > 60 else 'bullish' if avg_probability < 40 else 'neutral'
                        else:
                            signal = 'neutral'
                        
                        sentiment_data[category] = {
                            'probability': avg_probability,
                            'signal': signal,
                            'market_count': len(relevant_markets),
                            'total_volume': sum(volumes)
                        }
                        
                        logger.debug(f"   🔮 {category}: {avg_probability:.1f}% ({signal})")
            
            logger.info(f"🔮 Analyzed {len(sentiment_data)} Polymarket categories")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch Polymarket sentiment: {e}")
            return {}
    
    def _calculate_fear_greed_index(self, fred_indicators: Dict, polymarket_sentiment: Dict) -> float:
        """Calculate custom Fear & Greed Index (0-100)"""
        try:
            fear_factors = 0.0
            greed_factors = 0.0
            total_weight = 0.0
            
            # FRED indicators contribution
            for series_id, indicator in fred_indicators.items():
                weight = indicator.weight
                
                if series_id == 'VIXCLS':  # VIX - high = fear
                    if indicator.value > 30:
                        fear_factors += weight * 2
                    elif indicator.value > 20:
                        fear_factors += weight
                    else:
                        greed_factors += weight
                
                elif series_id == 'FEDFUNDS':  # Fed Funds - high = restrictive
                    if indicator.value > 4:
                        fear_factors += weight
                    elif indicator.value < 2:
                        greed_factors += weight
                
                elif series_id == 'CPILFESL':  # Core CPI - high = fear of inflation
                    if indicator.change_pct > 0.5:  # Rising inflation
                        fear_factors += weight
                    else:
                        greed_factors += weight * 0.5
                
                total_weight += weight
            
            # Polymarket sentiment contribution
            if 'recession' in polymarket_sentiment:
                prob = polymarket_sentiment['recession']['probability']
                if prob > 60:
                    fear_factors += 0.3 * 2  # High recession probability = fear
                elif prob < 30:
                    greed_factors += 0.3
                total_weight += 0.3
            
            if 'banking' in polymarket_sentiment:
                prob = polymarket_sentiment['banking']['probability']
                if prob > 50:
                    fear_factors += 0.2 * 2  # Banking crisis risk = fear
                total_weight += 0.2
            
            # Calculate final index
            if total_weight > 0:
                net_sentiment = (greed_factors - fear_factors) / total_weight
                fear_greed_index = max(0, min(100, 50 + net_sentiment * 50))
            else:
                fear_greed_index = 50  # Neutral
            
            return fear_greed_index
            
        except Exception as e:
            logger.warning(f"Failed to calculate fear/greed index: {e}")
            return 50.0  # Default to neutral
    
    def _calculate_bitcoin_sentiment(self, polymarket_sentiment: Dict) -> float:
        """Calculate Bitcoin-specific sentiment (0-100)"""
        try:
            bitcoin_sentiment = 50.0  # Start neutral
            
            # Direct crypto sentiment
            if 'crypto' in polymarket_sentiment:
                crypto_prob = polymarket_sentiment['crypto']['probability']
                bitcoin_sentiment = crypto_prob
            
            # Fed policy impact (rate cuts = bullish for Bitcoin)
            if 'fed_policy' in polymarket_sentiment:
                fed_prob = polymarket_sentiment['fed_policy']['probability']
                # If high probability of rate cuts, bullish for Bitcoin
                if 'cut' in str(polymarket_sentiment['fed_policy']).lower():
                    bitcoin_sentiment = (bitcoin_sentiment + fed_prob) / 2
            
            # Recession impact (mild recession can be bullish due to QE)
            if 'recession' in polymarket_sentiment:
                recession_prob = polymarket_sentiment['recession']['probability']
                if recession_prob > 60:
                    # High recession probability -> likely QE -> bullish Bitcoin
                    bitcoin_sentiment = (bitcoin_sentiment + (100 - recession_prob)) / 2
            
            return max(0, min(100, bitcoin_sentiment))
            
        except Exception as e:
            logger.warning(f"Failed to calculate Bitcoin sentiment: {e}")
            return 50.0
    
    def _classify_macro_regime(self, fred_indicators: Dict, fear_greed_index: float) -> Tuple[MacroRegime, float]:
        """Classify current macro economic regime"""
        try:
            regime_scores = {
                MacroRegime.EXPANSION: 0.0,
                MacroRegime.RECESSION: 0.0,
                MacroRegime.CRISIS: 0.0,
                MacroRegime.RECOVERY: 0.0,
                MacroRegime.STAGFLATION: 0.0,
                MacroRegime.BUBBLE: 0.0
            }
            
            # VIX analysis
            if 'VIXCLS' in fred_indicators:
                vix = fred_indicators['VIXCLS'].value
                if vix > 40:
                    regime_scores[MacroRegime.CRISIS] += 3
                elif vix > 30:
                    regime_scores[MacroRegime.RECESSION] += 2
                elif vix < 15:
                    regime_scores[MacroRegime.BUBBLE] += 2
                    regime_scores[MacroRegime.EXPANSION] += 1
                else:
                    regime_scores[MacroRegime.EXPANSION] += 1
            
            # Fed Funds Rate analysis
            if 'FEDFUNDS' in fred_indicators:
                fed_rate = fred_indicators['FEDFUNDS'].value
                if fed_rate > 5:
                    regime_scores[MacroRegime.RECESSION] += 2
                elif fed_rate < 1:
                    regime_scores[MacroRegime.RECOVERY] += 2
                    regime_scores[MacroRegime.CRISIS] += 1
                else:
                    regime_scores[MacroRegime.EXPANSION] += 1
            
            # Inflation analysis
            if 'CPILFESL' in fred_indicators:
                cpi_change = fred_indicators['CPILFESL'].change_pct
                if cpi_change > 0.5:  # Rising inflation
                    regime_scores[MacroRegime.STAGFLATION] += 2
                    regime_scores[MacroRegime.BUBBLE] += 1
            
            # Fear/Greed contribution
            if fear_greed_index < 20:  # Extreme fear
                regime_scores[MacroRegime.CRISIS] += 2
                regime_scores[MacroRegime.RECESSION] += 1
            elif fear_greed_index > 80:  # Extreme greed
                regime_scores[MacroRegime.BUBBLE] += 2
            elif fear_greed_index > 60:
                regime_scores[MacroRegime.EXPANSION] += 1
            
            # Determine regime
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            regime = best_regime[0]
            confidence = min(1.0, best_regime[1] / 10.0)  # Normalize to 0-1
            
            return regime, confidence
            
        except Exception as e:
            logger.warning(f"Failed to classify macro regime: {e}")
            return MacroRegime.EXPANSION, 0.5
    
    def _generate_overall_signal(self, regime: MacroRegime, fear_greed_index: float, bitcoin_sentiment: float) -> str:
        """Generate overall trading signal"""
        try:
            bullish_score = 0
            bearish_score = 0
            
            # Regime contribution
            if regime in [MacroRegime.CRISIS, MacroRegime.RECESSION]:
                bullish_score += 2  # Crisis = opportunity for permanent DCA
            elif regime in [MacroRegime.RECOVERY, MacroRegime.EXPANSION]:
                bullish_score += 1
            elif regime == MacroRegime.BUBBLE:
                bearish_score += 1  # Be cautious during bubbles
            
            # Fear/Greed contribution
            if fear_greed_index < 30:  # Fear = opportunity
                bullish_score += 2
            elif fear_greed_index > 70:  # Greed = caution
                bearish_score += 1
            
            # Bitcoin sentiment contribution
            if bitcoin_sentiment > 60:
                bullish_score += 1
            elif bitcoin_sentiment < 40:
                bearish_score += 1
            
            # Determine signal
            if bullish_score > bearish_score + 1:
                return 'bullish'
            elif bearish_score > bullish_score:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.warning(f"Failed to generate overall signal: {e}")
            return 'neutral'
    
    def _calculate_position_scaling_factor(self, regime: MacroRegime, fear_greed_index: float, polymarket_sentiment: Dict) -> float:
        """Calculate position scaling factor for Fibonacci levels"""
        try:
            base_multiplier = self.config['position_scaling']['base_multiplier']
            scaling_factor = base_multiplier
            
            # Regime-based scaling
            if regime == MacroRegime.CRISIS:
                scaling_factor *= self.config['position_scaling']['crisis_multiplier']
            elif regime == MacroRegime.RECESSION:
                scaling_factor *= self.config['position_scaling']['recession_multiplier']
            elif regime == MacroRegime.RECOVERY:
                scaling_factor *= 1.2  # Modest increase during recovery
            
            # Fear/Greed scaling
            if fear_greed_index < self.config['thresholds']['extreme_fear']:
                scaling_factor *= self.config['position_scaling']['extreme_fear_multiplier']
            elif fear_greed_index < 40:
                scaling_factor *= 1.3  # Fear is opportunity
            
            # Recession probability scaling
            if 'recession' in polymarket_sentiment:
                recession_prob = polymarket_sentiment['recession']['probability']
                if recession_prob > self.config['thresholds']['high_recession_risk']:
                    scaling_factor *= 1.5  # High recession risk = opportunity
            
            # Fed policy scaling
            if 'fed_policy' in polymarket_sentiment:
                fed_prob = polymarket_sentiment['fed_policy']['probability']
                if fed_prob > self.config['thresholds']['high_rate_cut_prob']:
                    scaling_factor *= self.config['position_scaling']['fed_easing_multiplier']
            
            # Cap the scaling factor to prevent excessive leverage
            scaling_factor = min(scaling_factor, 3.0)  # Max 3x scaling
            scaling_factor = max(scaling_factor, 0.5)  # Min 0.5x scaling
            
            return scaling_factor
            
        except Exception as e:
            logger.warning(f"Failed to calculate position scaling factor: {e}")
            return 1.0
    
    def _calculate_risk_adjustment(self, regime: MacroRegime, fear_greed_index: float) -> float:
        """Calculate risk adjustment factor"""
        try:
            risk_adjustment = 1.0
            
            # Lower risk during uncertain times
            if regime == MacroRegime.CRISIS:
                risk_adjustment = 0.7  # Reduce position sizes due to high volatility
            elif regime == MacroRegime.BUBBLE:
                risk_adjustment = 0.8  # Be cautious during bubbles
            elif regime == MacroRegime.STAGFLATION:
                risk_adjustment = 0.9  # Slightly reduced risk
            
            # Extreme greed = higher risk
            if fear_greed_index > 80:
                risk_adjustment *= 0.8
            
            return risk_adjustment
            
        except Exception as e:
            logger.warning(f"Failed to calculate ri                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#!/usr/bin/env python3
"""
🌸 Polymarket Prediction Market API Client
Access prediction market sentiment for macro-informed Nanpin trading
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
import re

logger = logging.getLogger(__name__)

@dataclass
class PolymarketOutcome:
    """Single outcome in a Polymarket market"""
    token_id: str
    outcome: str
    price: float
    volume_24h: float
    
@dataclass
class PolymarketMarket:
    """Complete Polymarket prediction market"""
    id: str
    question: str
    description: str
    category: str
    end_date: datetime
    volume: float
    liquidity: float
    outcomes: List[PolymarketOutcome]
    active: bool
    resolved: bool
    
    def get_yes_probability(self) -> float:
        """Get probability of 'Yes' outcome"""
        for outcome in self.outcomes:
            if 'yes' in outcome.outcome.lower():
                return outcome.price * 100
        return 50.0  # Default neutral
    
    def get_primary_probability(self) -> float:
        """Get probability of primary/first outcome"""
        if self.outcomes:
            return self.outcomes[0].price * 100
        return 50.0

@dataclass
class PredictionSentiment:
    """Aggregated prediction market sentiment"""
    category: str
    average_probability: float
    volume_weighted_probability: float
    market_count: int
    total_volume: float
    confidence_score: float
    signal: str  # 'bullish', 'bearish', 'neutral'
    key_markets: List[str]
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'category': self.category,
            'average_probability': self.average_probability,
            'volume_weighted_probability': self.volume_weighted_probability,
            'market_count': self.market_count,
            'total_volume': self.total_volume,
            'confidence_score': self.confidence_score,
            'signal': self.signal,
            'key_markets': self.key_markets,
            'last_updated': self.last_updated.isoformat()
        }

class PolymarketClient:
    """
    🌸 Polymarket Prediction Market API Client
    
    Features:
    - Gamma Markets API integration (free access)
    - Category-based sentiment analysis
    - Bitcoin-relevant prediction tracking
    - Volume-weighted probability calculations
    - Confidence scoring based on liquidity
    - Real-time market monitoring
    """
    
    def __init__(self):
        """Initialize Polymarket Client"""
        self.gamma_base_url = "https://gamma-api.polymarket.com"
        self.session = None
        
        # Cache for market data
        self.cache = {}
        self.cache_duration = 900  # 15 minutes
        
        # Bitcoin-relevant prediction categories
        self.prediction_categories = {
            'crypto_price': {
                'keywords': ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'price', 'value'],
                'weight': 0.4,
                'bitcoin_impact': 'direct',
                'signal_interpretation': 'bullish_above_60'
            },
            'fed_policy': {
                'keywords': ['fed', 'federal reserve', 'interest rate', 'rate cut', 'rate hike', 'powell', 'monetary policy'],
                'weight': 0.35,
                'bitcoin_impact': 'inverse',  # Rate cuts bullish for Bitcoin
                'signal_interpretation': 'inverse_fed_hawkish'
            },
            'recession_risk': {
                'keywords': ['recession', 'economic downturn', 'gdp', 'unemployment', 'job', 'employment'],
                'weight': 0.3,
                'bitcoin_impact': 'complex',  # Mild recession can be bullish (QE), severe recession bearish
                'signal_interpretation': 'complex_recession'
            },
            'inflation_outlook': {
                'keywords': ['inflation', 'cpi', 'pce', 'price', 'cost'],
                'weight': 0.25,
                'bitcoin_impact': 'positive',  # Inflation bullish for Bitcoin as hedge
                'signal_interpretation': 'bullish_above_50'
            },
            'political_events': {
                'keywords': ['election', 'trump', 'biden', 'president', 'congress', 'policy'],
                'weight': 0.2,
                'bitcoin_impact': 'variable',
                'signal_interpretation': 'context_dependent'
            },
            'banking_stability': {
                'keywords': ['bank', 'banking', 'financial crisis', 'credit', 'lending'],
                'weight': 0.25,
                'bitcoin_impact': 'positive',  # Banking instability bullish for Bitcoin
                'signal_interpretation': 'bullish_crisis_risk'
            },
            'market_volatility': {
                'keywords': ['volatility', 'vix', 'market crash', 'correction', 'bear market'],
                'weight': 0.2,
                'bitcoin_impact': 'complex',
                'signal_interpretation': 'complex_volatility'
            }
        }
        
        logger.info("🌸 Polymarket Client initialized")
        logger.info(f"   Tracking {len(self.prediction_categories)} categories")
    
    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()
        
        # Test API connection
        await self._test_connection()
        logger.info("✅ Polymarket Client ready")
    
    async def close(self):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self):
        """Test Polymarket API connection"""
        try:
            url = f"{self.gamma_base_url}/markets"
            params = {'limit': 1}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        logger.info("✅ Polymarket API connection successful")
                        return True
                
                raise Exception(f"Polymarket API test failed: {response.status}")
                
        except Exception as e:
            logger.error(f"❌ Polymarket API connection failed: {e}")
            raise
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key"""
        return f"{endpoint}_{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (datetime.now().timestamp() - cache_time) < self.cache_duration
    
    async def fetch_markets(self, limit: int = 200, closed: bool = False, 
                          category: Optional[str] = None) -> List[PolymarketMarket]:
        """
        Fetch markets from Polymarket
        
        Args:
            limit: Maximum number of markets to fetch
            closed: Include closed markets
            category: Specific category filter
            
        Returns:
            List of PolymarketMarket objects
        """
        try:
            # Check cache first
            params = {'limit': limit, 'closed': str(closed).lower()}
            if category:
                params['category'] = category
                
            cache_key = self._get_cache_key('markets', params)
            
            if self._is_cache_valid(cache_key):
                logger.debug("📦 Using cached market data")
                return self.cache[cache_key]['data']
            
            # Fetch from API
            url = f"{self.gamma_base_url}/markets"
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch markets: {response.status}")
                    return []
                
                markets_data = await response.json()
                
                if not isinstance(markets_data, list):
                    logger.error("Unexpected markets API response format")
                    return []
            
            # Parse markets
            markets = []
            for market_data in markets_data:
                try:
                    market = self._parse_market(market_data)
                    if market:
                        markets.append(market)
                except Exception as e:
                    logger.debug(f"Failed to parse market {market_data.get('id', 'unknown')}: {e}")
                    continue
            
            # Cache the result
            self.cache[cache_key] = {
                'data': markets,
                'timestamp': datetime.now().timestamp()
            }
            
            logger.debug(f"📊 Fetched {len(markets)} markets from Polymarket")
            return markets
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch markets: {e}")
            return []
    
    def _parse_market(self, market_data: Dict) -> Optional[PolymarketMarket]:
        """Parse market data from API response"""
        try:
            # Extract basic info
            market_id = market_data.get('id', '')
            question = market_data.get('question', '')
            description = market_data.get('description', '')
            category = market_data.get('category', 'other')
            
            # Parse end date
            end_date_str = market_data.get('end_date_iso')
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
            else:
                end_date = datetime.now() + timedelta(days=30)  # Default
            
            # Extract market metrics
            volume = float(market_data.get('volume', 0))
            liquidity = float(market_data.get('liquidity', 0))
            active = market_data.get('active', True)
            resolved = market_data.get('resolved', False)
            
            # Parse outcomes
            outcomes = []
            outcomes_data = market_data.get('outcomes', [])
            
            for outcome_data in outcomes_data:
                try:
                    outcome = PolymarketOutcome(
                        token_id=outcome_data.get('token_id', ''),
                        outcome=outcome_data.get('outcome', ''),
                        price=float(outcome_data.get('price', 0.5)),
                        volume_24h=float(outcome_data.get('volume_24h', 0))
                    )
                    outcomes.append(outcome)
                except Exception as e:
                    logger.debug(f"Failed to parse outcome: {e}")
                    continue
            
            market = PolymarketMarket(
                id=market_id,
                question=question,
                description=description,
                category=category,
                end_date=end_date,
                volume=volume,
                liquidity=liquidity,
                outcomes=outcomes,
                active=active,
                resolved=resolved
            )
            
            return market
            
        except Exception as e:
            logger.debug(f"Failed to parse market: {e}")
            return None
    
    def categorize_markets(self, markets: List[PolymarketMarket]) -> Dict[str, List[PolymarketMarket]]:
        """
        Categorize markets based on keywords
        
        Args:
            markets: List of markets to categorize
            
        Returns:
            Dictionary mapping categories to relevant markets
        """
        try:
            categorized = {category: [] for category in self.prediction_categories.keys()}
            
            for market in markets:
                # Skip resolved or inactive markets
                if market.resolved or not market.active:
                    continue
                
                # Check each category
                for category, config in self.prediction_categories.items():
                    keywords = config['keywords']
                    
                    # Check if market matches category keywords
                    market_text = f"{market.question} {market.description}".lower()
                    
                    for keyword in keywords:
                        if keyword.lower() in market_text:
                            categorized[category].append(market)
                            break
            
            # Log categorization results
            for category, markets_list in categorized.items():
                if markets_list:
                    logger.debug(f"   {category}: {len(markets_list)} markets")
            
            return categorized
            
        except Exception as e:
            logger.error(f"❌ Failed to categorize markets: {e}")
            return {}
    
    def analyze_category_sentiment(self, category: str, markets: List[PolymarketMarket]) -> Optional[PredictionSentiment]:
        """
        Analyze sentiment for a category of markets
        
        Args:
            category: Category name
            markets: List of markets in category
            
        Returns:
            PredictionSentiment object
        """
        try:
            if not markets:
                return None
            
            config = self.prediction_categories.get(category, {})
            
            # Calculate probabilities and volumes
            probabilities = []
            volumes = []
            market_names = []
            
            for market in markets[:20]:  # Limit to top 20 markets
                prob = market.get_primary_probability()
                vol = market.volume
                
                probabilities.append(prob)
                volumes.append(vol)
                market_names.append(market.question[:50])  # Truncate long questions
            
            if not probabilities:
                return None
            
            # Calculate average and volume-weighted probabilities
            avg_probability = sum(probabilities) / len(probabilities)
            
            if sum(volumes) > 0:
                volume_weighted_prob = sum(p * v for p, v in zip(probabilities, volumes)) / sum(volumes)
            else:
                volume_weighted_prob = avg_probability
            
            # Calculate confidence score based on volume and market count
            total_volume = sum(volumes)
            confidence_score = min(1.0, (total_volume / 100000) * (len(markets) / 5))  # Normalize
            
            # Generate signal based on category interpretation
            signal = self._interpret_category_signal(category, volume_weighted_prob, config)
            
            sentiment = PredictionSentiment(
                category=category,
                average_probability=avg_probability,
                volume_weighted_probability=volume_weighted_prob,
                market_count=len(markets),
                total_volume=total_volume,
                confidence_score=confidence_score,
                signal=signal,
                key_markets=market_names[:5],  # Top 5 markets
                last_updated=datetime.now()
            )
            
            logger.debug(f"📈 {category}: {volume_weighted_prob:.1f}% ({signal})")
            return sentiment
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze category {category}: {e}")
            return None
    
    def _interpret_category_signal(self, category: str, probability: float, config: Dict) -> str:
        """Interpret probability as Bitcoin trading signal"""
        try:
            interpretation = config.get('signal_interpretation', 'neutral')
            
            if interpretation == 'bullish_above_60':
                return 'bullish' if probability > 60 else 'bearish' if probability < 40 else 'neutral'
            
            elif interpretation == 'bullish_above_50':
                return 'bullish' if probability > 50 else 'bearish' if probability < 50 else 'neutral'
            
            elif interpretation == 'inverse_fed_hawkish':
                # For Fed policy: high probability of hawkish policy = bearish for Bitcoin
                return 'bearish' if probability > 60 else 'bullish' if probability < 40 else 'neutral'
            
            elif interpretation == 'complex_recession':
                # Recession risk: moderate risk can be bullish (QE), high risk bearish
                if probability > 70:
                    return 'bearish'  # High recession risk = bearish
                elif probability > 40:
                    return 'bullish'  # Moderate recession risk = potential QE = bullish
                else:
                    return 'neutral'
            
            elif interpretation == 'bullish_crisis_risk':
                # Banking/crisis risk: higher probability = more bullish for Bitcoin as safe haven
                return 'bullish' if probability > 50 else 'neutral'
            
            elif interpretation == 'complex_volatility':
                # Market volatility: moderate volatility can be bullish, extreme volatility bearish
                if probability > 80:
                    return 'bearish'  # Extreme volatility = risk-off
                elif probability > 50:
                    return 'bullish'  # Moderate volatility = opportunity
                else:
                    return 'neutral'
            
            elif interpretation == 'context_dependent':
                # Political events require more nuanced analysis
                return 'neutral'  # Default to neutral for complex political outcomes
            
            else:
                return 'neutral'
                
        except Exception as e:
            logger.warning(f"Failed to interpret signal for {category}: {e}")
            return 'neutral'
    
    async def get_prediction_sentiment(self) -> Dict[str, PredictionSentiment]:
        """
        Get comprehensive prediction market sentiment analysis
        
        Returns:
            Dictionary mapping categories to sentiment analysis
        """
        try:
            logger.info("🔮 Fetching prediction market sentiment...")
            
            # Fetch active markets
            markets = await self.fetch_markets(limit=300, closed=False)
            
            if not markets:
                logger.warning("No markets fetched from Polymarket")
                return {}
            
            # Categorize markets
            categorized_markets = self.categorize_markets(markets)
            
            # Analyze each category
            sentiment_analysis = {}
            
            for category, category_markets in categorized_markets.items():
                if category_markets:
                    sentiment = self.analyze_category_sentiment(category, category_markets)
                    if sentiment:
                        sentiment_analysis[category] = sentiment
            
            logger.info(f"✅ Analyzed {len(sentiment_analysis)} prediction categories")
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction sentiment: {e}")
            return {}
    
    async def get_bitcoin_specific_markets(self) -> List[PolymarketMarket]:
        """Get markets specifically about Bitcoin price predictions"""
        try:
            markets = await self.fetch_markets(limit=100)
            
            bitcoin_markets = []
            bitcoin_keywords = ['bitcoin', 'btc', 'crypto']
            
            for market in markets:
                if market.resolved or not market.active:
                    continue
                
                market_text = f"{market.question} {market.description}".lower()
                
                for keyword in bitcoin_keywords:
                    if keyword in market_text and ('price' in market_text or 'value' in market_text):
                        bitcoin_markets.append(market)
                        break
            
            logger.info(f"📊 Found {len(bitcoin_markets)} Bitcoin-specific markets")
            return bitcoin_markets
            
        except Exception as e:
            logger.error(f"❌ Failed to get Bitcoin markets: {e}")
            return []
    
    def calculate_macro_sentiment_score(self, sentiment_data: Dict[str, PredictionSentiment]) -> Dict[str, float]:
        """
        Calculate overall macro sentiment score for Bitcoin
        
        Args:
            sentiment_data: Dictionary of category sentiment analysis
            
        Returns:
            Macro sentiment scores
        """
        try:
            category_scores = {}
            total_weight = 0.0
            weighted_score = 0.0
            
            for category, sentiment in sentiment_data.items():
                config = self.prediction_categories.get(category, {})
                weight = config.get('weight', 0.1)
                
                # Convert signal to numeric score
                signal_score = {'bullish': 1.0, 'neutral': 0.0, 'bearish': -1.0}.get(sentiment.signal, 0.0)
                
                # Apply confidence weighting
                confidence_weighted_score = signal_score * sentiment.confidence_score
                
                category_scores[category] = confidence_weighted_score
                
                # Add to overall weighted score
                weighted_score += confidence_weighted_score * weight
                total_weight += weight
            
            # Calculate overall score
            if total_weight > 0:
                overall_score = weighted_score / total_weight
            else:
                overall_score = 0.0
            
            return {
                'overall_sentiment_score': overall_score,
                'category_scores': category_scores,
                'confidence_weighted': True,
                'total_categories': len(sentiment_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate sentiment score: {e}")
            return {'overall_sentiment_score': 0.0}
    
    def export_sentiment_analysis(self, sentiment_data: Dict[str, PredictionSentiment]) -> Dict:
        """Export sentiment analysis to dictionary format"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'categories_analyzed': len(sentiment_data),
                'sentiment_by_category': {},
                'summary': {}
            }
            
            # Export individual categories
            for category, sentiment in sentiment_data.items():
                export_data['sentiment_by_category'][category] = sentiment.to_dict()
            
            # Calculate summary statistics
            signals = [sentiment.signal for sentiment in sentiment_data.values()]
            total_volume = sum(sentiment.total_volume for sentiment in sentiment_data.values())
            avg_confidence = np.mean([sentiment.confidence_score for sentiment in sentiment_data.values()])
            
            export_data['summary'] = {
                'total_markets_analyzed': sum(sentiment.market_count for sentiment in sentiment_data.values()),
                'total_volume': total_volume,
                'average_confidence': avg_confidence,
                'bullish_categories': signals.count('bullish'),
                'bearish_categories': signals.count('bearish'),
                'neutral_categories': signals.count('neutral'),
                'overall_bias': 'bullish' if signals.count('bullish') > signals.count('bearish') else 'bearish' if signals.count('bearish') > signals.count('bullish') else 'neutral'
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export sentiment analysis: {e}")
            return {}


# ===========================
# UTILITY FUNCTIONS
# ===========================

async def test_polymarket_client():
    """Test Polymarket Client functionality"""
    print("🧪 Testing Polymarket Client")
    print("=" * 50)
    
    # Initialize client
    client = PolymarketClient()
    
    try:
        await client.initialize()
        
        # Test market fetching
        print("\n📊 Testing market fetch...")
        markets = await client.fetch_markets(limit=50)
        print(f"   ✅ Fetched {len(markets)} markets")
        
        if markets:
            sample_market = markets[0]
            print(f"   Sample: {sample_market.question[:60]}...")
            print(f"   Volume: ${sample_market.volume:,.0f}")
            print(f"   Outcomes: {len(sample_market.outcomes)}")
        
        # Test categorization
        print("\n📊 Testing market categorization...")
        categorized = client.categorize_markets(markets)
        for category, category_markets in categorized.items():
            if category_markets:
                print(f"   {category}: {len(category_markets)} markets")
        
        # Test sentiment analysis
        print("\n📊 Testing sentiment analysis...")
        sentiment_data = await client.get_prediction_sentiment()
        print(f"   ✅ Analyzed {len(sentiment_data)} categories")
        
        for category, sentiment in sentiment_data.items():
            print(f"   {category}: {sentiment.volume_weighted_probability:.1f}% ({sentiment.signal})")
        
        # Test macro score calculation
        print("\n📊 Testing macro sentiment score...")
        macro_scores = client.calculate_macro_sentiment_score(sentiment_data)
        overall_score = macro_scores.get('overall_sentiment_score', 0)
        print(f"   Overall prediction market sentiment: {overall_score:.2f}")
        
        # Test Bitcoin-specific markets
        print("\n₿ Testing Bitcoin-specific markets...")
        btc_markets = await client.get_bitcoin_specific_markets()
        print(f"   ✅ Found {len(btc_markets)} Bitcoin markets")
        
        for market in btc_markets[:3]:  # Show first 3
            prob = market.get_primary_probability()
            print(f"   {market.question[:50]}... ({prob:.1f}%)")
        
    finally:
        await client.close()
    
    print("\n🏁 Polymarket Client test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_polymarket_client())
#!/usr/bin/env python3
"""
🌸 Fibonacci Retracement Engine for Nanpin Strategy
Mathematical foundation for 永久ナンピン (Permanent DCA) entry levels
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FibonacciLevel(Enum):
    """Fibonacci retracement levels"""
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.500
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786

@dataclass
class FibonacciRetracement:
    """Single Fibonacci retracement level data"""
    level: float
    price: float
    percentage: str
    strength: str
    confidence: float
    confluence_score: float
    volume_confirmation: bool
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'level': self.level,
            'price': self.price,
            'percentage': self.percentage,
            'strength': self.strength,
            'confidence': self.confidence,
            'confluence_score': self.confluence_score,
            'volume_confirmation': self.volume_confirmation,
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class SwingPoint:
    """Swing high/low point for Fibonacci calculation"""
    timestamp: datetime
    price: float
    volume: float
    type: str  # 'high' or 'low'
    strength: int  # Number of bars on each side
    confirmed: bool
    
class FibonacciEngine:
    """
    🌸 Advanced Fibonacci Retracement Engine
    
    Features:
    - Dynamic swing high/low detection
    - Multiple timeframe analysis
    - Confluence factor calculation
    - Volume confirmation
    - Market regime awareness
    - Position scaling recommendations
    """
    
    def __init__(self, config_path: str = None, macro_analyzer=None):
        """
        Initialize Fibonacci Engine
        
        Args:
            config_path: Path to fibonacci configuration file
            macro_analyzer: MacroAnalyzer instance for dynamic scaling
        """
        self.config = self._load_config(config_path)
        self.current_levels = {}
        self.swing_highs = []
        self.swing_lows = []
        self.last_calculation = None
        self.macro_analyzer = macro_analyzer
        
        # Fibonacci ratios from config
        self.fibonacci_ratios = self._extract_fibonacci_ratios()
        
        # Calculation parameters
        self.min_swing_strength = self.config['calculation']['swing_detection']['min_swing_strength']
        self.lookback_periods = self.config['calculation']['swing_detection']['lookback_periods']
        self.min_move_percentage = self.config['calculation']['swing_detection']['min_move_percentage']
        
        logger.info("🌸 Fibonacci Engine initialized")
        logger.info(f"   Primary levels: {len(self.fibonacci_ratios['primary'])}")
        logger.info(f"   Lookback periods: {self.lookback_periods}")
        logger.info(f"   Min swing strength: {self.min_swing_strength}")
        if macro_analyzer:
            logger.info("   🔮 Macro-aware scaling enabled")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load Fibonacci configuration"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load Fibonacci config: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default Fibonacci configuration"""
        return {
            'fibonacci_ratios': {
                'primary': [
                    {'level': 0.236, 'name': '23.6%', 'strength': 'weak', 'confluence_weight': 1.0},
                    {'level': 0.382, 'name': '38.2%', 'strength': 'moderate', 'confluence_weight': 1.5},
                    {'level': 0.500, 'name': '50.0%', 'strength': 'strong', 'confluence_weight': 2.0},
                    {'level': 0.618, 'name': '61.8%', 'strength': 'very_strong', 'confluence_weight': 3.0},
                    {'level': 0.786, 'name': '78.6%', 'strength': 'extreme', 'confluence_weight': 4.0}
                ]
            },
            'calculation': {
                'swing_detection': {
                    'min_swing_strength': 5,
                    'lookback_periods': 720,
                    'min_move_percentage': 5.0
                }
            },
            'confluence': {
                'technical': {
                    'support_resistance': {'weight': 2.0},
                    'moving_averages': {'weight': 1.5},
                    'round_numbers': {'weight': 1.0}
                }
            }
        }
    
    def _extract_fibonacci_ratios(self) -> Dict:
        """Extract Fibonacci ratios from config"""
        return self.config.get('fibonacci_ratios', {})
    
    def detect_swing_points(self, price_data: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows in price data
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        try:
            swing_highs = []
            swing_lows = []
            
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in price_data.columns:
                    logger.error(f"Missing required column: {col}")
                    return [], []
            
            # Calculate pivot points
            highs = price_data['high'].values
            lows = price_data['low'].values
            volumes = price_data['volume'].values
            timestamps = price_data.index
            
            # Find swing highs
            for i in range(self.min_swing_strength, len(highs) - self.min_swing_strength):
                is_swing_high = True
                current_high = highs[i]
                
                # Check left side
                for j in range(i - self.min_swing_strength, i):
                    if highs[j] >= current_high:
                        is_swing_high = False
                        break
                
                # Check right side
                if is_swing_high:
                    for j in range(i + 1, i + self.min_swing_strength + 1):
                        if highs[j] >= current_high:
                            is_swing_high = False
                            break
                
                if is_swing_high:
                    swing_point = SwingPoint(
                        timestamp=timestamps[i],
                        price=current_high,
                        volume=volumes[i],
                        type='high',
                        strength=self.min_swing_strength,
                        confirmed=True
                    )
                    swing_highs.append(swing_point)
            
            # Find swing lows
            for i in range(self.min_swing_strength, len(lows) - self.min_swing_strength):
                is_swing_low = True
                current_low = lows[i]
                
                # Check left side
                for j in range(i - self.min_swing_strength, i):
                    if lows[j] <= current_low:
                        is_swing_low = False
                        break
                
                # Check right side
                if is_swing_low:
                    for j in range(i + 1, i + self.min_swing_strength + 1):
                        if lows[j] <= current_low:
                            is_swing_low = False
                            break
                
                if is_swing_low:
                    swing_point = SwingPoint(
                        timestamp=timestamps[i],
                        price=current_low,
                        volume=volumes[i],
                        type='low',
                        strength=self.min_swing_strength,
                        confirmed=True
                    )
                    swing_lows.append(swing_point)
            
            # Sort by timestamp
            swing_highs.sort(key=lambda x: x.timestamp)
            swing_lows.sort(key=lambda x: x.timestamp)
            
            logger.info(f"📊 Detected {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
            
            self.swing_highs = swing_highs
            self.swing_lows = swing_lows
            
            return swing_highs, swing_lows
            
        except Exception as e:
            logger.error(f"❌ Failed to detect swing points: {e}")
            return [], []
    
    def calculate_fibonacci_levels(self, price_data: pd.DataFrame, 
                                 reference_high: float = None, 
                                 reference_low: float = None) -> Dict[str, FibonacciRetracement]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            price_data: Historical price data
            reference_high: Specific high to use (optional)
            reference_low: Specific low to use (optional)
            
        Returns:
            Dictionary of Fibonacci retracement levels
        """
        try:
            # Detect swing points if not provided\n            if reference_high is None or reference_low is None:\n                swing_highs, swing_lows = self.detect_swing_points(price_data)\n                \n                if not swing_highs or not swing_lows:\n                    logger.warning(\"⚠️ No swing points detected, cannot calculate Fibonacci levels\")\n                    return {}\n                \n                # Use most recent significant swing high and low\n                reference_high = max([s.price for s in swing_highs[-3:]])  # Recent high\n                reference_low = min([s.price for s in swing_lows[-3:]])   # Recent low\n            \n            # Validate the swing range\n            price_range = reference_high - reference_low\n            range_percentage = (price_range / reference_low) * 100\n            \n            if range_percentage < self.min_move_percentage:\n                logger.warning(f\"⚠️ Price range {range_percentage:.1f}% below minimum {self.min_move_percentage}%\")\n                return {}\n            \n            logger.info(f\"📊 Calculating Fibonacci levels:\")\n            logger.info(f\"   Reference High: ${reference_high:,.2f}\")\n            logger.info(f\"   Reference Low: ${reference_low:,.2f}\")\n            logger.info(f\"   Price Range: ${price_range:,.2f} ({range_percentage:.1f}%)\")\n            \n            # Calculate retracement levels\n            fibonacci_levels = {}\n            \n            for level_data in self.fibonacci_ratios.get('primary', []):\n                level = level_data['level']\n                name = level_data['name']\n                strength = level_data['strength']\n                confluence_weight = level_data['confluence_weight']\n                \n                # Calculate retracement price\n                retracement_price = reference_high - (price_range * level)\n                \n                # Calculate confluence score\n                confluence_score = self._calculate_confluence_score(\n                    retracement_price, price_data, confluence_weight\n                )\n                \n                # Check volume confirmation\n                volume_confirmation = self._check_volume_confirmation(\n                    retracement_price, price_data\n                )\n                \n                # Create retracement object\n                fibonacci_level = FibonacciRetracement(\n                    level=level,\n                    price=retracement_price,\n                    percentage=name,\n                    strength=strength,\n                    confidence=self._calculate_confidence(level, confluence_score),\n                    confluence_score=confluence_score,\n                    volume_confirmation=volume_confirmation,\n                    last_updated=datetime.now()\n                )\n                \n                fibonacci_levels[name] = fibonacci_level\n                \n                logger.info(f\"   {name}: ${retracement_price:,.2f} (confidence: {fibonacci_level.confidence:.2f})\")\n            \n            # Store current levels\n            self.current_levels = fibonacci_levels\n            self.last_calculation = datetime.now()\n            \n            return fibonacci_levels\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to calculate Fibonacci levels: {e}\")\n            return {}\n    \n    def _calculate_confluence_score(self, price: float, price_data: pd.DataFrame, \n                                  base_weight: float) -> float:\n        \"\"\"\n        Calculate confluence score for a price level\n        \n        Args:\n            price: Price level to analyze\n            price_data: Historical price data\n            base_weight: Base confluence weight\n            \n        Returns:\n            Confluence score (higher = more significant)\n        \"\"\"\n        try:\n            confluence_score = base_weight\n            \n            # Support/Resistance confluence\n            sr_weight = self.config.get('confluence', {}).get('technical', {}).get('support_resistance', {}).get('weight', 2.0)\n            sr_score = self._check_support_resistance(price, price_data)\n            confluence_score += sr_score * sr_weight\n            \n            # Moving average confluence\n            ma_weight = self.config.get('confluence', {}).get('technical', {}).get('moving_averages', {}).get('weight', 1.5)\n            ma_score = self._check_moving_average_confluence(price, price_data)\n            confluence_score += ma_score * ma_weight\n            \n            # Round number confluence\n            rn_weight = self.config.get('confluence', {}).get('technical', {}).get('round_numbers', {}).get('weight', 1.0)\n            rn_score = self._check_round_number_confluence(price)\n            confluence_score += rn_score * rn_weight\n            \n            return confluence_score\n            \n        except Exception as e:\n            logger.debug(f\"Failed to calculate confluence score: {e}\")\n            return base_weight\n    \n    def _check_support_resistance(self, price: float, price_data: pd.DataFrame) -> float:\n        \"\"\"Check if price level aligns with historical support/resistance\"\"\"\n        try:\n            tolerance = 0.005  # 0.5% tolerance\n            \n            # Check previous highs and lows\n            highs = price_data['high'].values\n            lows = price_data['low'].values\n            \n            touches = 0\n            \n            # Count touches within tolerance\n            for high in highs:\n                if abs(high - price) / price <= tolerance:\n                    touches += 1\n            \n            for low in lows:\n                if abs(low - price) / price <= tolerance:\n                    touches += 1\n            \n            # Score based on number of touches\n            if touches >= 3:\n                return 2.0  # Strong S/R\n            elif touches >= 2:\n                return 1.0  # Moderate S/R\n            elif touches >= 1:\n                return 0.5  # Weak S/R\n            else:\n                return 0.0  # No S/R\n                \n        except Exception:\n            return 0.0\n    \n    def _check_moving_average_confluence(self, price: float, price_data: pd.DataFrame) -> float:\n        \"\"\"Check if price level aligns with moving averages\"\"\"\n        try:\n            if len(price_data) < 200:\n                return 0.0\n            \n            # Calculate common moving averages\n            ma_periods = [21, 50, 100, 200]\n            proximity_threshold = 0.01  # 1%\n            \n            confluence = 0.0\n            \n            for period in ma_periods:\n                if len(price_data) >= period:\n                    ma = price_data['close'].rolling(period).mean().iloc[-1]\n                    if abs(ma - price) / price <= proximity_threshold:\n                        confluence += 0.5  # Each MA confluence adds 0.5\n            \n            return min(confluence, 2.0)  # Cap at 2.0\n            \n        except Exception:\n            return 0.0\n    \n    def _check_round_number_confluence(self, price: float) -> float:\n        \"\"\"Check if price level is near psychological round numbers\"\"\"\n        try:\n            # Major round numbers for BTC\n            major_levels = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]\n            minor_step = 1000\n            \n            # Check major levels (within 1%)\n            for level in major_levels:\n                if abs(price - level) / level <= 0.01:\n                    return 2.0  # Strong round number\n            \n            # Check minor levels (within 0.5%)\n            nearest_thousand = round(price / minor_step) * minor_step\n            if abs(price - nearest_thousand) / price <= 0.005:\n                return 1.0  # Minor round number\n            \n            return 0.0\n            \n        except Exception:\n            return 0.0\n    \n    def _check_volume_confirmation(self, price: float, price_data: pd.DataFrame) -> bool:\n        \"\"\"Check if price level has volume confirmation\"\"\"\n        try:\n            tolerance = 0.01  # 1% tolerance\n            \n            # Find bars where price was near this level\n            volume_at_level = []\n            \n            for i, row in price_data.iterrows():\n                if (abs(row['high'] - price) / price <= tolerance or \n                    abs(row['low'] - price) / price <= tolerance):\n                    volume_at_level.append(row['volume'])\n            \n            if not volume_at_level:\n                return False\n            \n            # Check if volume was above average\n            avg_volume = price_data['volume'].mean()\n            high_volume_count = sum(1 for v in volume_at_level if v > avg_volume * 1.5)\n            \n            return high_volume_count >= 1\n            \n        except Exception:\n            return False\n    \n    def _calculate_confidence(self, level: float, confluence_score: float) -> float:\n        \"\"\"Calculate confidence score for a Fibonacci level\"\"\"\n        try:\n            # Base confidence based on Fibonacci level strength\n            base_confidence = {\n                0.236: 0.7,   # Weak level\n                0.382: 0.8,   # Moderate level\n                0.500: 0.85,  # Strong psychological level\n                0.618: 0.9,   # Golden ratio - very strong\n                0.786: 0.95   # Deep retracement - extreme\n            }.get(level, 0.75)\n            \n            # Adjust based on confluence\n            confluence_adjustment = min(confluence_score / 10.0, 0.15)  # Max 15% adjustment\n            \n            confidence = min(base_confidence + confluence_adjustment, 1.0)\n            \n            return confidence\n            \n        except Exception:\n            return 0.75  # Default confidence\n    \n    def get_position_scaling_recommendations(self, current_price: float) -> Dict[str, Dict]:\n        \"\"\"\n        Get position scaling recommendations based on current price vs Fibonacci levels\n        \n        Args:\n            current_price: Current BTC price\n            \n        Returns:\n            Dictionary with scaling recommendations for each level\n        \"\"\"\n        try:\n            if not self.current_levels:\n                logger.warning(\"⚠️ No Fibonacci levels calculated\")\n                return {}\n            \n            recommendations = {}\n            \n            for level_name, fib_level in self.current_levels.items():\n                target_price = fib_level.price\n                distance = (current_price - target_price) / target_price\n                \n                # Calculate base multiplier from Fibonacci sequence\n                base_multipliers = {\n                    '23.6%': 1,\n                    '38.2%': 2,\n                    '50.0%': 3,\n                    '61.8%': 5,\n                    '78.6%': 8\n                }\n                \n                base_multiplier = base_multipliers.get(level_name, 1)\n                \n                # Adjust multiplier based on confluence\n                confluence_boost = fib_level.confluence_score / 10.0\n                adjusted_multiplier = base_multiplier * (1 + confluence_boost)\n                \n                # Determine action based on price proximity\n                if distance <= -0.005:  # Price is 0.5% below level\n                    action = \"BUY\"\n                    urgency = \"HIGH\" if distance <= -0.02 else \"MEDIUM\"\n                elif distance <= 0.005:  # Price is within 0.5% of level\n                    action = \"PREPARE\"\n                    urgency = \"MEDIUM\"\n                else:\n                    action = \"WAIT\"\n                    urgency = \"LOW\"\n                \n                recommendations[level_name] = {\n                    'target_price': target_price,\n                    'current_distance_pct': distance * 100,\n                    'action': action,\n                    'urgency': urgency,\n                    'base_multiplier': base_multiplier,\n                    'adjusted_multiplier': adjusted_multiplier,\n                    'confidence': fib_level.confidence,\n                    'confluence_score': fib_level.confluence_score,\n                    'volume_confirmation': fib_level.volume_confirmation,\n                    'reasoning': self._generate_reasoning(fib_level, distance, action)\n                }\n            \n            return recommendations\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get scaling recommendations: {e}\")\n            return {}\n    \n    def _generate_reasoning(self, fib_level: FibonacciRetracement, distance: float, action: str) -> str:\n        \"\"\"Generate human-readable reasoning for recommendation\"\"\"\n        try:\n            reasoning_parts = []\n            \n            # Fibonacci strength\n            reasoning_parts.append(f\"{fib_level.percentage} Fibonacci level ({fib_level.strength} strength)\")\n            \n            # Price distance\n            if distance <= -0.02:\n                reasoning_parts.append(\"price significantly below level (strong buy signal)\")\n            elif distance <= -0.005:\n                reasoning_parts.append(\"price below level (buy signal)\")\n            elif abs(distance) <= 0.005:\n                reasoning_parts.append(\"price at level (prepare for entry)\")\n            else:\n                reasoning_parts.append(\"price above level (wait for retracement)\")\n            \n            # Confluence factors\n            if fib_level.confluence_score > 6:\n                reasoning_parts.append(\"high confluence\")\n            elif fib_level.confluence_score > 4:\n                reasoning_parts.append(\"moderate confluence\")\n            \n            # Volume confirmation\n            if fib_level.volume_confirmation:\n                reasoning_parts.append(\"volume confirmed\")\n            \n            return \", \".join(reasoning_parts)\n            \n        except Exception:\n            return \"Standard Fibonacci analysis\"\n    \n    def get_next_target_level(self, current_price: float) -> Optional[FibonacciRetracement]:\n        \"\"\"\n        Get the next Fibonacci level below current price\n        \n        Args:\n            current_price: Current BTC price\n            \n        Returns:\n            Next Fibonacci level to target, or None\n        \"\"\"\n        try:\n            if not self.current_levels:\n                return None\n            \n            # Find levels below current price\n            levels_below = []\n            \n            for level_name, fib_level in self.current_levels.items():\n                if fib_level.price < current_price:\n                    levels_below.append((fib_level.price, fib_level))\n            \n            if not levels_below:\n                return None\n            \n            # Sort by price (descending) and return the highest level below current price\n            levels_below.sort(key=lambda x: x[0], reverse=True)\n            return levels_below[0][1]\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get next target level: {e}\")\n            return None\n    \n    def validate_levels(self, price_data: pd.DataFrame) -> Dict[str, bool]:\n        \"\"\"\n        Validate calculated Fibonacci levels against recent price action\n        \n        Args:\n            price_data: Recent price data for validation\n            \n        Returns:\n            Dictionary of validation results for each level\n        \"\"\"\n        try:\n            if not self.current_levels:\n                return {}\n            \n            validation_results = {}\n            \n            for level_name, fib_level in self.current_levels.items():\n                target_price = fib_level.price\n                \n                # Check if price has tested this level recently\n                tolerance = 0.01  # 1% tolerance\n                tested = False\n                bounced = False\n                \n                for i, row in price_data.tail(168).iterrows():  # Last 7 days\n                    if abs(row['low'] - target_price) / target_price <= tolerance:\n                        tested = True\n                        \n                        # Check if price bounced from this level\n                        next_close = price_data.loc[price_data.index > i, 'close'].iloc[0] if len(price_data.loc[price_data.index > i]) > 0 else None\n                        if next_close and next_close > target_price:\n                            bounced = True\n                            break\n                \n                validation_results[level_name] = {\n                    'tested': tested,\n                    'bounced': bounced,\n                    'valid': bounced if tested else True,  # Untested levels are considered valid\n                    'target_price': target_price\n                }\n            \n            return validation_results\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to validate levels: {e}\")\n            return {}\n    \n    def export_levels_to_dict(self) -> Dict:\n        \"\"\"\n        Export current Fibonacci levels to dictionary format\n        \n        Returns:\n            Dictionary containing all level data\n        \"\"\"\n        try:\n            export_data = {\n                'calculation_time': self.last_calculation.isoformat() if self.last_calculation else None,\n                'levels': {},\n                'swing_points': {\n                    'highs': [{\n                        'timestamp': sp.timestamp.isoformat(),\n                        'price': sp.price,\n                        'volume': sp.volume,\n                        'strength': sp.strength\n                    } for sp in self.swing_highs[-5:]],  # Last 5 swing highs\n                    'lows': [{\n                        'timestamp': sp.timestamp.isoformat(),\n                        'price': sp.price,\n                        'volume': sp.volume,\n                        'strength': sp.strength\n                    } for sp in self.swing_lows[-5:]]   # Last 5 swing lows\n                },\n                'statistics': {\n                    'total_levels': len(self.current_levels),\n                    'avg_confidence': np.mean([level.confidence for level in self.current_levels.values()]) if self.current_levels else 0,\n                    'volume_confirmed_count': sum(1 for level in self.current_levels.values() if level.volume_confirmation)\n                }\n            }\n            \n            # Add individual levels\n            for level_name, fib_level in self.current_levels.items():\n                export_data['levels'][level_name] = fib_level.to_dict()\n            \n            return export_data\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to export levels: {e}\")\n            return {}\n    \n    def __str__(self) -> str:\n        \"\"\"String representation\"\"\"\n        return f\"FibonacciEngine(levels={len(self.current_levels)}, last_calc={self.last_calculation})\"\n    \n    def __repr__(self) -> str:\n        \"\"\"Detailed representation\"\"\"\n        return (f\"FibonacciEngine(\"\n                f\"levels={len(self.current_levels)}, \"\n                f\"swing_highs={len(self.swing_highs)}, \"\n                f\"swing_lows={len(self.swing_lows)}, \"\n                f\"last_calculation='{self.last_calculation}')\")\n\n\n# ===========================\n# UTILITY FUNCTIONS\n# ===========================\n\ndef create_sample_price_data(start_price: float = 100000, \n                            num_periods: int = 720, \n                            volatility: float = 0.02) -> pd.DataFrame:\n    \"\"\"\n    Create sample BTC price data for testing\n    \n    Args:\n        start_price: Starting BTC price\n        num_periods: Number of hourly periods\n        volatility: Price volatility (standard deviation)\n        \n    Returns:\n        DataFrame with OHLCV data\n    \"\"\"\n    np.random.seed(42)  # For reproducible results\n    \n    # Generate price series with trend and volatility\n    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=num_periods), \n                              periods=num_periods, freq='H')\n    \n    # Create realistic price movements\n    returns = np.random.normal(0.0001, volatility, num_periods)  # Slight upward bias\n    \n    # Add some larger moves (simulating market events)\n    for i in np.random.choice(num_periods, size=int(num_periods * 0.05), replace=False):\n        returns[i] *= np.random.choice([3, -3])  # Random large moves\n    \n    # Calculate prices\n    prices = [start_price]\n    for ret in returns[1:]:\n        new_price = prices[-1] * (1 + ret)\n        prices.append(max(new_price, start_price * 0.5))  # Floor at 50% of start\n    \n    # Create OHLCV data\n    price_data = pd.DataFrame(index=timestamps)\n    price_data['close'] = prices\n    \n    # Generate realistic OHLC\n    price_data['open'] = price_data['close'].shift(1).fillna(start_price)\n    \n    # Add random intrabar volatility\n    intrabar_range = np.random.uniform(0.005, 0.03, num_periods)\n    price_data['high'] = price_data['close'] * (1 + intrabar_range)\n    price_data['low'] = price_data['close'] * (1 - intrabar_range)\n    \n    # Ensure OHLC logic\n    price_data['high'] = np.maximum.reduce([price_data['open'], price_data['close'], price_data['high']])\n    price_data['low'] = np.minimum.reduce([price_data['open'], price_data['close'], price_data['low']])\n    \n    # Generate volume\n    base_volume = 1000000\n    volume_multiplier = np.random.lognormal(0, 0.5, num_periods)\n    price_data['volume'] = base_volume * volume_multiplier\n    \n    return price_data\n\n\n# ===========================\n# TESTING AND EXAMPLES\n# ===========================\n\ndef test_fibonacci_engine():\n    \"\"\"\n    Test the Fibonacci Engine with sample data\n    \"\"\"\n    print(\"🧪 Testing Fibonacci Engine\")\n    print(\"=\" * 50)\n    \n    # Create sample data\n    print(\"📊 Generating sample price data...\")\n    price_data = create_sample_price_data(start_price=95000, num_periods=720)\n    print(f\"   Generated {len(price_data)} hourly candles\")\n    print(f\"   Price range: ${price_data['low'].min():,.0f} - ${price_data['high'].max():,.0f}\")\n    \n    # Initialize engine\n    print(\"\\n🌸 Initializing Fibonacci Engine...\")\n    fib_engine = FibonacciEngine()\n    \n    # Calculate levels\n    print(\"\\n📐 Calculating Fibonacci levels...\")\n    levels = fib_engine.calculate_fibonacci_levels(price_data)\n    \n    if levels:\n        print(f\"\\n✅ Successfully calculated {len(levels)} Fibonacci levels:\")\n        for name, level in levels.items():\n            print(f\"   {name}: ${level.price:,.2f} (confidence: {level.confidence:.2%})\")\n        \n        # Get scaling recommendations\n        current_price = price_data['close'].iloc[-1]\n        print(f\"\\n💰 Current price: ${current_price:,.2f}\")\n        print(\"\\n🎯 Position scaling recommendations:\")\n        \n        recommendations = fib_engine.get_position_scaling_recommendations(current_price)\n        for level_name, rec in recommendations.items():\n            print(f\"   {level_name}: {rec['action']} (multiplier: {rec['adjusted_multiplier']:.1f}x)\")\n            print(f\"      Reasoning: {rec['reasoning']}\")\n        \n        # Validate levels\n        print(\"\\n✅ Validating levels against recent price action:\")\n        validation = fib_engine.validate_levels(price_data)\n        for level_name, result in validation.items():\n            status = \"✅\" if result['valid'] else \"❌\"\n            print(f\"   {level_name}: {status} {'Tested & bounced' if result['bounced'] else 'Valid' if result['valid'] else 'Failed'}\")\n        \n        # Export data\n        print(\"\\n📤 Exporting level data...\")\n        export_data = fib_engine.export_levels_to_dict()\n        print(f\"   Exported {export_data['statistics']['total_levels']} levels\")\n        print(f\"   Average confidence: {export_data['statistics']['avg_confidence']:.2%}\")\n        print(f\"   Volume confirmed: {export_data['statistics']['volume_confirmed_count']} levels\")\n        \n    else:\n        print(\"❌ Failed to calculate Fibonacci levels\")\n    \n    print(\"\\n🏁 Test completed!\")\n\n\nif __name__ == \"__main__\":\n    # Configure logging\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n    )\n    \n    # Run test\n    test_fibonacci_engine()
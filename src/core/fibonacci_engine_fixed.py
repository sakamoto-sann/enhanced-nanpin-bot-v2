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

class SwingPointDetector:
    """Advanced swing point detection for Fibonacci calculations"""
    
    def __init__(self, lookback_period: int = 14, smoothing_factor: float = 0.1):
        self.lookback_period = lookback_period
        self.smoothing_factor = smoothing_factor
    
    def find_swing_points(self, data: pd.DataFrame, volatility_threshold: float = 0.02) -> Tuple[float, float]:
        """Find significant swing high and low points"""
        try:
            # Handle both uppercase and lowercase column names
            high_col = 'high' if 'high' in data.columns else 'High'
            low_col = 'low' if 'low' in data.columns else 'Low' 
            close_col = 'close' if 'close' in data.columns else 'Close'
            
            if len(data) < self.lookback_period:
                return data[high_col].max(), data[low_col].min()
            
            # Calculate price volatility for adaptive periods
            returns = data[close_col].pct_change()
            volatility = returns.rolling(self.lookback_period).std().iloc[-1]
            
            # Adjust lookback based on volatility
            if volatility > volatility_threshold * 2:
                effective_lookback = max(7, self.lookback_period // 2)
            elif volatility < volatility_threshold * 0.5:
                effective_lookback = min(len(data), self.lookback_period * 2)
            else:
                effective_lookback = self.lookback_period
            
            # Find swing points
            recent_data = data.tail(effective_lookback)
            swing_high = recent_data[high_col].max()
            swing_low = recent_data[low_col].min()
            
            # Validate swing points
            price_range = swing_high - swing_low
            if price_range < swing_high * 0.05:  # Less than 5% range
                # Expand search if range is too small
                extended_data = data.tail(effective_lookback * 2)
                swing_high = extended_data[high_col].max()
                swing_low = extended_data[low_col].min()
            
            return swing_high, swing_low
            
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            # Fallback with proper column detection
            high_col = 'high' if 'high' in data.columns else 'High'
            low_col = 'low' if 'low' in data.columns else 'Low'
            return data[high_col].max(), data[low_col].min()

class ConfluenceAnalyzer:
    """Analyze confluence factors for Fibonacci levels"""
    
    def __init__(self):
        self.support_resistance_tolerance = 0.02  # 2%
        self.volume_threshold = 1.2  # 20% above average
    
    def calculate_confluence_score(self, price_level: float, historical_data: pd.DataFrame, 
                                 current_volume: float = None) -> float:
        """Calculate confluence score for a price level"""
        try:
            score = 1.0
            
            # Historical support/resistance
            score += self._analyze_support_resistance(price_level, historical_data)
            
            # Round number psychology
            score += self._analyze_round_numbers(price_level)
            
            # Volume confirmation
            if current_volume:
                score += self._analyze_volume(current_volume, historical_data)
            
            # Moving average confluence
            score += self._analyze_moving_averages(price_level, historical_data)
            
            return min(score, 3.0)  # Cap at 3.0
            
        except Exception as e:
            logger.error(f"Error calculating confluence: {e}")
            return 1.0
    
    def _analyze_support_resistance(self, price_level: float, data: pd.DataFrame) -> float:
        """Analyze historical support/resistance at price level"""
        tolerance = price_level * self.support_resistance_tolerance
        
        touches = data[
            (data['Low'] <= price_level + tolerance) &
            (data['High'] >= price_level - tolerance)
        ]
        
        if len(touches) >= 3:
            return 0.5
        elif len(touches) == 2:
            return 0.3
        elif len(touches) == 1:
            return 0.1
        
        return 0.0
    
    def _analyze_round_numbers(self, price_level: float) -> float:
        """Analyze psychological round number significance"""
        # Major round numbers (every $10,000)
        if price_level % 10000 < 500 or price_level % 10000 > 9500:
            return 0.3
        
        # Medium round numbers (every $5,000)
        if price_level % 5000 < 250 or price_level % 5000 > 4750:
            return 0.2
        
        # Minor round numbers (every $1,000)
        if price_level % 1000 < 100 or price_level % 1000 > 900:
            return 0.1
        
        return 0.0
    
    def _analyze_volume(self, current_volume: float, data: pd.DataFrame) -> float:
        """Analyze volume confirmation"""
        if 'Volume' not in data.columns:
            return 0.0
        
        avg_volume = data['Volume'].tail(20).mean()
        
        if current_volume > avg_volume * self.volume_threshold:
            return 0.2
        elif current_volume > avg_volume:
            return 0.1
        
        return 0.0
    
    def _analyze_moving_averages(self, price_level: float, data: pd.DataFrame) -> float:
        """Analyze moving average confluence"""
        try:
            # Calculate key moving averages
            ma20 = data['Close'].rolling(20).mean().iloc[-1]
            ma50 = data['Close'].rolling(50).mean().iloc[-1]
            ma200 = data['Close'].rolling(200).mean().iloc[-1]
            
            tolerance = price_level * 0.01  # 1%
            
            score = 0.0
            for ma in [ma20, ma50, ma200]:
                if abs(price_level - ma) < tolerance:
                    score += 0.1
            
            return score
            
        except Exception:
            return 0.0

class FibonacciEngine:
    """
    🌸 Advanced Fibonacci Retracement Engine
    
    Calculates optimal Fibonacci entry levels with:
    - Dynamic swing point detection
    - Confluence analysis
    - Volume confirmation
    - Macro intelligence integration
    """
    
    def __init__(self, config_path: str = None, macro_analyzer=None):
        """Initialize Fibonacci engine"""
        self.config_path = config_path
        self.macro_analyzer = macro_analyzer
        self.config = self._load_config()
        
        # Initialize components
        self.swing_detector = SwingPointDetector(
            lookback_period=self.config.get('swing_detection', {}).get('lookback_period', 14)
        )
        self.confluence_analyzer = ConfluenceAnalyzer()
        
        # Current state
        self.current_levels = {}
        self.last_calculation_time = None
        self.last_swing_high = None
        self.last_swing_low = None
        
        logger.info("🌸 Fibonacci Engine initialized")
        logger.info(f"   Levels enabled: {list(self.config.get('levels', {}).keys())}")
    
    def _load_config(self) -> Dict:
        """Load Fibonacci configuration"""
        try:
            if self.config_path and Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✅ Loaded Fibonacci config from {self.config_path}")
                return config
            else:
                logger.warning("⚠️ Using default Fibonacci configuration")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"❌ Failed to load Fibonacci config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default Fibonacci configuration with all levels enabled"""
        return {
            'levels': {
                '23.6%': {
                    'ratio': 0.236,
                    'enabled': True,  # CRITICAL FIX - Enable all levels
                    'multiplier': 2.0,
                    'confidence_threshold': 0.7,
                    'strength': 'weak',
                    'confluence_weight': 1.0
                },
                '38.2%': {
                    'ratio': 0.382,
                    'enabled': True,  # CRITICAL FIX - Enable all levels
                    'multiplier': 3.0,
                    'confidence_threshold': 0.8,
                    'strength': 'moderate',
                    'confluence_weight': 1.5
                },
                '50.0%': {
                    'ratio': 0.500,
                    'enabled': True,  # CRITICAL FIX - Enable all levels
                    'multiplier': 5.0,
                    'confidence_threshold': 0.85,
                    'strength': 'strong',
                    'confluence_weight': 2.0
                },
                '61.8%': {
                    'ratio': 0.618,
                    'enabled': True,  # CRITICAL FIX - Enable all levels
                    'multiplier': 8.0,
                    'confidence_threshold': 0.9,
                    'strength': 'very_strong',
                    'confluence_weight': 3.0
                },
                '78.6%': {
                    'ratio': 0.786,
                    'enabled': True,  # CRITICAL FIX - Enable all levels
                    'multiplier': 13.0,
                    'confidence_threshold': 0.95,
                    'strength': 'extreme',
                    'confluence_weight': 4.0
                }
            },
            'swing_detection': {
                'lookback_period': 14,
                'volatility_threshold': 0.02,
                'min_swing_strength': 5,
                'min_move_percentage': 5.0
            },
            'calculation': {
                'min_data_points': 168,
                'preferred_timeframe': '1h',
                'max_data_age_hours': 2160,
                'min_price_distance': 0.01,
                'max_level_spread': 0.20,
                'require_volume_confirmation': True
            },
            'confluence': {
                'support_resistance_weight': 2.0,
                'volume_weight': 2.5,
                'round_numbers_weight': 1.0,
                'moving_averages_weight': 1.5
            },
            'update_frequency': 300,  # 5 minutes
            'min_price_range_pct': 0.05  # 5%
        }
    
    def calculate_fibonacci_levels(self, market_data: pd.DataFrame, 
                                 current_volume: float = None) -> Dict[str, FibonacciRetracement]:
        """Calculate Fibonacci retracement levels"""
        try:
            if len(market_data) < 20:
                logger.warning("Insufficient data for Fibonacci calculation")
                return {}
            
            # Find swing points
            swing_high, swing_low = self.swing_detector.find_swing_points(market_data)
            price_range = swing_high - swing_low
            
            # Validate price range
            min_range = swing_high * self.config.get('min_price_range_pct', 0.05)
            if price_range < min_range:
                logger.debug(f"Price range too small: {price_range:.2f} < {min_range:.2f}")
                return {}
            
            # Calculate levels
            fibonacci_levels = {}
            current_time = datetime.now()
            
            for level_name, level_config in self.config['levels'].items():
                if not level_config.get('enabled', True):
                    continue
                
                ratio = level_config['ratio']
                fib_price = swing_high - (price_range * ratio)
                
                # Calculate confluence
                confluence_score = self.confluence_analyzer.calculate_confluence_score(
                    fib_price, market_data, current_volume
                )
                
                # Determine strength and confidence
                strength = self._determine_level_strength(confluence_score, level_config)
                confidence = self._calculate_confidence(confluence_score, level_config, market_data)
                
                # Volume confirmation
                volume_confirmation = self._check_volume_confirmation(
                    current_volume, market_data
                ) if current_volume else False
                
                # Create Fibonacci retracement object
                fibonacci_levels[level_name] = FibonacciRetracement(
                    level=ratio,
                    price=fib_price,
                    percentage=level_name,
                    strength=strength,
                    confidence=confidence,
                    confluence_score=confluence_score,
                    volume_confirmation=volume_confirmation,
                    last_updated=current_time
                )
            
            # Update state
            self.current_levels = fibonacci_levels
            self.last_calculation_time = current_time
            self.last_swing_high = swing_high
            self.last_swing_low = swing_low
            
            logger.info(f"✅ Calculated {len(fibonacci_levels)} Fibonacci levels")
            logger.info(f"   Swing range: ${swing_low:,.0f} - ${swing_high:,.0f}")
            
            return fibonacci_levels
            
        except Exception as e:
            logger.error(f"❌ Error calculating Fibonacci levels: {e}")
            return {}
    
    def _determine_level_strength(self, confluence_score: float, level_config: Dict) -> str:
        """Determine the strength of a Fibonacci level"""
        if confluence_score >= 2.5:
            return "VERY_STRONG"
        elif confluence_score >= 2.0:
            return "STRONG"
        elif confluence_score >= 1.5:
            return "MODERATE"
        elif confluence_score >= 1.0:
            return "WEAK"
        else:
            return "VERY_WEAK"
    
    def _calculate_confidence(self, confluence_score: float, level_config: Dict, 
                            market_data: pd.DataFrame) -> float:
        """Calculate confidence score for the level"""
        base_confidence = level_config.get('confidence_threshold', 0.8)
        
        # Adjust based on confluence
        confluence_adjustment = (confluence_score - 1.0) * 0.1
        
        # Adjust based on market volatility
        returns = market_data['Close'].pct_change()
        volatility = returns.rolling(14).std().iloc[-1] if len(returns) > 14 else 0.02
        
        if volatility > 0.05:  # High volatility reduces confidence
            volatility_adjustment = -0.1
        elif volatility < 0.01:  # Very low volatility also reduces confidence
            volatility_adjustment = -0.05
        else:
            volatility_adjustment = 0.0
        
        final_confidence = base_confidence + confluence_adjustment + volatility_adjustment
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _check_volume_confirmation(self, current_volume: float, 
                                 market_data: pd.DataFrame) -> bool:
        """Check if volume confirms the level"""
        if 'Volume' not in market_data.columns:
            return False
        
        avg_volume = market_data['Volume'].tail(20).mean()
        return current_volume > avg_volume * 1.2
    
    def get_position_scaling_recommendations(self, current_price: float) -> Dict[str, Dict]:
        """Get position scaling recommendations based on current price"""
        if not self.current_levels:
            return {}
        
        recommendations = {}
        
        for level_name, fib_level in self.current_levels.items():
            distance_pct = (current_price - fib_level.price) / fib_level.price * 100
            
            # Determine action based on distance and strength
            action, urgency, reasoning = self._determine_action(
                distance_pct, fib_level, level_name
            )
            
            # Calculate adjusted multiplier based on macro conditions
            adjusted_multiplier = self._calculate_adjusted_multiplier(
                fib_level, level_name
            )
            
            recommendations[level_name] = {
                'action': action,
                'urgency': urgency,
                'target_price': fib_level.price,
                'current_distance_pct': distance_pct,
                'strength': fib_level.strength,
                'confidence': fib_level.confidence,
                'base_multiplier': self.config['levels'][level_name]['multiplier'],
                'adjusted_multiplier': adjusted_multiplier,
                'confluence_score': fib_level.confluence_score,
                'volume_confirmation': fib_level.volume_confirmation,
                'reasoning': reasoning
            }
        
        return recommendations
    
    def _determine_action(self, distance_pct: float, fib_level: FibonacciRetracement, 
                        level_name: str) -> Tuple[str, str, str]:
        """Determine trading action for a Fibonacci level"""
        # Entry windows for different levels
        entry_windows = {
            '23.6%': (-2.0, -0.5),
            '38.2%': (-3.0, -0.5),
            '50.0%': (-4.0, -1.0),
            '61.8%': (-5.0, -1.0),
            '78.6%': (-8.0, -2.0)
        }
        
        min_distance, max_distance = entry_windows.get(level_name, (-3.0, -0.5))
        
        if min_distance <= distance_pct <= max_distance:
            # Price is in optimal entry zone
            if fib_level.strength in ['VERY_STRONG', 'STRONG']:
                return 'BUY', 'HIGH', f'Strong {level_name} level with excellent confluence'
            elif fib_level.strength == 'MODERATE':
                return 'BUY', 'MEDIUM', f'Moderate {level_name} level, good opportunity'
            else:
                return 'WATCH', 'LOW', f'Weak {level_name} level, monitor for improvement'
        
        elif distance_pct < min_distance:
            return 'WAIT', 'LOW', f'Price too far below {level_name} level'
        
        elif distance_pct > 0:
            return 'MISSED', 'NONE', f'Price above {level_name} level'
        
        else:
            return 'WATCH', 'MEDIUM', f'Approaching {level_name} level'
    
    def _calculate_adjusted_multiplier(self, fib_level: FibonacciRetracement, 
                                     level_name: str) -> float:
        """Calculate position size multiplier adjusted for macro conditions"""
        base_multiplier = self.config['levels'][level_name]['multiplier']
        
        # Confluence adjustment
        confluence_multiplier = 1.0 + (fib_level.confluence_score - 1.0) * 0.2
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + (fib_level.confidence * 0.5)
        
        # Volume confirmation bonus
        volume_multiplier = 1.1 if fib_level.volume_confirmation else 1.0
        
        # Macro conditions adjustment (if macro analyzer available)
        macro_multiplier = 1.0
        if self.macro_analyzer:
            try:
                macro_conditions = self.macro_analyzer.get_current_sentiment()
                fear_greed = macro_conditions.get('fear_greed_index', 50)
                
                if fear_greed < 25:  # Extreme fear
                    macro_multiplier = 1.3
                elif fear_greed < 40:  # Fear
                    macro_multiplier = 1.1
                elif fear_greed > 75:  # Greed
                    macro_multiplier = 0.8
                
            except Exception as e:
                logger.debug(f"Could not get macro conditions: {e}")
        
        adjusted_multiplier = (
            base_multiplier * 
            confluence_multiplier * 
            confidence_multiplier * 
            volume_multiplier * 
            macro_multiplier
        )
        
        return round(adjusted_multiplier, 2)
    
    def get_current_market_structure(self) -> Dict:
        """Get current market structure analysis"""
        if not self.last_swing_high or not self.last_swing_low:
            return {}
        
        price_range = self.last_swing_high - self.last_swing_low
        range_pct = price_range / self.last_swing_high * 100
        
        return {
            'swing_high': self.last_swing_high,
            'swing_low': self.last_swing_low,
            'price_range': price_range,
            'range_percentage': range_pct,
            'last_calculation': self.last_calculation_time,
            'levels_count': len(self.current_levels),
            'structure_strength': 'STRONG' if range_pct > 15 else 'MODERATE' if range_pct > 8 else 'WEAK'
        }
    
    def is_update_needed(self) -> bool:
        """Check if Fibonacci levels need updating"""
        if not self.last_calculation_time:
            return True
        
        update_frequency = self.config.get('update_frequency', 300)
        elapsed = (datetime.now() - self.last_calculation_time).total_seconds()
        
        return elapsed >= update_frequency
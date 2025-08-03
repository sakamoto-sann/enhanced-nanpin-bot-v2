"""
🧮 Dynamic Position Sizer
Automatically adjusts position sizes based on current collateral balance
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation"""
    base_margin: float
    position_value: float
    leverage: int
    max_levels: int
    scaling_multiplier: float
    total_possible_exposure: float
    capital_usage_pct: float
    risk_level: str
    reasoning: str

class DynamicPositionSizer:
    """
    🧮 Dynamic Position Sizer
    
    Automatically calculates optimal position sizes based on:
    - Current account balance
    - Risk tolerance
    - Market volatility
    - Performance history
    """
    
    def __init__(self, backpack_client, config: Dict):
        """Initialize dynamic position sizer"""
        self.backpack_client = backpack_client
        self.config = config
        
        # Strategy parameters
        strategy_config = config.get('nanpin_strategy', {})
        self.take_profit = strategy_config.get('take_profit_percentage', 0.08)
        self.stop_loss = strategy_config.get('max_drawdown_stop', 0.15)
        self.max_daily_loss = config.get('trading', {}).get('max_daily_loss', 0.05)
        
        # Dynamic sizing parameters
        self.min_position_pct = 0.02    # 2% minimum position size
        self.max_position_pct = 0.15    # 15% maximum position size  
        self.target_leverage = 5        # Default leverage
        self.max_leverage = 20          # Maximum allowed leverage
        self.min_leverage = 1           # Minimum leverage
        
        # Risk management
        self.max_capital_usage = 0.80   # Use max 80% of capital
        self.emergency_buffer = 0.20    # Keep 20% as emergency buffer
        
        # Performance tracking
        self.win_rate = 0.75            # Initial assumption, will update
        self.avg_trade_duration = 393   # Hours, from backtest
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🧮 Dynamic Position Sizer initialized")
        self.logger.info(f"   Target Leverage: {self.target_leverage}x")
        self.logger.info(f"   Position Range: {self.min_position_pct*100:.1f}% - {self.max_position_pct*100:.1f}%")
        self.logger.info(f"   Max Capital Usage: {self.max_capital_usage*100:.1f}%")
    
    async def get_current_balance(self) -> Dict:
        """Get current account balance from Backpack - FUTURES COLLATERAL"""
        try:
            # Get FUTURES COLLATERAL balance (not spot USDC)
            collateral_response = await self.backpack_client.get_collateral_info()
            
            if not collateral_response:
                self.logger.error("❌ Failed to get collateral from Backpack")
                return {}
            
            # Extract total available equity for futures trading
            net_equity_available = float(collateral_response.get('netEquityAvailable', 0))
            assets_value = float(collateral_response.get('assetsValue', 0))
            
            self.logger.info(f"💰 Futures collateral found: ${net_equity_available:.2f}")
            
            # Get open positions value from collateral data
            net_exposure_futures = float(collateral_response.get('netExposureFutures', 0))
            net_equity_locked = float(collateral_response.get('netEquityLocked', 0))
            
            # Calculate available margin for new positions
            total_balance = net_equity_available
            available_margin = net_equity_available - net_equity_locked
            used_margin = net_equity_locked
            open_positions_value = abs(net_exposure_futures)
            
            balance_info = {
                'total_balance': total_balance,
                'available_margin': available_margin,
                'used_margin': used_margin,
                'open_positions_value': open_positions_value,
                'free_balance_pct': (available_margin / total_balance) * 100 if total_balance > 0 else 0
            }
            
            self.logger.info(f"📊 Account Balance Update:")
            self.logger.info(f"   Total Balance: ${total_balance:.2f}")
            self.logger.info(f"   Available Margin: ${available_margin:.2f}")
            self.logger.info(f"   Used Margin: ${used_margin:.2f}")
            self.logger.info(f"   Open Positions Value: ${open_positions_value:.2f}")
            
            return balance_info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting balance: {e}")
            return {}
    
    def calculate_kelly_position_size(self, balance: float, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return self.min_position_pct
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply conservative scaling (use 25% of Kelly)
            conservative_kelly = kelly_fraction * 0.25
            
            # Cap between min and max
            kelly_position_pct = max(self.min_position_pct, min(self.max_position_pct, conservative_kelly))
            
            self.logger.debug(f"Kelly calculation: {kelly_fraction:.3f} → {kelly_position_pct:.3f} (conservative)")
            
            return kelly_position_pct
            
        except Exception as e:
            self.logger.error(f"❌ Kelly calculation error: {e}")
            return self.min_position_pct
    
    def calculate_optimal_leverage(self, balance: float, volatility: float = 0.15) -> int:
        """Calculate optimal leverage based on balance and market conditions"""
        try:
            # Base leverage calculation
            # Smaller accounts can handle more leverage (proportionally)
            if balance < 200:
                base_leverage = 8      # High leverage for small accounts
            elif balance < 500:
                base_leverage = 6      # Medium-high leverage
            elif balance < 1000:
                base_leverage = 5      # Medium leverage
            elif balance < 5000:
                base_leverage = 4      # Medium-low leverage
            else:
                base_leverage = 3      # Conservative for large accounts
            
            # Adjust for volatility
            # Higher volatility = lower leverage
            volatility_factor = max(0.5, min(1.5, 1.0 - (volatility - 0.10) * 2))
            adjusted_leverage = int(base_leverage * volatility_factor)
            
            # Ensure within bounds
            optimal_leverage = max(self.min_leverage, min(self.max_leverage, adjusted_leverage))
            
            self.logger.debug(f"Leverage calc: base={base_leverage}, volatility_factor={volatility_factor:.2f}, final={optimal_leverage}")
            
            return optimal_leverage
            
        except Exception as e:
            self.logger.error(f"❌ Leverage calculation error: {e}")
            return self.target_leverage
    
    def calculate_scaling_sequence(self, base_margin: float, available_margin: float, leverage: int) -> Tuple[float, int]:
        """Calculate optimal scaling multiplier and max levels"""
        try:
            # Adjust scaling based on leverage and available margin
            # Higher leverage = more conservative scaling
            if leverage >= 10:
                base_scaling = 1.2      # Very conservative
            elif leverage >= 5:
                base_scaling = 1.3      # Conservative  
            elif leverage >= 3:
                base_scaling = 1.4      # Moderate
            else:
                base_scaling = 1.5      # Normal
            
            # Calculate how many levels we can afford
            max_affordable_levels = 0
            total_margin_needed = 0
            current_level_margin = base_margin
            
            max_possible_levels = 10  # Theoretical maximum
            
            for level in range(max_possible_levels):
                if total_margin_needed + current_level_margin <= available_margin * self.max_capital_usage:
                    max_affordable_levels += 1
                    total_margin_needed += current_level_margin
                    current_level_margin *= base_scaling
                else:
                    break
            
            # Ensure we have at least 1 level, max 8 for safety
            max_levels = max(1, min(8, max_affordable_levels))
            
            self.logger.debug(f"Scaling calc: multiplier={base_scaling:.2f}, max_levels={max_levels}")
            
            return base_scaling, max_levels
            
        except Exception as e:
            self.logger.error(f"❌ Scaling calculation error: {e}")
            return 1.3, 3
    
    async def calculate_dynamic_position_size(self, current_price: float = None) -> Optional[PositionSizeRecommendation]:
        """Calculate dynamic position size based on current conditions"""
        try:
            self.logger.info("🧮 Calculating dynamic position size...")
            
            # Get current balance
            balance_info = await self.get_current_balance()
            if not balance_info:
                self.logger.error("❌ Cannot calculate position size without balance info")
                return None
            
            total_balance = balance_info['total_balance']
            available_margin = balance_info['available_margin']
            
            # Minimum balance check
            if total_balance < 50:
                self.logger.warning("⚠️ Balance too low for trading")
                return None
            
            # Calculate Kelly position size
            kelly_position_pct = self.calculate_kelly_position_size(
                balance=total_balance,
                win_rate=self.win_rate,
                avg_win=self.take_profit,
                avg_loss=self.stop_loss
            )
            
            # Calculate optimal leverage
            leverage = self.calculate_optimal_leverage(total_balance)
            
            # Calculate base position
            base_margin = total_balance * kelly_position_pct
            base_position_value = base_margin * leverage
            
            # Calculate scaling parameters
            scaling_multiplier, max_levels = self.calculate_scaling_sequence(
                base_margin, available_margin, leverage
            )
            
            # Calculate total possible exposure
            total_margin_needed = 0
            current_margin = base_margin
            
            for level in range(max_levels):
                total_margin_needed += current_margin
                current_margin *= scaling_multiplier
            
            total_possible_exposure = total_margin_needed * leverage
            capital_usage_pct = (total_margin_needed / total_balance) * 100
            
            # Risk assessment
            if leverage >= 10:
                risk_level = "HIGH"
            elif leverage >= 5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Generate reasoning
            reasoning = f"Based on ${total_balance:.0f} balance: {kelly_position_pct*100:.1f}% Kelly sizing with {leverage}x leverage"
            
            recommendation = PositionSizeRecommendation(
                base_margin=base_margin,
                position_value=base_position_value,
                leverage=leverage,
                max_levels=max_levels,
                scaling_multiplier=scaling_multiplier,
                total_possible_exposure=total_possible_exposure,
                capital_usage_pct=capital_usage_pct,
                risk_level=risk_level,
                reasoning=reasoning
            )
            
            self.logger.info("✅ Dynamic position size calculated:")
            self.logger.info(f"   Base Margin: ${recommendation.base_margin:.2f}")
            self.logger.info(f"   Position Value: ${recommendation.position_value:.2f}")
            self.logger.info(f"   Leverage: {recommendation.leverage}x")
            self.logger.info(f"   Max Levels: {recommendation.max_levels}")
            self.logger.info(f"   Scaling: {recommendation.scaling_multiplier:.2f}x")
            self.logger.info(f"   Capital Usage: {recommendation.capital_usage_pct:.1f}%")
            self.logger.info(f"   Risk Level: {recommendation.risk_level}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic position size calculation failed: {e}")
            return None
    
    async def update_performance_metrics(self, trades_history: list):
        """Update performance metrics based on recent trades"""
        try:
            if not trades_history:
                return
            
            # Calculate recent win rate
            recent_trades = trades_history[-50:]  # Last 50 trades
            winning_trades = [t for t in recent_trades if t.get('profit', 0) > 0]
            
            if len(recent_trades) >= 10:  # Need at least 10 trades for meaningful stats
                self.win_rate = len(winning_trades) / len(recent_trades)
                
                # Calculate average win/loss
                wins = [t['profit'] for t in winning_trades]
                losses = [abs(t['profit']) for t in recent_trades if t.get('profit', 0) < 0]
                
                if wins:
                    avg_win = sum(wins) / len(wins)
                else:
                    avg_win = self.take_profit
                
                if losses:
                    avg_loss = sum(losses) / len(losses)
                else:
                    avg_loss = self.stop_loss
                
                self.logger.info(f"📊 Performance metrics updated:")
                self.logger.info(f"   Win Rate: {self.win_rate*100:.1f}%")
                self.logger.info(f"   Avg Win: {avg_win*100:.1f}%")
                self.logger.info(f"   Avg Loss: {avg_loss*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Performance metrics update failed: {e}")
    
    def get_current_config_overrides(self, recommendation: PositionSizeRecommendation) -> Dict:
        """Generate config overrides for current recommendation"""
        try:
            config_overrides = {
                'base_investment': round(recommendation.base_margin, 2),
                'scaling_multiplier': round(recommendation.scaling_multiplier, 2),
                'max_nanpin_levels': recommendation.max_levels,
                'dynamic_leverage': recommendation.leverage,
                'max_position_size': round(recommendation.base_margin / 1000, 3),  # As fraction for 1k base
                'position_sizing_method': 'dynamic_kelly',
                'last_updated': datetime.now().isoformat(),
                'reasoning': recommendation.reasoning
            }
            
            return config_overrides
            
        except Exception as e:
            self.logger.error(f"❌ Config override generation failed: {e}")
            return {}
    
    async def should_adjust_position_size(self) -> bool:
        """Check if position size should be adjusted"""
        try:
            # Get current balance
            balance_info = await self.get_current_balance()
            if not balance_info:
                return False
            
            # Get last recommendation time (would be stored somewhere)
            # For now, adjust every time balance changes significantly
            current_balance = balance_info['total_balance']
            
            # You'd store last_balance somewhere persistent
            # For now, assume we should always check
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Position size adjustment check failed: {e}")
            return False
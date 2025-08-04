#!/usr/bin/env python3
"""
⚖️ Optimal Leverage Calculator for Backpack BTC Futures
Risk/Reward analysis for Enhanced Nanpin Bot
"""

import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import logging

class LeverageOptimizer:
    """Calculate optimal leverage for Backpack BTC futures trading"""
    
    def __init__(self, config_path: str = "config/enhanced_nanpin_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Backtest results (from our analysis)
        self.backtest_results = {
            'total_return': 0.5241,        # 52.41%
            'sharpe_ratio': 2.04,
            'max_drawdown': 0.0822,        # 8.22%
            'win_rate': 0.7882,            # 78.82%
            'avg_trade_duration': 392.8,   # hours
            'volatility': 0.1079           # 10.79% annual
        }
        
        # Strategy parameters
        self.strategy = self.config['nanpin_strategy']
        self.take_profit = self.strategy['take_profit_percentage']  # 8%
        self.stop_loss = self.strategy['max_drawdown_stop']        # 15%
        self.base_investment = self.strategy['base_investment']     # $100
        self.max_levels = self.config['trading']['max_nanpin_levels']  # 8
        
        # Risk management
        self.total_capital = 10000  # $10k assumption
        self.max_daily_loss = self.config['trading']['max_daily_loss']  # 5%
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size_requirements(self):
        """Calculate what position sizes we need"""
        try:
            # Current Kelly-optimal position size
            kelly_optimal = 2500  # 25% of $10k (from our analysis)
            
            # Current scaling sequence
            scaling_multiplier = self.strategy['scaling_multiplier']  # 1.5
            position_sizes = []
            
            for level in range(self.max_levels):
                size = self.base_investment * (scaling_multiplier ** level)
                position_sizes.append(size)
            
            total_max_exposure = sum(position_sizes)
            
            results = {
                'kelly_optimal_single': kelly_optimal,
                'current_base': self.base_investment,
                'current_scaling_sequence': position_sizes,
                'current_max_exposure': total_max_exposure,
                'max_exposure_pct': (total_max_exposure / self.total_capital) * 100
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Position size calculation failed: {e}")
            return {}
    
    def calculate_leverage_scenarios(self):
        """Calculate different leverage scenarios"""
        try:
            scenarios = {}
            
            # Leverage options on Backpack (typically 1x to 20x for BTC)
            leverage_options = [1, 2, 3, 5, 8, 10, 15, 20]
            
            # Kelly optimal position size
            kelly_position = 2500  # $2500 per trade
            
            for leverage in leverage_options:
                # Calculate required margin for Kelly position
                required_margin = kelly_position / leverage
                
                # Calculate max position with available capital
                available_margin = self.total_capital * 0.8  # Keep 20% buffer
                max_position_value = available_margin * leverage
                
                # Risk calculations
                liquidation_distance = 1 / leverage  # Approximate liquidation distance
                
                # Account for fees (Backpack futures fees ~0.02% maker, 0.05% taker)
                entry_fee = 0.0005  # 0.05%
                exit_fee = 0.0005   # 0.05%
                total_fee_cost = entry_fee + exit_fee
                
                # Effective take profit and stop loss after fees
                effective_take_profit = self.take_profit - total_fee_cost
                effective_stop_loss = self.stop_loss + total_fee_cost
                
                # Risk/Reward ratio
                risk_reward_ratio = effective_take_profit / effective_stop_loss if effective_stop_loss > 0 else 0
                
                # Expected value calculation
                win_rate = self.backtest_results['win_rate']
                expected_value = (win_rate * effective_take_profit) - ((1 - win_rate) * effective_stop_loss)
                
                # Safety margin from liquidation
                safety_margin = liquidation_distance - effective_stop_loss
                
                scenarios[leverage] = {
                    'required_margin': required_margin,
                    'max_position_value': max_position_value,
                    'liquidation_distance': liquidation_distance * 100,  # %
                    'effective_take_profit': effective_take_profit * 100,  # %
                    'effective_stop_loss': effective_stop_loss * 100,      # %
                    'risk_reward_ratio': risk_reward_ratio,
                    'expected_value': expected_value * 100,  # %
                    'safety_margin': safety_margin * 100,    # %
                    'total_fee_cost': total_fee_cost * 100,  # %
                    'kelly_positions_possible': int(available_margin / required_margin),
                    'recommended': False
                }
            
            # Mark recommended leverage levels
            for leverage, data in scenarios.items():
                # Criteria for recommendation:
                # 1. Safety margin > 5% (well above liquidation)
                # 2. Expected value > 2%
                # 3. Can make at least 3 Kelly positions
                # 4. Risk/reward ratio > 0.4
                
                is_recommended = (
                    data['safety_margin'] > 5.0 and
                    data['expected_value'] > 2.0 and
                    data['kelly_positions_possible'] >= 3 and
                    data['risk_reward_ratio'] > 0.4
                )
                
                scenarios[leverage]['recommended'] = is_recommended
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"❌ Leverage scenario calculation failed: {e}")
            return {}
    
    def calculate_nanpin_scaling_with_leverage(self, leverage: int):
        """Calculate how nanpin scaling works with leverage"""
        try:
            # Kelly optimal first position
            kelly_base = 2500
            required_margin_base = kelly_base / leverage
            
            # Calculate scaled positions
            scaling_multiplier = self.strategy['scaling_multiplier']
            
            positions = []
            total_margin_used = 0
            total_position_value = 0
            
            for level in range(self.max_levels):
                position_value = kelly_base * (scaling_multiplier ** level)
                margin_required = position_value / leverage
                
                # Check if we have enough margin
                if total_margin_used + margin_required <= self.total_capital * 0.8:  # 80% margin usage limit
                    positions.append({
                        'level': level + 1,
                        'position_value': position_value,
                        'margin_required': margin_required,
                        'cumulative_margin': total_margin_used + margin_required,
                        'cumulative_position': total_position_value + position_value
                    })
                    
                    total_margin_used += margin_required
                    total_position_value += position_value
                else:
                    break
            
            # Calculate liquidation risk
            # Approximate liquidation price = entry_price * (1 - 1/leverage + maintenance_margin)
            maintenance_margin = 0.005  # 0.5% typical for BTC futures
            liquidation_distance = (1 / leverage) - maintenance_margin
            
            results = {
                'leverage': leverage,
                'max_levels_possible': len(positions),
                'total_margin_required': total_margin_used,
                'total_position_value': total_position_value,
                'margin_usage_pct': (total_margin_used / self.total_capital) * 100,
                'liquidation_distance_pct': liquidation_distance * 100,
                'position_sequence': positions
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Nanpin scaling calculation failed: {e}")
            return {}
    
    def generate_leverage_recommendation(self):
        """Generate comprehensive leverage recommendation"""
        try:
            report = []
            report.append("⚖️ OPTIMAL LEVERAGE ANALYSIS FOR BACKPACK BTC FUTURES")
            report.append("=" * 60)
            report.append("")
            
            # Position size analysis
            position_analysis = self.calculate_position_size_requirements()
            
            report.append("💰 CURRENT POSITION SIZING ANALYSIS")
            report.append("-" * 35)
            report.append(f"Kelly Optimal Position: ${position_analysis.get('kelly_optimal_single', 0):,.0f}")
            report.append(f"Current Base Position:  ${position_analysis.get('current_base', 0):,.0f}")
            report.append(f"Current Max Exposure:   ${position_analysis.get('current_max_exposure', 0):,.0f}")
            report.append(f"Max Exposure %:         {position_analysis.get('max_exposure_pct', 0):.1f}%")
            report.append("")
            
            # Leverage scenarios
            scenarios = self.calculate_leverage_scenarios()
            
            report.append("⚖️ LEVERAGE SCENARIO ANALYSIS")
            report.append("-" * 35)
            report.append("Lev | Margin | Position | R/R  | ExpVal | Safety | Recommended")
            report.append("-" * 65)
            
            recommended_leverages = []
            
            for leverage, data in sorted(scenarios.items()):
                recommendation = "✅ YES" if data['recommended'] else "❌ NO"
                if data['recommended']:
                    recommended_leverages.append(leverage)
                
                report.append(f"{leverage:>2}x | ${data['required_margin']:>5.0f} | ${data['max_position_value']:>7.0f} | "
                            f"{data['risk_reward_ratio']:>4.2f} | {data['expected_value']:>5.1f}% | "
                            f"{data['safety_margin']:>5.1f}% | {recommendation}")
            
            report.append("")
            
            # Detailed analysis for recommended leverages
            if recommended_leverages:
                report.append("🎯 RECOMMENDED LEVERAGE SETTINGS")
                report.append("-" * 35)
                
                for leverage in recommended_leverages[:3]:  # Top 3 recommendations
                    data = scenarios[leverage]
                    nanpin_analysis = self.calculate_nanpin_scaling_with_leverage(leverage)
                    
                    report.append(f"📊 {leverage}x LEVERAGE ANALYSIS:")
                    report.append(f"   Margin per Kelly position: ${data['required_margin']:,.0f}")
                    report.append(f"   Max nanpin levels:         {nanpin_analysis.get('max_levels_possible', 0)}")
                    report.append(f"   Total margin usage:        {nanpin_analysis.get('margin_usage_pct', 0):.1f}%")
                    report.append(f"   Risk/Reward ratio:         {data['risk_reward_ratio']:.2f}")
                    report.append(f"   Expected value:            {data['expected_value']:.1f}%")
                    report.append(f"   Safety from liquidation:   {data['safety_margin']:.1f}%")
                    report.append(f"   Fee cost:                  {data['total_fee_cost']:.2f}%")
                    report.append("")
            
            # Final recommendation
            report.append("🚀 FINAL RECOMMENDATION")
            report.append("-" * 35)
            
            if recommended_leverages:
                optimal_leverage = recommended_leverages[0]  # Best recommended leverage
                optimal_data = scenarios[optimal_leverage]
                
                report.append(f"OPTIMAL LEVERAGE: {optimal_leverage}x")
                report.append("")
                report.append("Backpack Settings:")
                report.append(f"   Symbol: BTC_USDC_PERP")
                report.append(f"   Leverage: {optimal_leverage}x")
                report.append(f"   Position Size: ${optimal_data['required_margin']:,.0f} margin per trade")
                report.append(f"   Max Positions: {optimal_data['kelly_positions_possible']} simultaneous")
                report.append("")
                
                # Updated config recommendations
                updated_base = optimal_data['required_margin']
                
                report.append("Config Updates Needed:")
                report.append(f"   base_investment: {updated_base:.0f}  # Was: {self.base_investment}")
                report.append(f"   max_position_size: 0.25      # 25% of capital (Kelly optimal)")
                report.append(f"   # This gives {updated_base * optimal_leverage:.0f} position value with {optimal_leverage}x leverage")
                report.append("")
                
                # Risk management
                report.append("Risk Management:")
                report.append(f"   Stop Loss: {self.stop_loss*100:.1f}% (Safe: {optimal_data['safety_margin']:.1f}% from liquidation)")
                report.append(f"   Take Profit: {self.take_profit*100:.1f}% (After {optimal_data['total_fee_cost']:.2f}% fees)")
                report.append(f"   Daily Loss Limit: {self.max_daily_loss*100:.1f}% of total capital")
                report.append("")
                
                # Expected performance
                annual_trades = 365 * 24 / optimal_data.get('avg_trade_duration', 393)  # trades per year
                annual_expected_return = annual_trades * optimal_data['expected_value'] / 100
                
                report.append("Expected Performance:")
                report.append(f"   Expected return per trade: {optimal_data['expected_value']:.1f}%")
                report.append(f"   Estimated trades per year: {annual_trades:.0f}")
                report.append(f"   Estimated annual return:   {annual_expected_return*100:.1f}%")
                report.append(f"   Estimated Sharpe ratio:    {annual_expected_return/0.15:.1f} (vs target 3.0)")
                
            else:
                report.append("⚠️ NO LEVERAGE RECOMMENDED with current parameters")
                report.append("Consider reducing position sizes or adjusting stop losses")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            return f"Error generating leverage recommendation: {e}"

def main():
    """Run leverage optimization analysis"""
    try:
        print("⚖️ Starting Optimal Leverage Analysis for Backpack BTC Futures...")
        
        optimizer = LeverageOptimizer()
        recommendation = optimizer.generate_leverage_recommendation()
        
        print("\n" + recommendation)
        
        # Save recommendation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"backtest_results/leverage_recommendation_{timestamp}.txt", 'w') as f:
            f.write(recommendation)
        
        print(f"\n💾 Recommendation saved to: backtest_results/leverage_recommendation_{timestamp}.txt")
        
    except Exception as e:
        print(f"❌ Leverage analysis failed: {e}")

if __name__ == "__main__":
    main()
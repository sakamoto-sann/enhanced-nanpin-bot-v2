#!/usr/bin/env python3
"""
🚀 High Leverage Optimization for Small Collateral
Find optimal leverage for limited capital scenarios
"""

import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import logging

class HighLeverageOptimizer:
    """Optimize leverage for small collateral scenarios"""
    
    def __init__(self, config_path: str = "config/enhanced_nanpin_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Strategy parameters
        self.strategy = self.config['nanpin_strategy']
        self.take_profit = self.strategy['take_profit_percentage']  # 8%
        self.stop_loss = self.strategy['max_drawdown_stop']        # 15%
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_leverage_scenarios_small_capital(self, capital_amounts: list):
        """Calculate optimal leverage for different capital amounts"""
        try:
            scenarios = {}
            
            # High leverage options (what you actually want to use)
            leverage_options = [3, 5, 8, 10, 15, 20]
            
            for capital in capital_amounts:
                scenarios[capital] = {}
                
                for leverage in leverage_options:
                    # Calculate position sizes
                    # Use more conservative position sizing for high leverage
                    conservative_position_pct = min(0.15, 0.25 / np.sqrt(leverage))  # Scale down with leverage
                    position_size_usd = capital * conservative_position_pct * leverage
                    margin_required = position_size_usd / leverage
                    
                    # Liquidation calculations for BTC futures
                    # Backpack liquidation ≈ entry_price * (1 - 1/leverage + maintenance_margin)
                    maintenance_margin = 0.005  # 0.5% for BTC
                    liquidation_distance = (1 / leverage) - maintenance_margin
                    
                    # Our stop loss vs liquidation safety
                    safety_buffer = liquidation_distance - self.stop_loss
                    
                    # Fee impact (higher with leverage)
                    maker_fee = 0.0002  # 0.02%
                    taker_fee = 0.0005  # 0.05% 
                    avg_fee = (maker_fee + taker_fee) / 2
                    total_fee_cost = avg_fee * 2  # Entry + exit
                    
                    # Effective returns after fees
                    effective_take_profit = self.take_profit - total_fee_cost
                    effective_stop_loss = self.stop_loss + total_fee_cost
                    
                    # Risk/reward with leverage
                    risk_reward_ratio = effective_take_profit / effective_stop_loss
                    
                    # Expected value (using 78.82% win rate from backtest)
                    win_rate = 0.7882
                    expected_value = (win_rate * effective_take_profit) - ((1 - win_rate) * effective_stop_loss)
                    
                    # ROI on margin (the key metric for leverage)
                    roi_on_margin = expected_value * leverage
                    
                    # How many positions can you open
                    max_positions = int((capital * 0.8) / margin_required)  # 80% capital usage
                    
                    # Daily/monthly return potential
                    trades_per_day = 24 / 392.8  # Based on avg trade duration from backtest
                    daily_return_potential = roi_on_margin * trades_per_day
                    monthly_return_potential = daily_return_potential * 30
                    
                    # Risk assessment
                    is_safe = safety_buffer > 0.02  # At least 2% buffer above stop loss
                    is_profitable = expected_value > 0.02  # At least 2% expected value
                    is_practical = max_positions >= 1  # Can open at least 1 position
                    
                    recommended = is_safe and is_profitable and is_practical
                    
                    scenarios[capital][leverage] = {
                        'position_size_usd': position_size_usd,
                        'margin_required': margin_required,
                        'liquidation_distance_pct': liquidation_distance * 100,
                        'safety_buffer_pct': safety_buffer * 100,
                        'effective_take_profit_pct': effective_take_profit * 100,
                        'effective_stop_loss_pct': effective_stop_loss * 100,
                        'risk_reward_ratio': risk_reward_ratio,
                        'expected_value_pct': expected_value * 100,
                        'roi_on_margin_pct': roi_on_margin * 100,
                        'max_positions': max_positions,
                        'daily_return_potential_pct': daily_return_potential * 100,
                        'monthly_return_potential_pct': monthly_return_potential * 100,
                        'total_fee_cost_pct': total_fee_cost * 100,
                        'recommended': recommended
                    }
            
            return scenarios
            
        except Exception as e:
            self.logger.error(f"❌ High leverage calculation failed: {e}")
            return {}
    
    def calculate_nanpin_scaling_high_leverage(self, capital: float, leverage: int):
        """Calculate nanpin scaling with high leverage"""
        try:
            # Conservative base position for high leverage
            base_position_pct = min(0.15, 0.25 / np.sqrt(leverage))
            base_margin = capital * base_position_pct
            base_position_value = base_margin * leverage
            
            # Scaling multiplier (reduce for high leverage)
            scaling_multiplier = max(1.2, 1.5 - (leverage * 0.02))  # Reduce scaling with higher leverage
            
            positions = []
            total_margin_used = 0
            total_position_value = 0
            
            max_levels = min(8, int(10 - leverage * 0.3))  # Fewer levels with higher leverage
            
            for level in range(max_levels):
                margin_this_level = base_margin * (scaling_multiplier ** level)
                position_value_this_level = margin_this_level * leverage
                
                # Check margin availability
                if total_margin_used + margin_this_level <= capital * 0.8:  # 80% max usage
                    positions.append({
                        'level': level + 1,
                        'margin_required': margin_this_level,
                        'position_value': position_value_this_level,
                        'cumulative_margin': total_margin_used + margin_this_level,
                        'cumulative_position_value': total_position_value + position_value_this_level
                    })
                    
                    total_margin_used += margin_this_level
                    total_position_value += position_value_this_level
                else:
                    break
            
            # Risk metrics
            margin_usage_pct = (total_margin_used / capital) * 100
            max_exposure_ratio = total_position_value / capital
            
            results = {
                'capital': capital,
                'leverage': leverage,
                'base_margin': base_margin,
                'base_position_value': base_position_value,
                'scaling_multiplier': scaling_multiplier,
                'max_levels_possible': len(positions),
                'total_margin_used': total_margin_used,
                'total_position_value': total_position_value,
                'margin_usage_pct': margin_usage_pct,
                'max_exposure_ratio': max_exposure_ratio,
                'positions': positions
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Nanpin scaling calculation failed: {e}")
            return {}
    
    def generate_small_capital_recommendations(self):
        """Generate recommendations for small capital scenarios"""
        try:
            report = []
            report.append("🚀 HIGH LEVERAGE OPTIMIZATION FOR SMALL CAPITAL")
            report.append("=" * 55)
            report.append("")
            
            # Test different capital amounts
            capital_amounts = [500, 1000, 2000, 5000, 10000]
            scenarios = self.calculate_leverage_scenarios_small_capital(capital_amounts)
            
            # Find best scenarios for each capital amount
            best_recommendations = {}
            
            for capital, leverage_data in scenarios.items():
                best_leverage = None
                best_roi = 0
                
                for leverage, data in leverage_data.items():
                    if data['recommended'] and data['roi_on_margin_pct'] > best_roi:
                        best_roi = data['roi_on_margin_pct']
                        best_leverage = leverage
                
                if best_leverage:
                    best_recommendations[capital] = {
                        'leverage': best_leverage,
                        'data': leverage_data[best_leverage]
                    }
            
            # Display recommendations
            report.append("💰 OPTIMAL LEVERAGE BY CAPITAL AMOUNT")
            report.append("-" * 45)
            report.append("Capital | Leverage | Position | Margin | ROI/Trade | Monthly ROI | Risk")
            report.append("-" * 75)
            
            for capital in sorted(best_recommendations.keys()):
                rec = best_recommendations[capital]
                leverage = rec['leverage']
                data = rec['data']
                
                risk_level = "LOW" if data['safety_buffer_pct'] > 5 else "MED" if data['safety_buffer_pct'] > 2 else "HIGH"
                
                report.append(f"${capital:>4} | {leverage:>7}x | ${data['position_size_usd']:>7.0f} | "
                            f"${data['margin_required']:>5.0f} | {data['roi_on_margin_pct']:>8.1f}% | "
                            f"{data['monthly_return_potential_pct']:>10.1f}% | {risk_level}")
            
            report.append("")
            
            # Detailed analysis for small capital scenarios
            small_capital_scenarios = [500, 1000, 2000]
            
            for capital in small_capital_scenarios:
                if capital in best_recommendations:
                    rec = best_recommendations[capital]
                    leverage = rec['leverage']
                    data = rec['data']
                    
                    # Get nanpin analysis
                    nanpin_analysis = self.calculate_nanpin_scaling_high_leverage(capital, leverage)
                    
                    report.append(f"🎯 DETAILED ANALYSIS: ${capital} CAPITAL WITH {leverage}x LEVERAGE")
                    report.append("-" * 50)
                    report.append(f"Base Position Size:     ${data['position_size_usd']:,.0f}")
                    report.append(f"Margin Required:        ${data['margin_required']:,.0f}")
                    report.append(f"Max Positions:          {data['max_positions']}")
                    report.append(f"Liquidation Distance:   {data['liquidation_distance_pct']:.1f}%")
                    report.append(f"Stop Loss:              {data['effective_stop_loss_pct']:.1f}%")
                    report.append(f"Safety Buffer:          {data['safety_buffer_pct']:.1f}%")
                    report.append(f"Take Profit:            {data['effective_take_profit_pct']:.1f}%")
                    report.append(f"Expected Value:         {data['expected_value_pct']:.1f}%")
                    report.append(f"ROI on Margin:          {data['roi_on_margin_pct']:.1f}%")
                    report.append(f"Daily Return Potential: {data['daily_return_potential_pct']:.1f}%")
                    report.append(f"Monthly Return Potential: {data['monthly_return_potential_pct']:.1f}%")
                    report.append("")
                    
                    # Nanpin scaling details
                    report.append("Nanpin Scaling Sequence:")
                    for i, pos in enumerate(nanpin_analysis['positions'][:5]):  # Show first 5 levels
                        report.append(f"  Level {pos['level']}: ${pos['margin_required']:>6.0f} margin → "
                                    f"${pos['position_value']:>8.0f} position")
                    
                    report.append(f"Total Margin Usage: ${nanpin_analysis['total_margin_used']:,.0f} "
                                f"({nanpin_analysis['margin_usage_pct']:.1f}%)")
                    report.append("")
                    
                    # Config recommendations
                    report.append("Bot Config Updates:")
                    report.append(f"  base_investment: {data['margin_required']:.0f}")
                    report.append(f"  scaling_multiplier: {nanpin_analysis['scaling_multiplier']:.2f}")
                    report.append(f"  max_nanpin_levels: {nanpin_analysis['max_levels_possible']}")
                    report.append("")
                    
                    # Backpack settings
                    report.append("Backpack Settings:")
                    report.append(f"  Symbol: BTC_USDC_PERP")
                    report.append(f"  Leverage: {leverage}x")
                    report.append(f"  Position Size: ${data['margin_required']:.0f} margin per trade")
                    report.append(f"  Risk Management: {data['effective_stop_loss_pct']:.1f}% stop loss")
                    report.append("")
            
            # Risk warnings and tips
            report.append("⚠️ HIGH LEVERAGE RISK MANAGEMENT")
            report.append("-" * 35)
            report.append("1. START SMALL: Test with minimum position sizes first")
            report.append("2. MONITOR CLOSELY: Check positions every few hours")
            report.append("3. TIGHT STOPS: Use 15% stop loss religiously") 
            report.append("4. LIQUIDATION ALERTS: Set alerts at 20% from entry")
            report.append("5. CAPITAL BUFFER: Keep 20% capital unused for emergencies")
            report.append("6. SCALE GRADUALLY: Increase size only after proven success")
            report.append("")
            
            # Expected performance comparison
            if 1000 in best_recommendations:
                rec_1k = best_recommendations[1000]
                leverage_1k = rec_1k['leverage']
                data_1k = rec_1k['data']
                
                report.append("📊 PERFORMANCE COMPARISON ($1000 CAPITAL)")
                report.append("-" * 40)
                report.append(f"Current Strategy (1x):     ~5% monthly return")
                report.append(f"High Leverage ({leverage_1k}x):         {data_1k['monthly_return_potential_pct']:.1f}% monthly return")
                report.append(f"Improvement:               {data_1k['monthly_return_potential_pct']/5:.1f}x better returns")
                report.append("")
                report.append(f"Annual Potential:          {data_1k['monthly_return_potential_pct'] * 12:.0f}% per year")
                report.append(f"Risk Level:                {'HIGH' if data_1k['safety_buffer_pct'] < 3 else 'MEDIUM'}")
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            return f"Error generating high leverage recommendations: {e}"

def main():
    """Run high leverage optimization"""
    try:
        print("🚀 Starting High Leverage Optimization for Small Capital...")
        
        optimizer = HighLeverageOptimizer()
        recommendations = optimizer.generate_small_capital_recommendations()
        
        print("\n" + recommendations)
        
        # Save recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"backtest_results/high_leverage_recommendations_{timestamp}.txt", 'w') as f:
            f.write(recommendations)
        
        print(f"\n💾 Recommendations saved to: backtest_results/high_leverage_recommendations_{timestamp}.txt")
        
    except Exception as e:
        print(f"❌ High leverage analysis failed: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🧮 Test Dynamic Position Sizer
Test the dynamic position sizing system independently
"""

import asyncio
import logging
import yaml
from src.core.dynamic_position_sizer import DynamicPositionSizer

class MockBackpackClient:
    """Mock Backpack client for testing"""
    
    def __init__(self, balance: float = 100):
        self.balance = balance
    
    async def get_balances(self):
        """Mock balance response"""
        return {
            'data': [
                {
                    'coin': 'USDC',
                    'available': str(self.balance)
                }
            ]
        }
    
    async def get_positions(self):
        """Mock positions response"""
        return {
            'data': []  # No open positions
        }

async def test_dynamic_position_sizer():
    """Test dynamic position sizer with different balance amounts"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open('config/enhanced_nanpin_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Test different balance scenarios
    test_scenarios = [
        {"balance": 100, "name": "Small Account ($100)"},
        {"balance": 500, "name": "Medium Account ($500)"},
        {"balance": 1000, "name": "Large Account ($1000)"},
        {"balance": 2000, "name": "XL Account ($2000)"},
        {"balance": 50, "name": "Minimum Account ($50)"},
        {"balance": 25, "name": "Below Minimum ($25)"}
    ]
    
    print("🧮 DYNAMIC POSITION SIZER TEST RESULTS")
    print("=" * 60)
    print()
    
    for scenario in test_scenarios:
        balance = scenario["balance"]
        name = scenario["name"]
        
        print(f"📊 {name}")
        print("-" * 40)
        
        # Create mock client
        mock_client = MockBackpackClient(balance)
        
        # Create position sizer
        position_sizer = DynamicPositionSizer(mock_client, config)
        
        try:
            # Calculate recommendation
            recommendation = await position_sizer.calculate_dynamic_position_size()
            
            if recommendation:
                print(f"Balance: ${balance}")
                print(f"Recommended Leverage: {recommendation.leverage}x")
                print(f"Base Margin: ${recommendation.base_margin:.2f}")
                print(f"Position Value: ${recommendation.position_value:.2f}")
                print(f"Max Levels: {recommendation.max_levels}")
                print(f"Scaling Multiplier: {recommendation.scaling_multiplier:.2f}")
                print(f"Capital Usage: {recommendation.capital_usage_pct:.1f}%")
                print(f"Risk Level: {recommendation.risk_level}")
                print(f"Reasoning: {recommendation.reasoning}")
                
                # Calculate Backpack settings
                print()
                print("Backpack Settings:")
                print(f"  Symbol: BTC_USDC_PERP")
                print(f"  Leverage: {recommendation.leverage}x")
                print(f"  Position Size: ${recommendation.base_margin:.2f} margin")
                print(f"  Position Value: ${recommendation.position_value:.2f}")
                
                # Calculate scaling sequence
                print()
                print("Nanpin Scaling Sequence:")
                total_margin = 0
                current_margin = recommendation.base_margin
                
                for level in range(min(recommendation.max_levels, 5)):  # Show first 5 levels
                    position_value = current_margin * recommendation.leverage
                    total_margin += current_margin
                    print(f"  Level {level+1}: ${current_margin:.2f} margin → ${position_value:.2f} position")
                    current_margin *= recommendation.scaling_multiplier
                
                print(f"  Total Margin Used: ${total_margin:.2f} ({(total_margin/balance)*100:.1f}%)")
                
                # Performance projections
                print()
                print("Performance Projections:")
                daily_return_pct = 0.9  # From our analysis
                monthly_return = (recommendation.base_margin * recommendation.leverage) * (daily_return_pct/100) * 30
                print(f"  Estimated Monthly Profit: ${monthly_return:.2f}")
                print(f"  Estimated Monthly ROI: {(monthly_return/balance)*100:.1f}%")
                
            else:
                print(f"❌ No recommendation (balance too low or error)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
        print()

async def main():
    """Main test function"""
    try:
        await test_dynamic_position_sizer()
        
        print("✅ Dynamic Position Sizer test completed!")
        print()
        print("💡 To use these settings:")
        print("1. Set your Backpack leverage to the recommended value")
        print("2. The bot will automatically adjust position sizes")
        print("3. Monitor your balance - sizes will update hourly")
        print("4. Risk levels will adjust as your balance grows/shrinks")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
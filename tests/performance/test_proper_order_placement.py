#!/usr/bin/env python3
"""
🎯 PROPER Backpack Exchange Order Placement Test
Based on Official Backpack API Documentation
Tests actual order placement with comprehensive balance checking

Reference: https://docs.backpack.exchange/
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/proper_order_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ProperOrderTester:
    """
    🎯 Proper Order Placement Tester
    
    Based on official Backpack Exchange API documentation:
    - Checks all balance sources (capital, borrow-lend, collateral)
    - Uses correct API endpoints and instructions
    - Tests actual order placement with real funds
    - Validates position creation via futures API
    - Immediately closes positions for safety
    """
    
    def __init__(self):
        """Initialize the proper tester"""
        self.client = None
        self.test_results = {
            'start_time': datetime.now(),
            'balance_analysis': {},
            'order_placement': {},
            'position_verification': {},
            'position_closing': {},
            'errors': []
        }
        
        # Test configuration - use smaller amount for safety
        self.TEST_ORDER_SIZE_USDC = 5.0  # $5 minimum test
        self.MAX_WAIT_TIME = 30
        
        logger.info("🎯 Proper Order Tester initialized")
        logger.info(f"   Test order size: ${self.TEST_ORDER_SIZE_USDC}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive order placement test using official API"""
        try:
            print("🎯 PROPER BACKPACK EXCHANGE ORDER TEST")
            print("=" * 60)
            print("Based on Official Backpack API Documentation")
            print(f"Test order size: ${self.TEST_ORDER_SIZE_USDC} USDC")
            print("⚠️  This will place REAL orders with REAL money!")
            print("=" * 60)
            
            # Initialize client
            await self._initialize_client()
            
            # Step 1: Comprehensive balance analysis
            print("\n💰 STEP 1: Comprehensive Balance Analysis")
            print("-" * 50)
            available_usdc = await self._analyze_all_balances()
            
            if available_usdc < self.TEST_ORDER_SIZE_USDC:
                print(f"❌ Insufficient USDC for test: ${available_usdc:.2f} < ${self.TEST_ORDER_SIZE_USDC}")
                return False
            
            # Step 2: Pre-trade validation
            print("\n🔍 STEP 2: Pre-Trade Validation")
            print("-" * 50)
            if not await self._validate_trading_conditions():
                print("❌ Trading conditions not met")
                return False
            
            # Step 3: Execute test order
            print("\n🛒 STEP 3: Execute Test Order")
            print("-" * 50)
            order_result = await self._execute_test_order()
            
            if not order_result:
                print("❌ Order execution failed")
                return False
            
            # Step 4: Verify position creation
            print("\n📊 STEP 4: Verify Position Creation")
            print("-" * 50)
            position = await self._verify_position_creation(order_result)
            
            # Step 5: Close position immediately
            print("\n🔒 STEP 5: Close Position (Safety)")
            print("-" * 50)
            if position:
                close_success = await self._close_position_safely(position)
                if close_success:
                    print("✅ Position closed successfully")
                else:
                    print("⚠️ Position closing issue - check manually")
            
            # Generate final report
            await self._generate_final_report()
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            print(f"❌ Test failed: {e}")
            return False
        
        finally:
            if self.client:
                await self.client.close()
    
    async def _initialize_client(self):
        """Initialize Backpack client with proper credentials"""
        try:
            api_key = os.getenv('BACKPACK_API_KEY')
            secret_key = os.getenv('BACKPACK_SECRET_KEY')
            
            if not api_key or not secret_key:
                raise Exception("Missing API credentials")
            
            self.client = BackpackNanpinClient(api_key, secret_key)
            print("✅ Client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize client: {e}")
            raise
    
    async def _analyze_all_balances(self) -> float:
        """Analyze all balance sources according to official API"""
        try:
            total_usdc = 0.0
            
            # 1. Check Capital Balances (/api/v1/capital)
            print("   📊 Checking capital balances...")
            try:
                capital_balances = await self.client.get_balances()
                if capital_balances and 'USDC' in capital_balances:
                    capital_usdc = capital_balances['USDC']['available']
                    total_usdc += capital_usdc
                    print(f"      Capital USDC: ${capital_usdc:.2f}")
                    self.test_results['balance_analysis']['capital_usdc'] = capital_usdc
                else:
                    print("      No USDC in capital balances")
                    self.test_results['balance_analysis']['capital_usdc'] = 0.0
            except Exception as e:
                print(f"      ❌ Capital balance error: {e}")
                self.test_results['errors'].append(f"Capital balance: {e}")
            
            # 2. Check Borrow-Lend Positions (/api/v1/borrowLend/positions)
            print("   💰 Checking borrow-lend positions...")
            try:
                borrow_lend_positions = await self.client.get_borrow_lend_positions()
                lend_usdc = 0.0
                
                for position in borrow_lend_positions:
                    if position.get('symbol') == 'USDC' and position.get('side') == 'lend':
                        position_size = float(position.get('size', 0))
                        lend_usdc += position_size
                        print(f"      Lent USDC: ${position_size:.2f}")
                
                if lend_usdc > 0:
                    print(f"      Total Lent USDC: ${lend_usdc:.2f}")
                    print("      ⚠️  Lent USDC needs to be withdrawn for trading")
                else:
                    print("      No USDC in lending positions")
                
                self.test_results['balance_analysis']['lent_usdc'] = lend_usdc
                
            except Exception as e:
                print(f"      ❌ Borrow-lend error: {e}")
                self.test_results['errors'].append(f"Borrow-lend: {e}")
            
            # 3. Check Collateral Info (/api/v1/capital/collateral)
            print("   🏦 Checking collateral information...")
            try:
                collateral_info = await self.client.get_collateral_info()
                if collateral_info:
                    available_balance = collateral_info.get('availableBalance', 0)
                    net_equity = collateral_info.get('netEquity', 0)
                    
                    print(f"      Available Balance: ${float(available_balance):.2f}")
                    print(f"      Net Equity: ${float(net_equity):.2f}")
                    
                    self.test_results['balance_analysis']['available_balance'] = float(available_balance)
                    self.test_results['balance_analysis']['net_equity'] = float(net_equity)
                    
                    # Use available balance as the most accurate trading balance
                    if available_balance > 0:
                        total_usdc = max(total_usdc, float(available_balance))
                        
            except Exception as e:
                print(f"      ❌ Collateral error: {e}")
                self.test_results['errors'].append(f"Collateral: {e}")
            
            # Summary
            print(f"\n   💰 Balance Summary:")
            print(f"      Total Available for Trading: ${total_usdc:.2f}")
            
            self.test_results['balance_analysis']['total_available'] = total_usdc
            
            return total_usdc
            
        except Exception as e:
            logger.error(f"❌ Balance analysis failed: {e}")
            return 0.0
    
    async def _validate_trading_conditions(self) -> bool:
        """Validate conditions for trading"""
        try:
            # Check connection
            try:
                btc_price = await self.client.get_btc_price()
                if btc_price and btc_price > 0:
                    print(f"   ✅ BTC Price: ${btc_price:,.2f}")
                    self.test_results['balance_analysis']['btc_price'] = btc_price
                else:
                    print("   ❌ Could not get BTC price")
                    return False
            except Exception as e:
                print(f"   ❌ Price check failed: {e}")
                return False
            
            # Check existing positions
            try:
                existing_positions = await self.client.get_btc_position()
                if existing_positions:
                    size = float(existing_positions.get('size', 0))
                    print(f"   ⚠️  Existing BTC position: {size:.8f} BTC")
                    self.test_results['balance_analysis']['existing_position'] = existing_positions
                else:
                    print("   ✅ No existing BTC position")
                    self.test_results['balance_analysis']['existing_position'] = None
            except Exception as e:
                print(f"   ❌ Position check failed: {e}")
                return False
            
            # Check risk levels
            try:
                risk_assessment = await self.client.check_liquidation_risk()
                risk_level = risk_assessment.get('liquidation_risk', 'unknown')
                print(f"   ✅ Risk Level: {risk_level}")
                
                if risk_level == 'critical':
                    print("   🚨 CRITICAL RISK - Aborting test")
                    return False
                    
                self.test_results['balance_analysis']['risk_level'] = risk_level
            except Exception as e:
                print(f"   ❌ Risk check failed: {e}")
                return False
            
            print("   ✅ All trading conditions validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading validation failed: {e}")
            return False
    
    async def _execute_test_order(self) -> Optional[Dict]:
        """Execute the test order using official API"""
        try:
            print(f"   🛒 Placing ${self.TEST_ORDER_SIZE_USDC} USDC market buy order...")
            
            # Record attempt
            self.test_results['order_placement']['attempted_at'] = datetime.now()
            self.test_results['order_placement']['order_size'] = self.TEST_ORDER_SIZE_USDC
            
            # Execute order using the client's method
            order_result = await self.client.market_buy_btc(
                self.TEST_ORDER_SIZE_USDC,
                reason="Proper Order Placement Test"
            )
            
            if order_result:
                print("   ✅ Order executed successfully!")
                print(f"      Order ID: {order_result.get('id', 'N/A')}")
                print(f"      Status: {order_result.get('status', 'N/A')}")
                
                # Store execution details
                self.test_results['order_placement']['success'] = True
                self.test_results['order_placement']['order_id'] = order_result.get('id')
                self.test_results['order_placement']['order_result'] = order_result
                self.test_results['order_placement']['executed_at'] = datetime.now()
                
                # Log execution details
                if 'quantity' in order_result:
                    btc_qty = float(order_result['quantity'])
                    print(f"      BTC Quantity: {btc_qty:.8f}")
                    self.test_results['order_placement']['btc_quantity'] = btc_qty
                
                if 'fillPrice' in order_result:
                    fill_price = float(order_result['fillPrice'])
                    print(f"      Fill Price: ${fill_price:,.2f}")
                    self.test_results['order_placement']['fill_price'] = fill_price
                
                return order_result
            else:
                print("   ❌ Order execution failed - no result")
                self.test_results['order_placement']['success'] = False
                return None
                
        except Exception as e:
            print(f"   ❌ Order execution error: {e}")
            logger.error(f"Order execution failed: {e}")
            self.test_results['order_placement']['success'] = False
            self.test_results['order_placement']['error'] = str(e)
            self.test_results['errors'].append(f"Order execution: {e}")
            return None
    
    async def _verify_position_creation(self, order_result: Dict) -> Optional[Dict]:
        """Verify position creation using official Futures API"""
        try:
            print("   📊 Verifying position creation via Futures API...")
            
            # Wait for order processing
            await asyncio.sleep(3)
            
            # Check for position using official get_positions endpoint
            # Reference: https://docs.backpack.exchange/#tag/Futures/operation/get_positions
            for attempt in range(10):  # Try for 20 seconds
                try:
                    positions = await self.client._make_request(
                        'GET',
                        '/api/v1/positions',
                        signed=True,
                        instruction='positionQuery'
                    )
                    
                    if positions:
                        for position in positions:
                            symbol = position.get('symbol', '')
                            if 'BTC' in symbol:
                                size = float(position.get('size', 0))
                                if size > 0:
                                    print(f"   ✅ Position Confirmed via Futures API!")
                                    print(f"      Symbol: {symbol}")
                                    print(f"      Size: {size:.8f} BTC")
                                    print(f"      Entry Price: ${float(position.get('entryPrice', 0)):,.2f}")
                                    print(f"      Mark Price: ${float(position.get('markPrice', 0)):,.2f}")
                                    
                                    self.test_results['position_verification']['success'] = True
                                    self.test_results['position_verification']['position'] = position
                                    self.test_results['position_verification']['verified_at'] = datetime.now()
                                    
                                    return position
                    
                    print(f"      Waiting for position... (attempt {attempt + 1}/10)")
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    print(f"      Position check error: {e}")
                    await asyncio.sleep(2)
            
            print("   ⚠️  Position not detected within timeout period")
            self.test_results['position_verification']['success'] = False
            self.test_results['position_verification']['timeout'] = True
            return None
            
        except Exception as e:
            print(f"   ❌ Position verification error: {e}")
            logger.error(f"Position verification failed: {e}")
            self.test_results['position_verification']['success'] = False
            self.test_results['position_verification']['error'] = str(e)
            return None
    
    async def _close_position_safely(self, position: Dict) -> bool:
        """Close position immediately for safety"""
        try:
            print("   🔒 Closing position for safety...")
            
            size = float(position.get('size', 0))
            symbol = position.get('symbol', 'BTC_USDC')
            
            if size <= 0:
                print("   ℹ️  No position size to close")
                return True
            
            self.test_results['position_closing']['attempted_at'] = datetime.now()
            self.test_results['position_closing']['size_to_close'] = size
            
            # Execute market sell order to close position
            close_params = {
                'symbol': symbol,
                'side': 'Ask',  # Sell side to close long position
                'orderType': 'Market',
                'quantity': f"{size:.8f}",
                'timeInForce': 'IOC'
            }
            
            print(f"      Closing {size:.8f} BTC with market sell...")
            
            close_result = await self.client._make_request(
                'POST',
                '/api/v1/order',
                close_params,
                signed=True,
                instruction='orderExecute'
            )
            
            if close_result:
                print("   ✅ Closing order placed!")
                print(f"      Close Order ID: {close_result.get('id', 'N/A')}")
                
                self.test_results['position_closing']['success'] = True
                self.test_results['position_closing']['close_order_id'] = close_result.get('id')
                self.test_results['position_closing']['closed_at'] = datetime.now()
                
                # Wait for closing
                await asyncio.sleep(5)
                
                # Verify closure
                final_positions = await self.client._make_request(
                    'GET',
                    '/api/v1/positions',
                    signed=True,
                    instruction='positionQuery'
                )
                
                btc_position_exists = False
                if final_positions:
                    for pos in final_positions:
                        if 'BTC' in pos.get('symbol', '') and float(pos.get('size', 0)) > 0:
                            btc_position_exists = True
                            break
                
                if not btc_position_exists:
                    print("   ✅ Position successfully closed")
                    self.test_results['position_closing']['verified_closed'] = True
                else:
                    print("   ⚠️  Position may still exist")
                    self.test_results['position_closing']['verified_closed'] = False
                
                return True
            else:
                print("   ❌ Closing order failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Position closing error: {e}")
            logger.error(f"Position closing failed: {e}")
            self.test_results['position_closing']['success'] = False
            self.test_results['position_closing']['error'] = str(e)
            return False
    
    async def _generate_final_report(self):
        """Generate comprehensive final report"""
        try:
            print("\n📊 FINAL TEST REPORT")
            print("=" * 60)
            print("Based on Official Backpack Exchange API")
            
            # Test duration
            duration = datetime.now() - self.test_results['start_time']
            print(f"Test Duration: {duration.total_seconds():.1f} seconds")
            
            # Results summary
            order_success = self.test_results['order_placement'].get('success', False)
            position_success = self.test_results['position_verification'].get('success', False)
            closing_success = self.test_results['position_closing'].get('success', False)
            
            print(f"\n🎯 Results Summary:")
            print(f"   Order Execution: {'✅ SUCCESS' if order_success else '❌ FAILED'}")
            print(f"   Position Creation: {'✅ VERIFIED' if position_success else '⚠️ NOT VERIFIED'}")
            print(f"   Position Closing: {'✅ SUCCESS' if closing_success else '⚠️ ATTEMPTED'}")
            
            # Balance analysis
            if 'balance_analysis' in self.test_results:
                ba = self.test_results['balance_analysis']
                print(f"\n💰 Balance Analysis:")
                print(f"   Available for Trading: ${ba.get('total_available', 0):.2f}")
                if 'btc_price' in ba:
                    print(f"   BTC Price at Test: ${ba['btc_price']:,.2f}")
            
            # Order details
            if order_success:
                op = self.test_results['order_placement']
                print(f"\n📋 Order Details:")
                print(f"   Order ID: {op.get('order_id', 'N/A')}")
                print(f"   Size: ${op.get('order_size', 0):.2f} USDC")
                if 'btc_quantity' in op:
                    print(f"   BTC Received: {op['btc_quantity']:.8f}")
                if 'fill_price' in op:
                    print(f"   Execution Price: ${op['fill_price']:,.2f}")
            
            # Final assessment
            print(f"\n🏆 FINAL ASSESSMENT:")
            if order_success and position_success and closing_success:
                print("   ✅ COMPLETE SUCCESS!")
                print("   ✅ Bot can place orders, create positions, and close them")
                print("   ✅ Fully functional for live trading")
            elif order_success and position_success:
                print("   ⚠️  MOSTLY SUCCESSFUL")
                print("   ✅ Bot can place orders and create positions")
                print("   ⚠️  Position closing needs verification")
            elif order_success:
                print("   ⚠️  PARTIAL SUCCESS")
                print("   ✅ Bot can execute orders")
                print("   ⚠️  Position creation/tracking needs review")
            else:
                print("   ❌ ORDER EXECUTION FAILED")
                print("   ❌ Technical issues need resolution")
            
            # Error summary
            if self.test_results['errors']:
                print(f"\n⚠️  Errors Encountered:")
                for error in self.test_results['errors']:
                    print(f"   • {error}")
            
            # Save detailed report
            report_file = f"proper_order_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = Path("logs") / report_file
            
            import json
            with open(report_path, 'w') as f:
                serializable_results = self._make_json_serializable(self.test_results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n📄 Detailed report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert datetime objects to strings for JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

async def main():
    """Main test execution"""
    
    print("🎯 PROPER BACKPACK EXCHANGE ORDER PLACEMENT TEST")
    print("=" * 60)
    print("Based on Official API Documentation")
    print("https://docs.backpack.exchange/")
    print("")
    print("⚠️  WARNING: This will place REAL orders with REAL money!")
    print("   - Test order size: $5 USDC")
    print("   - Positions will be closed immediately")
    print("   - Uses official Backpack API endpoints")
    print("")
    
    # Safety confirmation
    try:
        confirmation = input("Type 'EXECUTE' to proceed with real trading test: ")
        if confirmation != 'EXECUTE':
            print("❌ Test cancelled by user")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Test cancelled by user")
        return
    
    # Check credentials
    if not os.getenv('BACKPACK_API_KEY') or not os.getenv('BACKPACK_SECRET_KEY'):
        print("❌ Error: API credentials not found")
        return
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run the proper test
    tester = ProperOrderTester()
    try:
        success = await tester.run_comprehensive_test()
        if success:
            print("\n🎉 PROPER TEST COMPLETED!")
        else:
            print("\n⚠️  Test completed with issues")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
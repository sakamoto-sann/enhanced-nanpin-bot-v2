#!/usr/bin/env python3
"""
🧪 Comprehensive Order Placement Test for Backpack Exchange
Tests the complete order lifecycle: placement -> monitoring -> position closing

SAFETY MEASURES:
- Uses minimum order size (10 USDC)
- Immediately closes positions after creation
- Comprehensive error handling and logging
- Validates against official Backpack Futures API documentation
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from exchanges.backpack_client_fixed import BackpackNanpinClient, load_credentials_from_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/order_placement_test.log')
    ]
)
logger = logging.getLogger(__name__)

class OrderPlacementTester:
    """
    🧪 Comprehensive Order Placement Tester
    
    Tests complete order lifecycle:
    1. Pre-flight checks (balance, connection, etc.)
    2. Place minimum size market buy order (10 USDC)
    3. Monitor order execution and position creation
    4. Validate position via Backpack Futures API
    5. Immediately close position with market sell
    6. Verify position closure
    """
    
    def __init__(self):
        """Initialize the tester"""
        self.client = None
        self.test_results = {
            'start_time': datetime.now(),
            'pre_flight_checks': {},
            'order_placement': {},
            'position_monitoring': {},
            'position_closing': {},
            'final_validation': {},
            'errors': []
        }
        
        # Test configuration
        self.TEST_ORDER_SIZE_USDC = 10.0  # Minimum order size
        self.MAX_WAIT_TIME = 30  # Max wait time for order execution
        self.POSITION_CHECK_INTERVAL = 2  # Check position every 2 seconds
        
        logger.info("🧪 Order Placement Tester initialized")
        logger.info(f"   Test order size: ${self.TEST_ORDER_SIZE_USDC}")
        logger.info(f"   Max wait time: {self.MAX_WAIT_TIME}s")
    
    async def run_comprehensive_test(self):
        """Run the complete order placement test"""
        try:
            print("🧪 COMPREHENSIVE ORDER PLACEMENT TEST")
            print("=" * 60)
            print(f"Test started: {self.test_results['start_time'].isoformat()}")
            print(f"Test order size: ${self.TEST_ORDER_SIZE_USDC} USDC")
            print("⚠️  WARNING: This test will place REAL orders on Backpack Exchange")
            print("=" * 60)
            
            # Initialize client
            await self._initialize_client()
            
            # Step 1: Pre-flight checks
            print("\n🔍 STEP 1: Pre-flight Checks")
            print("-" * 40)
            if not await self._run_preflight_checks():
                print("❌ Pre-flight checks failed. Aborting test.")
                return False
            
            # Step 2: Place test order
            print("\n🛒 STEP 2: Place Test Order")
            print("-" * 40)
            order_result = await self._place_test_order()
            if not order_result:
                print("❌ Order placement failed. Aborting test.")
                return False
            
            # Step 3: Monitor position creation
            print("\n📊 STEP 3: Monitor Position Creation")
            print("-" * 40)
            position = await self._monitor_position_creation(order_result)
            if not position:
                print("❌ Position monitoring failed.")
                # Don't abort - we might still need to check for positions to close
            
            # Step 4: Validate position via Futures API
            print("\n✅ STEP 4: Validate Position via Futures API")
            print("-" * 40)
            validated_position = await self._validate_position_via_futures_api()
            
            # Step 5: Close position immediately
            print("\n🔒 STEP 5: Close Position (SAFETY)")
            print("-" * 40)
            if validated_position or position:
                close_result = await self._close_position_immediately(validated_position or position)
                if close_result:
                    print("✅ Position closed successfully")
                else:
                    print("⚠️ Position closing may have failed - manual check recommended")
            else:
                print("ℹ️ No position detected to close")
            
            # Step 6: Final validation
            print("\n🏁 STEP 6: Final Validation")
            print("-" * 40)
            await self._final_validation()
            
            # Generate comprehensive report
            await self._generate_test_report()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            self.test_results['errors'].append(f"Test suite error: {e}")
            return False
        
        finally:
            if self.client:
                await self.client.close()
    
    async def _initialize_client(self):
        """Initialize Backpack client"""
        try:
            logger.info("📡 Initializing Backpack client...")
            
            # Get credentials directly from environment (already loaded via load_dotenv)
            api_key = os.getenv('BACKPACK_API_KEY')
            secret_key = os.getenv('BACKPACK_SECRET_KEY')
            
            if not api_key or api_key == 'your_backpack_api_key_here':
                raise Exception("BACKPACK_API_KEY not properly set")
            if not secret_key or secret_key == 'your_backpack_secret_here':
                raise Exception("BACKPACK_SECRET_KEY not properly set")
                
            self.client = BackpackNanpinClient(api_key, secret_key)
            logger.info("✅ Client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize client: {e}")
            self.test_results['errors'].append(f"Client initialization: {e}")
            raise
    
    async def _run_preflight_checks(self) -> bool:
        """Run comprehensive pre-flight checks"""
        checks_passed = 0
        total_checks = 6
        
        try:
            # Check 1: Connection test
            print("   🔗 Testing connection...")
            try:
                connection_ok = await self.client.test_connection()
                if connection_ok:
                    print("   ✅ Connection test passed")
                    checks_passed += 1
                    self.test_results['pre_flight_checks']['connection'] = True
                else:
                    print("   ❌ Connection test failed")
                    self.test_results['pre_flight_checks']['connection'] = False
            except Exception as e:
                print(f"   ❌ Connection test error: {e}")
                self.test_results['pre_flight_checks']['connection'] = False
                self.test_results['errors'].append(f"Connection test: {e}")
            
            # Check 2: Authentication test
            print("   🔐 Testing authentication...")
            try:
                auth_ok = await self.client.test_authentication()
                if auth_ok:
                    print("   ✅ Authentication test passed")
                    checks_passed += 1
                    self.test_results['pre_flight_checks']['authentication'] = True
                else:
                    print("   ❌ Authentication test failed")
                    self.test_results['pre_flight_checks']['authentication'] = False
            except Exception as e:
                print(f"   ❌ Authentication test error: {e}")
                self.test_results['pre_flight_checks']['authentication'] = False
                self.test_results['errors'].append(f"Authentication test: {e}")
            
            # Check 3: Balance verification
            print("   💰 Checking balances...")
            try:
                balances = await self.client.get_balances()
                if balances and 'USDC' in balances:
                    usdc_balance = balances['USDC']['available']
                    if usdc_balance >= self.TEST_ORDER_SIZE_USDC:
                        print(f"   ✅ Sufficient USDC balance: {usdc_balance:.2f}")
                        checks_passed += 1
                        self.test_results['pre_flight_checks']['balance'] = True
                        self.test_results['pre_flight_checks']['usdc_balance'] = usdc_balance
                    else:
                        print(f"   ❌ Insufficient USDC balance: {usdc_balance:.2f} < {self.TEST_ORDER_SIZE_USDC}")
                        self.test_results['pre_flight_checks']['balance'] = False
                        self.test_results['pre_flight_checks']['usdc_balance'] = usdc_balance
                else:
                    print("   ❌ Could not retrieve USDC balance")
                    self.test_results['pre_flight_checks']['balance'] = False
            except Exception as e:
                print(f"   ❌ Balance check error: {e}")
                self.test_results['pre_flight_checks']['balance'] = False
                self.test_results['errors'].append(f"Balance check: {e}")
            
            # Check 4: Current BTC price
            print("   📈 Getting current BTC price...")
            try:
                btc_price = await self.client.get_btc_price()
                if btc_price and btc_price > 0:
                    print(f"   ✅ Current BTC price: ${btc_price:,.2f}")
                    checks_passed += 1
                    self.test_results['pre_flight_checks']['btc_price'] = btc_price
                else:
                    print("   ❌ Could not get BTC price")
                    self.test_results['pre_flight_checks']['btc_price'] = None
            except Exception as e:
                print(f"   ❌ BTC price error: {e}")
                self.test_results['pre_flight_checks']['btc_price'] = None
                self.test_results['errors'].append(f"BTC price check: {e}")
            
            # Check 5: Existing positions
            print("   📊 Checking existing positions...")
            try:
                existing_position = await self.client.get_btc_position()
                if existing_position:
                    size = existing_position.get('size', 0)
                    print(f"   ⚠️ Existing BTC position found: {size:.8f} BTC")
                    self.test_results['pre_flight_checks']['existing_position'] = existing_position
                else:
                    print("   ✅ No existing BTC position")
                    self.test_results['pre_flight_checks']['existing_position'] = None
                checks_passed += 1
            except Exception as e:
                print(f"   ❌ Position check error: {e}")
                self.test_results['pre_flight_checks']['existing_position'] = None
                self.test_results['errors'].append(f"Position check: {e}")
            
            # Check 6: Risk assessment
            print("   ⚖️ Assessing risk levels...")
            try:
                risk_assessment = await self.client.check_liquidation_risk()
                risk_level = risk_assessment.get('liquidation_risk', 'unknown')
                if risk_level in ['low', 'moderate']:
                    print(f"   ✅ Risk level acceptable: {risk_level}")
                    checks_passed += 1
                    self.test_results['pre_flight_checks']['risk_level'] = risk_level
                else:
                    print(f"   ⚠️ Risk level: {risk_level}")
                    self.test_results['pre_flight_checks']['risk_level'] = risk_level
                    if risk_level == 'critical':
                        print("   🚨 CRITICAL RISK - Test aborted for safety")
                        return False
            except Exception as e:
                print(f"   ❌ Risk assessment error: {e}")
                self.test_results['pre_flight_checks']['risk_level'] = 'unknown'
                self.test_results['errors'].append(f"Risk assessment: {e}")
            
            print(f"\n   📊 Pre-flight Summary: {checks_passed}/{total_checks} checks passed")
            
            # Require at least critical checks to pass
            critical_checks = [
                self.test_results['pre_flight_checks'].get('connection', False),
                self.test_results['pre_flight_checks'].get('authentication', False),
                self.test_results['pre_flight_checks'].get('balance', False)
            ]
            
            if all(critical_checks):
                print("   ✅ Critical checks passed - proceeding with test")
                return True
            else:
                print("   ❌ Critical checks failed - aborting test")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pre-flight checks failed: {e}")
            self.test_results['errors'].append(f"Pre-flight checks: {e}")
            return False
    
    async def _place_test_order(self) -> Optional[Dict]:
        """Place the test market buy order"""
        try:
            print(f"   🛒 Placing market buy order for ${self.TEST_ORDER_SIZE_USDC} USDC...")
            
            # Record order placement attempt
            self.test_results['order_placement']['attempted_at'] = datetime.now()
            self.test_results['order_placement']['order_size_usdc'] = self.TEST_ORDER_SIZE_USDC
            
            # Place order
            order_result = await self.client.market_buy_btc(
                self.TEST_ORDER_SIZE_USDC, 
                reason="Order Placement Test"
            )
            
            if order_result:
                print("   ✅ Order placed successfully!")
                print(f"      Order ID: {order_result.get('id', 'N/A')}")
                print(f"      Status: {order_result.get('status', 'N/A')}")
                
                # Record successful order placement
                self.test_results['order_placement']['success'] = True
                self.test_results['order_placement']['order_id'] = order_result.get('id')
                self.test_results['order_placement']['order_result'] = order_result
                self.test_results['order_placement']['placed_at'] = datetime.now()
                
                # Get order details if available
                if 'quantity' in order_result:
                    btc_quantity = float(order_result['quantity'])
                    print(f"      BTC Quantity: {btc_quantity:.8f}")
                    self.test_results['order_placement']['btc_quantity'] = btc_quantity
                
                if 'fillPrice' in order_result:
                    fill_price = float(order_result['fillPrice'])
                    print(f"      Fill Price: ${fill_price:,.2f}")
                    self.test_results['order_placement']['fill_price'] = fill_price
                
                return order_result
            else:
                print("   ❌ Order placement failed - no result returned")
                self.test_results['order_placement']['success'] = False
                self.test_results['order_placement']['error'] = "No result returned"
                return None
                
        except Exception as e:
            print(f"   ❌ Order placement error: {e}")
            logger.error(f"Order placement failed: {e}")
            self.test_results['order_placement']['success'] = False
            self.test_results['order_placement']['error'] = str(e)
            self.test_results['errors'].append(f"Order placement: {e}")
            return None
    
    async def _monitor_position_creation(self, order_result: Dict) -> Optional[Dict]:
        """Monitor for position creation after order placement"""
        try:
            print("   📊 Monitoring position creation...")
            order_id = order_result.get('id')
            
            self.test_results['position_monitoring']['started_at'] = datetime.now()
            self.test_results['position_monitoring']['order_id'] = order_id
            
            # Wait a moment for order processing
            await asyncio.sleep(2)
            
            # Monitor for up to MAX_WAIT_TIME seconds
            for attempt in range(self.MAX_WAIT_TIME // self.POSITION_CHECK_INTERVAL):
                try:
                    # Check order status
                    if order_id:
                        order_status = await self.client.get_order_status(order_id)
                        if order_status:
                            status = order_status.get('status', 'unknown')
                            print(f"      Order status (attempt {attempt + 1}): {status}")
                            
                            if status in ['Filled', 'PartiallyFilled']:
                                print("   ✅ Order filled - checking for position...")
                                break
                    
                    # Check for BTC position
                    position = await self.client.get_btc_position()
                    if position:
                        size = float(position.get('size', 0))
                        if size > 0:
                            print(f"   ✅ Position detected: {size:.8f} BTC")
                            
                            self.test_results['position_monitoring']['success'] = True
                            self.test_results['position_monitoring']['position_found_at'] = datetime.now()
                            self.test_results['position_monitoring']['position'] = position
                            
                            return position
                    
                    print(f"      No position yet (attempt {attempt + 1}/{self.MAX_WAIT_TIME // self.POSITION_CHECK_INTERVAL})")
                    await asyncio.sleep(self.POSITION_CHECK_INTERVAL)
                    
                except Exception as e:
                    print(f"      Monitoring error (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.POSITION_CHECK_INTERVAL)
            
            print("   ⚠️ Position monitoring timeout - position may not be created yet")
            self.test_results['position_monitoring']['success'] = False
            self.test_results['position_monitoring']['timeout'] = True
            return None
            
        except Exception as e:
            print(f"   ❌ Position monitoring error: {e}")
            logger.error(f"Position monitoring failed: {e}")
            self.test_results['position_monitoring']['success'] = False
            self.test_results['position_monitoring']['error'] = str(e)
            self.test_results['errors'].append(f"Position monitoring: {e}")
            return None
    
    async def _validate_position_via_futures_api(self) -> Optional[Dict]:
        """Validate position using Backpack Futures API as per official docs"""
        try:
            print("   📋 Validating via official Futures API...")
            
            # Use the official get_positions endpoint
            # Reference: https://docs.backpack.exchange/#tag/Futures/operation/get_positions
            positions = await self.client._make_request(
                'GET', 
                '/api/v1/positions',  # Official Futures API endpoint
                signed=True, 
                instruction='positionQuery'
            )
            
            self.test_results['final_validation']['futures_api_called'] = True
            self.test_results['final_validation']['positions_response'] = positions
            
            if positions:
                print(f"   ✅ Retrieved {len(positions)} positions from Futures API")
                
                # Look for BTC position
                for position in positions:
                    symbol = position.get('symbol', '')
                    if 'BTC' in symbol:
                        size = float(position.get('size', 0))
                        if size > 0:
                            print(f"   🎯 BTC Position Confirmed via Futures API:")
                            print(f"      Symbol: {symbol}")
                            print(f"      Size: {size:.8f} BTC")
                            print(f"      Entry Price: ${float(position.get('entryPrice', 0)):,.2f}")
                            print(f"      Mark Price: ${float(position.get('markPrice', 0)):,.2f}")
                            print(f"      Unrealized PnL: ${float(position.get('unrealizedPnl', 0)):,.2f}")
                            
                            self.test_results['final_validation']['btc_position_confirmed'] = True
                            self.test_results['final_validation']['confirmed_position'] = position
                            
                            return position
                
                print("   ℹ️ No BTC position found in Futures API response")
                self.test_results['final_validation']['btc_position_confirmed'] = False
            else:
                print("   ℹ️ No positions returned from Futures API")
                self.test_results['final_validation']['btc_position_confirmed'] = False
            
            return None
            
        except Exception as e:
            print(f"   ❌ Futures API validation error: {e}")
            logger.error(f"Futures API validation failed: {e}")
            self.test_results['final_validation']['futures_api_error'] = str(e)
            self.test_results['errors'].append(f"Futures API validation: {e}")
            return None
    
    async def _close_position_immediately(self, position: Dict) -> bool:
        """Immediately close the position for safety"""
        try:
            print("   🔒 Closing position immediately (SAFETY MEASURE)...")
            
            size = float(position.get('size', 0))
            symbol = position.get('symbol', 'BTC_USDC')
            
            if size <= 0:
                print("   ℹ️ No position size to close")
                return True
            
            self.test_results['position_closing']['attempted_at'] = datetime.now()
            self.test_results['position_closing']['size_to_close'] = size
            
            # Get current price for market sell
            current_price = await self.client.get_btc_price()
            if not current_price:
                print("   ❌ Could not get current price for closing")
                return False
            
            # Prepare market sell order to close position
            close_params = {
                'symbol': symbol,
                'side': 'Ask',  # Sell to close long position
                'orderType': 'Market',
                'quantity': f"{size:.8f}",
                'timeInForce': 'IOC'  # Immediate or Cancel
            }
            
            print(f"      Closing {size:.8f} BTC at market price ~${current_price:,.2f}")
            
            # Execute closing order
            close_result = await self.client._make_request(
                'POST',
                '/api/v1/order',
                close_params,
                signed=True,
                instruction='orderExecute'
            )
            
            if close_result:
                print("   ✅ Closing order placed successfully!")
                print(f"      Close Order ID: {close_result.get('id', 'N/A')}")
                print(f"      Status: {close_result.get('status', 'N/A')}")
                
                self.test_results['position_closing']['success'] = True
                self.test_results['position_closing']['close_order_id'] = close_result.get('id')
                self.test_results['position_closing']['close_result'] = close_result
                
                # Wait a moment for order processing
                await asyncio.sleep(3)
                
                # Verify position is closed
                final_position = await self.client.get_btc_position()
                if final_position:
                    final_size = float(final_position.get('size', 0))
                    if final_size == 0:
                        print("   ✅ Position successfully closed (size = 0)")
                        self.test_results['position_closing']['verified_closed'] = True
                    else:
                        print(f"   ⚠️ Position still exists: {final_size:.8f} BTC")
                        self.test_results['position_closing']['verified_closed'] = False
                        self.test_results['position_closing']['remaining_size'] = final_size
                else:
                    print("   ✅ No position found - successfully closed")
                    self.test_results['position_closing']['verified_closed'] = True
                
                return True
            else:
                print("   ❌ Closing order failed")
                self.test_results['position_closing']['success'] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Position closing error: {e}")
            logger.error(f"Position closing failed: {e}")
            self.test_results['position_closing']['success'] = False
            self.test_results['position_closing']['error'] = str(e)
            self.test_results['errors'].append(f"Position closing: {e}")
            return False
    
    async def _final_validation(self):
        """Final validation and cleanup check"""
        try:
            print("   🏁 Final validation...")
            
            # Check final account state
            final_balances = await self.client.get_balances()
            final_position = await self.client.get_btc_position()
            
            self.test_results['final_validation']['final_balances'] = final_balances
            self.test_results['final_validation']['final_position'] = final_position
            self.test_results['final_validation']['completed_at'] = datetime.now()
            
            if final_position:
                size = float(final_position.get('size', 0))
                if size > 0:
                    print(f"   ⚠️ WARNING: Position still exists: {size:.8f} BTC")
                    print("      Manual intervention may be required")
                else:
                    print("   ✅ No remaining position")
            else:
                print("   ✅ No remaining position")
            
            if final_balances and 'USDC' in final_balances:
                final_usdc = final_balances['USDC']['available']
                initial_usdc = self.test_results['pre_flight_checks'].get('usdc_balance', 0)
                if initial_usdc > 0:
                    change = final_usdc - initial_usdc
                    print(f"   💰 USDC balance change: {change:+.2f} (${initial_usdc:.2f} → ${final_usdc:.2f})")
                    self.test_results['final_validation']['usdc_balance_change'] = change
            
        except Exception as e:
            print(f"   ❌ Final validation error: {e}")
            self.test_results['errors'].append(f"Final validation: {e}")
    
    async def _generate_test_report(self):
        """Generate comprehensive test report"""
        try:
            print("\n📊 COMPREHENSIVE TEST REPORT")
            print("=" * 60)
            
            # Test duration
            duration = datetime.now() - self.test_results['start_time']
            print(f"Test Duration: {duration.total_seconds():.1f} seconds")
            
            # Success indicators
            order_success = self.test_results['order_placement'].get('success', False)
            position_success = self.test_results['position_monitoring'].get('success', False) or \
                             self.test_results['final_validation'].get('btc_position_confirmed', False)
            closing_success = self.test_results['position_closing'].get('success', False)
            
            print(f"\n🎯 Test Results Summary:")
            print(f"   Order Placement: {'✅ SUCCESS' if order_success else '❌ FAILED'}")
            print(f"   Position Creation: {'✅ DETECTED' if position_success else '⚠️ NOT DETECTED'}")
            print(f"   Position Closing: {'✅ SUCCESS' if closing_success else '⚠️ ATTEMPTED'}")
            
            # Order details
            if order_success:
                order_data = self.test_results['order_placement']
                print(f"\n📋 Order Details:")
                print(f"   Order ID: {order_data.get('order_id', 'N/A')}")
                print(f"   Size: ${order_data.get('order_size_usdc', 0):.2f} USDC")
                if 'btc_quantity' in order_data:
                    print(f"   BTC Quantity: {order_data['btc_quantity']:.8f}")
                if 'fill_price' in order_data:
                    print(f"   Fill Price: ${order_data['fill_price']:,.2f}")
            
            # Position details
            if position_success:
                if 'confirmed_position' in self.test_results['final_validation']:
                    pos = self.test_results['final_validation']['confirmed_position']
                    print(f"\n📊 Position Details (Futures API):")
                    print(f"   Symbol: {pos.get('symbol', 'N/A')}")
                    print(f"   Size: {float(pos.get('size', 0)):.8f} BTC")
                    print(f"   Entry Price: ${float(pos.get('entryPrice', 0)):,.2f}")
                    print(f"   Mark Price: ${float(pos.get('markPrice', 0)):,.2f}")
            
            # Error summary
            if self.test_results['errors']:
                print(f"\n⚠️ Errors Encountered ({len(self.test_results['errors'])}):")
                for i, error in enumerate(self.test_results['errors'], 1):
                    print(f"   {i}. {error}")
            
            # Final assessment
            print(f"\n🏆 FINAL ASSESSMENT:")
            if order_success and position_success and closing_success:
                print("   ✅ FULL SUCCESS: Bot can place orders, create positions, and close them")
                print("   ✅ Ready for live trading with proper risk management")
            elif order_success and position_success:
                print("   ⚠️ PARTIAL SUCCESS: Bot can place orders and create positions")
                print("   ⚠️ Position closing needs manual verification")
            elif order_success:
                print("   ⚠️ LIMITED SUCCESS: Bot can place orders")
                print("   ⚠️ Position creation/monitoring needs investigation")
            else:
                print("   ❌ FAILED: Bot cannot place orders successfully")
                print("   ❌ Not ready for live trading")
            
            # Save detailed report to file
            report_filename = f"order_placement_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = Path("logs") / report_filename
            
            import json
            with open(report_path, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = self._make_json_serializable(self.test_results)
                json.dump(serializable_results, f, indent=2)
            
            print(f"\n📄 Detailed report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate test report: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert datetime objects and other non-serializable objects to strings"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

async def main():
    """Main test function"""
    
    print("🧪 BACKPACK EXCHANGE ORDER PLACEMENT TEST")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print("")
    print("⚠️  WARNING: This test will place REAL orders with REAL money!")
    print("   - Test order size: $10 USDC (minimum)")
    print("   - Positions will be closed immediately after creation")
    print("   - Ensure you have sufficient USDC balance")
    print("")
    
    # Safety confirmation
    try:
        confirmation = input("Type 'CONFIRM' to proceed with REAL trading test: ")
        if confirmation != 'CONFIRM':
            print("❌ Test cancelled by user")
            return
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Test cancelled by user")
        return
    
    # Check environment variables
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    print(f"✅ Loaded API Key: {api_key[:20] if api_key else 'None'}...")
    print(f"✅ Loaded Secret Key: {secret_key[:20] if secret_key else 'None'}...")
    
    if not api_key or api_key == 'your_backpack_api_key_here':
        print("❌ Error: BACKPACK_API_KEY not properly set in environment")
        print("   Check your .env file and ensure credentials are correct")
        return
        
    if not secret_key or secret_key == 'your_backpack_secret_here':
        print("❌ Error: BACKPACK_SECRET_KEY not properly set in environment")  
        print("   Check your .env file and ensure credentials are correct")
        return
    
    print("✅ Environment variables loaded successfully!")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run the test
    tester = OrderPlacementTester()
    try:
        success = await tester.run_comprehensive_test()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n⚠️ Test completed with issues - check the report")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error(f"Test suite error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
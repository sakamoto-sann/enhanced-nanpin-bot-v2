#!/usr/bin/env python3
"""
🚀 Backpack Exchange WebSocket Client for Real-time Data
Implements WebSocket streams for instant market data and order updates
"""

import asyncio
import json
import logging
import time
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from urllib.parse import urlencode
import base64
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

class BackpackWebSocketClient:
    """
    🚀 Backpack Exchange WebSocket Client
    
    Features:
    - Real-time price feeds
    - Order book depth streaming  
    - Position and order updates
    - Automatic reconnection
    - ED25519 signature authentication
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """Initialize WebSocket client"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.ws_url = "wss://ws.backpack.exchange"
        
        # Connection management
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.should_reconnect = True
        self.reconnect_interval = 5
        self.ping_interval = 30
        
        # Stream subscriptions
        self.subscriptions: Dict[str, bool] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        # Data storage
        self.latest_prices: Dict[str, float] = {}
        self.order_books: Dict[str, Dict] = {}
        self.account_updates: List[Dict] = []
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'reconnections': 0,
            'last_message_time': None,
            'connection_start': None
        }
        
        logger.info("🚀 Backpack WebSocket client initialized")
        logger.info(f"   WebSocket URL: {self.ws_url}")
        logger.info(f"   Authentication: {'✅ Enabled' if api_key else '❌ Public only'}")
    
    def _generate_signature(self, instruction: str, params: Dict = None) -> str:
        """Generate ED25519 signature for WebSocket authentication"""
        try:
            if not self.secret_key:
                raise ValueError("Secret key required for signature generation")
            
            timestamp = int(time.time() * 1000)  # Milliseconds
            window = 5000  # 5 seconds
            
            # Prepare signing parameters
            sign_params = {
                'instruction': instruction,
                'timestamp': timestamp,
                'window': window
            }
            
            # Add any additional parameters
            if params:
                sign_params.update(params)
            
            # Create signing string (alphabetically sorted)
            sorted_params = dict(sorted(sign_params.items()))
            query_string = urlencode(sorted_params)
            
            # Decode base64 secret key
            secret_key_bytes = base64.b64decode(self.secret_key)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(secret_key_bytes)
            
            # Sign the query string
            signature_bytes = private_key.sign(query_string.encode('utf-8'))
            signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
            
            return signature_b64
            
        except Exception as e:
            logger.error(f"❌ Signature generation failed: {e}")
            raise
    
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            logger.info("🔗 Connecting to Backpack WebSocket...")
            
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.is_connected = True
            self.stats['connection_start'] = datetime.now()
            
            logger.info("✅ WebSocket connection established")
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        try:
            self.should_reconnect = False
            
            if self.websocket:
                try:
                    if hasattr(self.websocket, 'closed') and not self.websocket.closed:
                        await self.websocket.close()
                    elif hasattr(self.websocket, 'close'):
                        await self.websocket.close()
                    logger.info("🔌 WebSocket disconnected")
                except Exception as close_error:
                    logger.debug(f"WebSocket close error: {close_error}")
            
            self.is_connected = False
            
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")
    
    async def subscribe_ticker(self, symbol: str = "BTC_USDC_PERP") -> bool:
        """Subscribe to real-time ticker data"""
        try:
            stream_name = f"ticker.{symbol}"
            
            # Backpack WebSocket subscription format (uppercase SUBSCRIBE)
            subscription_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name]
            }
            
            # Skip authentication for public streams (ticker is public)
            # Authentication is only needed for private streams like account updates
            
            await self._send_message(subscription_msg)
            
            self.subscriptions[stream_name] = True
            self.message_handlers[stream_name] = self._handle_ticker_update
            
            logger.info(f"📈 Subscribed to ticker stream: {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ticker subscription failed for {symbol}: {e}")
            return False
    
    async def subscribe_depth(self, symbol: str = "BTC_USDC", levels: int = 20) -> bool:
        """Subscribe to order book depth data"""
        try:
            stream_name = f"depth{levels}.{symbol}"
            
            subscription_msg = {
                "method": "SUBSCRIBE", 
                "params": [stream_name]
            }
            
            if self.api_key and self.secret_key:
                signature = self._generate_signature("subscribe", {"streams": [stream_name]})
                subscription_msg["signature"] = signature
            
            await self._send_message(subscription_msg)
            
            self.subscriptions[stream_name] = True
            self.message_handlers[stream_name] = self._handle_depth_update
            
            logger.info(f"📊 Subscribed to depth stream: {symbol} ({levels} levels)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Depth subscription failed for {symbol}: {e}")
            return False
    
    async def subscribe_klines(self, symbol: str = "BTC_USDC", interval: str = "1m") -> bool:
        """Subscribe to candlestick/K-line data"""
        try:
            stream_name = f"kline_{interval}.{symbol}"
            
            subscription_msg = {
                "method": "SUBSCRIBE",
                "params": [stream_name]
            }
            
            if self.api_key and self.secret_key:
                signature = self._generate_signature("subscribe", {"streams": [stream_name]})
                subscription_msg["signature"] = signature
            
            await self._send_message(subscription_msg)
            
            self.subscriptions[stream_name] = True
            self.message_handlers[stream_name] = self._handle_kline_update
            
            logger.info(f"🕯️ Subscribed to K-line stream: {symbol} ({interval})")
            return True
            
        except Exception as e:
            logger.error(f"❌ K-line subscription failed for {symbol}: {e}")
            return False
    
    async def subscribe_account_updates(self) -> bool:
        """Subscribe to private account updates (requires authentication)"""
        try:
            if not self.api_key or not self.secret_key:
                logger.warning("⚠️ Account updates require API credentials")
                return False
            
            streams = ["order_update", "position_update"]
            
            subscription_msg = {
                "method": "SUBSCRIBE",
                "params": streams
            }
            
            signature = self._generate_signature("subscribe", {"streams": streams})
            subscription_msg["signature"] = signature
            
            await self._send_message(subscription_msg)
            
            for stream in streams:
                self.subscriptions[stream] = True
                self.message_handlers[stream] = self._handle_account_update
            
            logger.info("👤 Subscribed to account update streams")
            return True
            
        except Exception as e:
            logger.error(f"❌ Account subscription failed: {e}")
            return False
    
    async def _send_message(self, message: Dict):
        """Send message to WebSocket"""
        try:
            if not self.is_connected or not self.websocket:
                raise ConnectionError("WebSocket not connected")
            
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            
            logger.debug(f"📤 Sent: {message_str}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            raise
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=60  # 1 minute timeout
                    )
                    
                    self.stats['messages_received'] += 1
                    self.stats['last_message_time'] = datetime.now()
                    
                    # Parse message
                    data = json.loads(message)
                    await self._process_message(data)
                    
                except asyncio.TimeoutError:
                    logger.warning("⚠️ WebSocket message timeout")
                    break
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("⚠️ WebSocket connection closed")
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Message handling error: {e}")
                    continue
            
            # Connection lost, attempt reconnection
            if self.should_reconnect:
                await self._reconnect()
                
        except Exception as e:
            logger.error(f"❌ Message handler crashed: {e}")
    
    async def _process_message(self, data: Dict):
        """Process received WebSocket message"""
        try:
            # Check if it's a stream update
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                # Route to appropriate handler
                if stream_name in self.message_handlers:
                    await self.message_handlers[stream_name](stream_data)
                else:
                    logger.debug(f"📥 Unhandled stream: {stream_name}")
            
            # Handle subscription confirmations
            elif 'result' in data:
                if data.get('result') is None:
                    logger.info("✅ Subscription confirmed")
                else:
                    logger.info(f"📬 Subscription result: {data['result']}")
            
            # Handle errors
            elif 'error' in data:
                logger.error(f"❌ WebSocket error: {data['error']}")
            
            else:
                logger.debug(f"📥 Other message: {data}")
                
        except Exception as e:
            logger.error(f"❌ Message processing error: {e}")
    
    async def _handle_ticker_update(self, data: Dict):
        """Handle ticker price updates"""
        try:
            symbol = data.get('s')  # Symbol
            price = float(data.get('c', 0))  # Close price (current price)
            
            if symbol and price > 0:
                self.latest_prices[symbol] = price
                logger.debug(f"💰 {symbol}: ${price:,.2f}")
                
                # Trigger any custom callbacks
                await self._trigger_price_callbacks(symbol, price, data)
            else:
                logger.debug(f"❌ Invalid ticker data: symbol={symbol}, price={price}")
                
        except Exception as e:
            logger.error(f"❌ Ticker update error: {e}")
            logger.debug(f"❌ Raw ticker data: {data}")
    
    async def _handle_depth_update(self, data: Dict):
        """Handle order book depth updates"""
        try:
            symbol = data.get('s')
            bids = data.get('b', [])  # Bids
            asks = data.get('a', [])  # Asks
            
            if symbol:
                self.order_books[symbol] = {
                    'bids': [[float(price), float(qty)] for price, qty in bids],
                    'asks': [[float(price), float(qty)] for price, qty in asks],
                    'timestamp': data.get('T', time.time() * 1000)
                }
                
                logger.debug(f"📊 {symbol} order book updated: {len(bids)} bids, {len(asks)} asks")
                
        except Exception as e:
            logger.error(f"❌ Depth update error: {e}")
    
    async def _handle_kline_update(self, data: Dict):
        """Handle K-line/candlestick updates"""
        try:
            kline_data = data.get('k', {})
            symbol = kline_data.get('s')
            close_price = float(kline_data.get('c', 0))
            
            if symbol and close_price > 0:
                self.latest_prices[symbol] = close_price
                logger.debug(f"🕯️ {symbol} K-line: ${close_price:,.2f}")
                
        except Exception as e:
            logger.error(f"❌ K-line update error: {e}")
    
    async def _handle_account_update(self, data: Dict):
        """Handle account/position updates"""
        try:
            update_type = data.get('e')  # Event type
            self.account_updates.append({
                'type': update_type,
                'data': data,
                'timestamp': datetime.now()
            })
            
            logger.info(f"💼 Account update: {update_type}")
            
            # Keep only last 100 updates
            if len(self.account_updates) > 100:
                self.account_updates = self.account_updates[-100:]
                
        except Exception as e:
            logger.error(f"❌ Account update error: {e}")
    
    async def _trigger_price_callbacks(self, symbol: str, price: float, data: Dict):
        """Trigger any registered price callbacks"""
        # This can be extended to support custom callbacks
        pass
    
    async def _reconnect(self):
        """Attempt to reconnect WebSocket"""
        try:
            self.stats['reconnections'] += 1
            logger.info(f"🔄 Attempting reconnection #{self.stats['reconnections']}")
            
            await asyncio.sleep(self.reconnect_interval)
            
            if await self.connect():
                # Re-subscribe to all previous streams
                for stream_name in list(self.subscriptions.keys()):
                    if stream_name.startswith('ticker.'):
                        symbol = stream_name.split('.')[1]
                        await self.subscribe_ticker(symbol)
                    elif stream_name.startswith('depth'):
                        parts = stream_name.split('.')
                        levels = int(parts[0].replace('depth', ''))
                        symbol = parts[1]
                        await self.subscribe_depth(symbol, levels)
                    elif stream_name.startswith('kline_'):
                        parts = stream_name.split('.')
                        interval = parts[0].replace('kline_', '')
                        symbol = parts[1]
                        await self.subscribe_klines(symbol, interval)
                
                logger.info("✅ Reconnection successful, subscriptions restored")
            else:
                logger.error("❌ Reconnection failed")
                
        except Exception as e:
            logger.error(f"❌ Reconnection error: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_order_book(self, symbol: str) -> Optional[Dict]:
        """Get latest order book for symbol"""
        return self.order_books.get(symbol)
    
    def get_stats(self) -> Dict:
        """Get connection statistics"""
        stats = self.stats.copy()
        stats['is_connected'] = self.is_connected
        stats['active_subscriptions'] = len(self.subscriptions)
        stats['cached_prices'] = len(self.latest_prices)
        return stats
    
    async def run_forever(self):
        """Run WebSocket client with automatic reconnection"""
        logger.info("🚀 Starting WebSocket client...")
        
        while self.should_reconnect:
            try:
                if await self.connect():
                    # Wait for connection to close
                    while self.is_connected:
                        await asyncio.sleep(1)
                
                if self.should_reconnect:
                    logger.info(f"⏱️ Reconnecting in {self.reconnect_interval} seconds...")
                    await asyncio.sleep(self.reconnect_interval)
                    
            except KeyboardInterrupt:
                logger.info("⏹️ Shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Client error: {e}")
                await asyncio.sleep(self.reconnect_interval)
        
        await self.disconnect()
        logger.info("🛑 WebSocket client stopped")

# Example usage and testing
async def main():
    """Test WebSocket client"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize client
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY')
    
    client = BackpackWebSocketClient(api_key, secret_key)
    
    try:
        # Connect and subscribe
        if await client.connect():
            await client.subscribe_ticker("BTC_USDC")
            await client.subscribe_depth("BTC_USDC", 10)
            
            # Run for 30 seconds
            await asyncio.sleep(30)
            
            # Show stats
            stats = client.get_stats()
            print(f"\n📊 Statistics: {stats}")
            
            # Show latest price
            price = client.get_latest_price("BTC_USDC")
            print(f"💰 Latest BTC price: ${price:,.2f}" if price else "❌ No price data")
            
        await client.disconnect()
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
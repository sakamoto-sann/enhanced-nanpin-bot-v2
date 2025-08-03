#!/usr/bin/env python3
"""
🎒 Backpack Exchange Client for Nanpin Strategy
100% API compliant implementation for 永久ナンピン trading
"""

import asyncio
import aiohttp
import hmac
import hashlib
import base64
import json
import time
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlencode
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# Import WebSocket client
try:
    from .backpack_websocket_client import BackpackWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("⚠️ WebSocket client not available")

logger = logging.getLogger(__name__)

class BackpackAPIError(Exception):
    """Custom exception for Backpack API errors"""
    pass

class BackpackRateLimitError(BackpackAPIError):
    """Rate limit exceeded"""
    pass

class BackpackAuthError(BackpackAPIError):
    """Authentication error"""
    pass

def load_credentials_from_env() -> tuple[str, str]:
    """Load Backpack API credentials from environment variables"""
    api_key = os.getenv('BACKPACK_API_KEY')
    secret_key = os.getenv('BACKPACK_SECRET_KEY') or os.getenv('BACKPACK_API_SECRET')
    
    if not api_key or api_key == 'your_api_key_here':
        raise BackpackAuthError("BACKPACK_API_KEY not set or is placeholder")
    
    if not secret_key or secret_key == 'your_api_secret_here':
        raise BackpackAuthError("BACKPACK_SECRET_KEY not set or is placeholder")
    
    return api_key, secret_key

class BackpackNanpinClient:
    """
    🎒 Backpack Exchange Client for Nanpin Strategy
    
    Features:
    - ED25519 signature authentication
    - Automatic rate limiting
    - Position and risk management
    - Real-time market data
    - Order execution with safety checks
    """
    
    def __init__(self, api_key: str, secret_key: str, config_path: str = None):
        """Initialize Backpack client"""
        self.api_key = api_key  # This should be the base64 public key
        self.secret_key = secret_key  # This should be the base64 private key
        self.base_url = "https://api.backpack.exchange"
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Session and rate limiting
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = {
            'requests_per_minute': self.config.get('rate_limit', {}).get('requests_per_minute', 60),
            'last_requests': []
        }
        
        # Cache
        self.price_cache = {}
        self.balance_cache = {}
        self.cache_ttl = 30  # 30 seconds
        
        # WebSocket client for real-time data
        self.ws_client: Optional[BackpackWebSocketClient] = None
        self.use_websocket = WEBSOCKET_AVAILABLE and self.config.get('enable_websocket', True)
        self._ws_price_callbacks = []
        
        logger.info("🎒 Backpack client initialized")
        logger.info(f"   API Key: {self.api_key[:8]}...")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   WebSocket: {'✅ Enabled' if self.use_websocket else '❌ Disabled'}")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load client configuration"""
        default_config = {
            'timeout': 30,
            'max_retries': 3,
            'retry_delay': 1.0,
            'rate_limit': {
                'requests_per_minute': 60,
                'burst_limit': 10
            },
            'order_defaults': {
                'time_in_force': 'IOC',
                'type': 'Market'
            },
            'risk_limits': {
                'max_order_size_usdc': 10000,
                'max_daily_volume_usdc': 50000,
                'min_collateral_ratio': 3.0
            },
            'websocket': {
                'enable_websocket': True,
                'auto_subscribe_ticker': True,
                'auto_subscribe_depth': False,
                'depth_levels': 10
            }
        }
        
        if config_path:
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
                logger.info(f"✅ Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load config from {config_path}: {e}")
        
        return default_config
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("🔗 HTTP session initialized")
        
        # Initialize WebSocket client if enabled
        if self.use_websocket and not self.ws_client:
            await self._init_websocket()
    
    async def _init_websocket(self):
        """Initialize WebSocket client for real-time data"""
        try:
            if not WEBSOCKET_AVAILABLE:
                logger.warning("⚠️ WebSocket not available, falling back to REST API")
                return
            
            self.ws_client = BackpackWebSocketClient(self.api_key, self.secret_key)
            
            if await self.ws_client.connect():
                logger.info("🚀 WebSocket connection established")
                
                # Auto-subscribe to ticker if enabled
                if self.config.get('websocket', {}).get('auto_subscribe_ticker', True):
                    await self.ws_client.subscribe_ticker("BTC_USDC_PERP")
                    logger.info("📈 Auto-subscribed to BTC_USDC_PERP futures ticker")
                
                # Auto-subscribe to depth if enabled
                if self.config.get('websocket', {}).get('auto_subscribe_depth', False):
                    depth_levels = self.config.get('websocket', {}).get('depth_levels', 10)
                    await self.ws_client.subscribe_depth("BTC_USDC_PERP", depth_levels)
                    logger.info(f"📊 Auto-subscribed to order book depth ({depth_levels} levels)")
                
                # Start WebSocket message processing in background
                asyncio.create_task(self._ws_price_monitor())
                
            else:
                logger.error("❌ WebSocket connection failed")
                self.ws_client = None
                
        except Exception as e:
            logger.error(f"❌ WebSocket initialization failed: {e}")
            self.ws_client = None
    
    async def close(self):
        """Close the client session"""
        if self.ws_client:
            await self.ws_client.disconnect()
            self.ws_client = None
            logger.debug("🚀 WebSocket connection closed")
            
        if self.session:
            await self.session.close()
            self.session = None
            logger.debug("🔗 HTTP session closed")
    
    def _generate_signature(self, instruction: str, params: Dict = None) -> tuple[str, str, str]:
        """Generate ED25519 signature for Backpack API - 100% Compliant with Official Docs"""
        try:
            # Create timestamp and window per official docs
            timestamp = int(time.time() * 1000)
            window = 5000
            
            # Build signature string exactly as per official documentation:
            # "instruction=value&param1=value1&param2=value2&timestamp=value&window=value"
            
            # Start with instruction
            sign_str_parts = [f"instruction={instruction}"]
            
            # Add sorted parameters (alphabetically)
            if params:
                sorted_keys = sorted(params.keys())
                for key in sorted_keys:
                    value = params[key]
                    # Handle boolean values as lowercase strings
                    if isinstance(value, bool):
                        value = str(value).lower()
                    sign_str_parts.append(f"{key}={value}")
            
            # Add timestamp and window at the end
            sign_str_parts.append(f"timestamp={timestamp}")
            sign_str_parts.append(f"window={window}")
            
            # Join with '&' to create the signing string
            sign_str = "&".join(sign_str_parts)
            
            # Sign with Ed25519 using base64 decoded private key
            try:
                # Try base64 decoding first (standard format)
                private_key_bytes = base64.b64decode(self.secret_key)
                if len(private_key_bytes) != 32:
                    raise ValueError(f"Decoded key is {len(private_key_bytes)} bytes, need 32")
            except Exception:
                # If base64 fails, try other formats
                try:
                    if len(self.secret_key) == 64:  # Hex format (64 hex chars = 32 bytes)
                        private_key_bytes = bytes.fromhex(self.secret_key)
                    elif len(self.secret_key) == 44 and self.secret_key.endswith('='):  # Base64 with padding
                        private_key_bytes = base64.b64decode(self.secret_key)
                    else:
                        # Create 32-byte key from secret (hash if needed)
                        import hashlib
                        private_key_bytes = hashlib.sha256(self.secret_key.encode('utf-8')).digest()
                        logger.warning(f"⚠️ Using hashed API secret as ED25519 key")
                    
                    if len(private_key_bytes) != 32:
                        raise ValueError(f"Key is {len(private_key_bytes)} bytes, need exactly 32")
                        
                except Exception as format_error:
                    raise Exception(f"Cannot create 32-byte ED25519 key from API secret: {format_error}")
            
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            signature = private_key.sign(sign_str.encode('utf-8'))
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            logger.debug(f"🔐 Generated signature for: {sign_str}")
            
            return signature_b64, str(timestamp), str(window)
            
        except Exception as e:
            logger.error(f"❌ Signature generation failed: {e}")
            raise BackpackAuthError(f"Failed to generate signature: {e}")
    
    async def _ws_price_monitor(self):
        """Monitor WebSocket price updates and update cache"""
        try:
            while self.ws_client and self.ws_client.is_connected:
                # Get latest price from WebSocket
                ws_price = self.ws_client.get_latest_price("BTC_USDC")
                
                if ws_price:
                    # Update cache with real-time price
                    self.price_cache['btc_price'] = {
                        'price': ws_price,
                        'timestamp': time.time(),
                        'source': 'websocket'
                    }
                    
                    # Trigger any registered callbacks
                    for callback in self._ws_price_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(ws_price)
                            else:
                                callback(ws_price)
                        except Exception as e:
                            logger.error(f"❌ Price callback error: {e}")
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
        except Exception as e:
            logger.error(f"❌ WebSocket price monitor error: {e}")
    
    def add_price_callback(self, callback):
        """Add callback for real-time price updates"""
        self._ws_price_callbacks.append(callback)
        logger.info(f"📈 Added price callback: {callback.__name__}")
    
    def remove_price_callback(self, callback):
        """Remove price callback"""
        if callback in self._ws_price_callbacks:
            self._ws_price_callbacks.remove(callback)
            logger.info(f"📈 Removed price callback: {callback.__name__}")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self.rate_limiter['last_requests'] = [
            req_time for req_time in self.rate_limiter['last_requests'] 
            if req_time > minute_ago
        ]
        
        # Check if we can make request
        if len(self.rate_limiter['last_requests']) >= self.rate_limiter['requests_per_minute']:
            sleep_time = 60 - (now - self.rate_limiter['last_requests'][0])
            if sleep_time > 0:
                logger.warning(f"⏱️ Rate limit reached, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.rate_limiter['last_requests'].append(now)
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                          signed: bool = False, instruction: str = None) -> Dict:
        """Make HTTP request to Backpack API"""
        await self._init_session()
        await self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'NanpinBot/1.0'
        }
        
        # Add authentication if signed
        if signed:
            if not instruction:
                instruction = endpoint.split('/')[-1]
            
            signature, timestamp, window = self._generate_signature(instruction, params)
            headers.update({
                'X-API-Key': self.api_key,
                'X-Signature': signature,
                'X-Timestamp': timestamp,
                'X-Window': window
            })
        
        try:
            # Make request with retries
            for attempt in range(self.config['max_retries']):
                try:
                    if method.upper() == 'GET':
                        async with self.session.get(url, headers=headers, params=params) as response:
                            return await self._handle_response(response)
                    
                    elif method.upper() == 'POST':
                        data = json.dumps(params) if params else None
                        async with self.session.post(url, headers=headers, data=data) as response:
                            return await self._handle_response(response)
                    
                    else:
                        raise BackpackAPIError(f"Unsupported HTTP method: {method}")
                        
                except aiohttp.ClientError as e:
                    if attempt == self.config['max_retries'] - 1:
                        raise BackpackAPIError(f"Request failed after {self.config['max_retries']} attempts: {e}")
                    
                    await asyncio.sleep(self.config['retry_delay'] * (attempt + 1))
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Request failed: {method} {endpoint} - {e}")
            raise
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Enhanced HTTP response handling with comprehensive error codes"""
        try:
            if response.content_type == 'application/json':
                data = await response.json()
            else:
                text = await response.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = {'message': text}
            
            # Success responses
            if response.status == 200:
                return data
            elif response.status == 201:
                return data
            elif response.status == 204:
                return {}  # No content
            
            # Client errors (4xx)
            elif response.status == 400:
                error_msg = data.get('message', 'Bad request - invalid parameters')
                raise BackpackAPIError(f"Bad Request (400): {error_msg}")
            elif response.status == 401:
                error_msg = data.get('message', 'Unauthorized - invalid API credentials')
                raise BackpackAuthError(f"Unauthorized (401): {error_msg}")
            elif response.status == 403:
                error_msg = data.get('message', 'Forbidden - insufficient permissions')
                raise BackpackAuthError(f"Forbidden (403): {error_msg}")
            elif response.status == 404:
                error_msg = data.get('message', 'Not found - endpoint or resource does not exist')
                raise BackpackAPIError(f"Not Found (404): {error_msg}")
            elif response.status == 422:
                error_msg = data.get('message', 'Unprocessable entity - invalid request format')
                raise BackpackAPIError(f"Unprocessable Entity (422): {error_msg}")
            elif response.status == 429:
                error_msg = data.get('message', 'Rate limit exceeded')
                retry_after = response.headers.get('Retry-After', '60')
                raise BackpackRateLimitError(f"Rate Limited (429): {error_msg}. Retry after {retry_after}s")
            
            # Server errors (5xx)
            elif response.status == 500:
                error_msg = data.get('message', 'Internal server error')
                raise BackpackAPIError(f"Server Error (500): {error_msg}")
            elif response.status == 502:
                error_msg = data.get('message', 'Bad gateway')
                raise BackpackAPIError(f"Bad Gateway (502): {error_msg}")
            elif response.status == 503:
                error_msg = data.get('message', 'Service unavailable')
                raise BackpackAPIError(f"Service Unavailable (503): {error_msg}")
            elif response.status == 504:
                error_msg = data.get('message', 'Gateway timeout')
                raise BackpackAPIError(f"Gateway Timeout (504): {error_msg}")
            
            # Other errors
            else:
                error_msg = data.get('message', f'Unknown error with status {response.status}')
                raise BackpackAPIError(f"HTTP {response.status}: {error_msg}")
                
        except Exception as e:
            if isinstance(e, (BackpackAPIError, BackpackRateLimitError, BackpackAuthError)):
                raise
            else:
                raise BackpackAPIError(f"Response handling failed: {e}")
    
    # Public Market Data Methods
    
    async def get_btc_price(self) -> float:
        """Get current BTC price with WebSocket priority"""
        try:
            now = time.time()
            
            # First: Try WebSocket real-time price (highest priority)
            if self.ws_client and self.ws_client.is_connected:
                ws_price = self.ws_client.get_latest_price("BTC_USDC")
                if ws_price:
                    logger.debug(f"💰 Real-time price from WebSocket: ${ws_price:,.2f}")
                    return ws_price
            
            # Second: Check cache (if WebSocket unavailable)
            if 'btc_price' in self.price_cache:
                price_data = self.price_cache['btc_price']
                if now - price_data['timestamp'] < self.cache_ttl:
                    source = price_data.get('source', 'rest')
                    logger.debug(f"💰 Cached price ({source}): ${price_data['price']:,.2f}")
                    return price_data['price']
            
            # Third: Fallback to REST API
            logger.debug("📡 Fetching price from REST API...")
            response = await self._make_request('GET', '/api/v1/ticker', {'symbol': 'BTC_USDC'})
            
            price = float(response['lastPrice'])
            
            # Update cache
            self.price_cache['btc_price'] = {
                'price': price,
                'timestamp': now,
                'source': 'rest'
            }
            
            logger.debug(f"💰 REST API price: ${price:,.2f}")
            return price
            
        except Exception as e:
            logger.error(f"❌ Failed to get BTC price: {e}")
            return None
    
    async def get_klines(self, symbol: str = 'BTC_USDC', interval: str = '1h', 
                        limit: int = 100) -> List[Dict]:
        """Get candlestick data"""
        try:
            # Calculate start time like the working v1.3 client
            interval_minutes = {
                "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720,
                "1d": 1440, "3d": 4320, "1w": 10080
            }.get(interval, 60)
            
            start_time = int(time.time() - (interval_minutes * limit * 60))
            
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time
            }
            
            response = await self._make_request('GET', '/api/v1/klines', params, signed=False)
            return response if response else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get klines: {e}")
            return []
    
    # ===========================
    # FUTURES API METHODS (CRITICAL)
    # ===========================
    
    async def get_mark_price(self, symbol: str = 'BTC_USDC_PERP') -> Optional[float]:
        """Get mark price for futures (critical for liquidation calculations)"""
        try:
            # First try to get it from ticker endpoint
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/api/v1/ticker', params, signed=False)
            
            if response and 'markPrice' in response:
                price = float(response['markPrice'])
                logger.debug(f"💰 Mark price from ticker for {symbol}: ${price:,.2f}")
                return price
            
            # If no markPrice in ticker, try the lastPrice  
            if response and 'lastPrice' in response:
                price = float(response['lastPrice'])
                logger.debug(f"💰 Last price from ticker for {symbol}: ${price:,.2f}")
                return price
                
            # Try current price as final fallback
            try:
                current_price = await self.get_current_price(symbol)
                if current_price:
                    logger.warning(f"⚠️ Using current price as mark price fallback: ${current_price:,.2f}")
                    return current_price
            except Exception as fallback_error:
                logger.debug(f"Fallback price fetch failed: {fallback_error}")
            
            logger.warning(f"⚠️ No mark price data found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get mark price for {symbol}: {e}")
            return None
    
    async def get_funding_rate(self, symbol: str = 'BTC_USDC') -> Optional[Dict]:
        """Get current funding rate for perpetual futures"""
        try:
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/api/v1/fundingRate', params, signed=False)
            
            if response:
                return {
                    'symbol': symbol,
                    'funding_rate': float(response.get('fundingRate', 0)),
                    'next_funding_time': response.get('nextFundingTime'),
                    'countdown': response.get('countdown', 0)
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get funding rate: {e}")
            return None
    
    async def get_open_interest(self, symbol: str = 'BTC_USDC') -> Optional[Dict]:
        """Get open interest data for futures"""
        try:
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/api/v1/openInterest', params, signed=False)
            
            if response:
                return {
                    'symbol': symbol,
                    'open_interest': float(response.get('openInterest', 0)),
                    'timestamp': response.get('timestamp')
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get open interest: {e}")
            return None
    
    async def get_liquidation_orders(self, symbol: str = 'BTC_USDC') -> List[Dict]:
        """Get recent liquidation orders"""
        try:
            params = {'symbol': symbol}
            response = await self._make_request('GET', '/api/v1/liquidationOrders', params, signed=False)
            return response if response else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get liquidation orders: {e}")
            return []
    
    # Account Information Methods
    
    async def get_balances(self) -> Dict:
        """Get account balances"""
        try:
            # Check cache
            now = time.time()
            if 'balances' in self.balance_cache:
                balance_data = self.balance_cache['balances']
                if now - balance_data['timestamp'] < self.cache_ttl:
                    return balance_data['balances']
            
            response = await self._make_request('GET', '/api/v1/capital', signed=True, instruction='balanceQuery')
            
            # Process balances - response is a dict format from Backpack
            balances = {}
            if isinstance(response, dict):
                for asset, balance_data in response.items():
                    balances[asset] = {
                        'available': float(balance_data['available']),
                        'locked': float(balance_data['locked']),
                        'staked': float(balance_data.get('staked', 0)),
                        'total': float(balance_data['available']) + float(balance_data['locked']) + float(balance_data.get('staked', 0))
                    }
            else:
                # Fallback for list format
                for balance in response:
                    asset = balance['asset']
                    balances[asset] = {
                        'available': float(balance['available']),
                        'locked': float(balance['locked']),
                        'total': float(balance['available']) + float(balance['locked'])
                    }
            
            # Update cache
            self.balance_cache['balances'] = {
                'balances': balances,
                'timestamp': now
            }
            
            return balances
            
        except Exception as e:
            logger.error(f"❌ Failed to get balances: {e}")
            return {}
    
    async def get_btc_position(self) -> Optional[Dict]:
        """Get current BTC position"""
        try:
            response = await self._make_request('GET', '/api/v1/position', signed=True, instruction='positionQuery')
            
            for position in response:
                if position['symbol'] == 'BTC_USDC':
                    return {
                        'symbol': position['symbol'],
                        'size': float(position['size']),
                        'entryPrice': float(position['entryPrice']) if position['entryPrice'] else None,
                        'markPrice': float(position['markPrice']) if position['markPrice'] else None,
                        'pnl': float(position['unrealizedPnl']) if position['unrealizedPnl'] else None,
                        'percentage': float(position['percentage']) if position['percentage'] else None
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get BTC position: {e}")
            return None
    
    async def get_collateral_info(self) -> Optional[Dict]:
        """Get collateral and margin information using official endpoint"""
        try:
            # Use the official collateral endpoint
            response = await self._make_request(
                'GET', 
                '/api/v1/capital/collateral', 
                signed=True, 
                instruction='collateralQuery'
            )
            
            if not response:
                logger.warning("⚠️ Empty response from collateral endpoint")
                return await self._calculate_collateral_from_balances()
            
            logger.info("💰 Retrieved collateral information from official endpoint")
            logger.debug(f"Collateral response: {response}")
            
            # Return the response as-is since it comes from the official endpoint
            # The response should contain:
            # - netEquity
            # - availableBalance 
            # - marginFraction
            # - borrowLiability
            # - unrealizedPnl
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get collateral info: {e}")
            # Fallback to balance-based calculation
            return await self._calculate_collateral_from_balances()
    
    async def _calculate_collateral_from_balances(self) -> Optional[Dict]:
        """Fallback: Calculate collateral info from balances"""
        try:
            balances = await self.get_balances()
            if not balances:
                return None
            
            # Calculate totals
            total_equity = 0
            total_available = 0
            
            for asset, balance_data in balances.items():
                if asset in ['USDC', 'BTC']:
                    available = balance_data['available']
                    locked = balance_data['locked']
                    
                    if asset == 'USDC':
                        total_equity += available + locked
                        total_available += available
                    elif asset == 'BTC':
                        # Convert BTC to USDC value
                        btc_price = await self.get_btc_price()
                        if btc_price:
                            btc_value = (available + locked) * btc_price
                            total_equity += btc_value
                            total_available += available * btc_price
            
            # Calculate margin ratio (simplified)
            margin_used = total_equity - total_available
            margin_ratio = total_available / margin_used if margin_used > 0 else float('inf')
            
            return {
                'netEquity': total_equity,
                'availableBalance': total_available,
                'marginUsed': margin_used,
                'marginFraction': margin_used / total_equity if total_equity > 0 else 0,
                'marginRatio': margin_ratio
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate collateral from balances: {e}")
            return None
    
    # Risk Management Methods
    
    async def check_liquidation_risk(self) -> Dict:
        """Check liquidation risk level"""
        try:
            collateral_info = await self.get_collateral_info()
            if not collateral_info:
                return {'liquidation_risk': 'unknown', 'reason': 'Could not get collateral info'}
            
            margin_ratio = collateral_info.get('marginRatio')
            margin_fraction = collateral_info.get('marginFraction')
            
            # Handle None values safely
            if margin_ratio is None:
                margin_ratio = float('inf')  # No margin used
            if margin_fraction is None:
                margin_fraction = 0.0  # No margin used
            
            # Determine risk level
            if margin_ratio < 2.0 or margin_fraction > 0.5:
                risk_level = 'critical'
            elif margin_ratio < 3.0 or margin_fraction > 0.33:
                risk_level = 'high'
            elif margin_ratio < 4.0 or margin_fraction > 0.25:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            return {
                'liquidation_risk': risk_level,
                'margin_ratio': margin_ratio,
                'margin_fraction': margin_fraction,
                'net_equity': collateral_info.get('netEquity', 0),
                'available_balance': collateral_info.get('availableBalance', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to check liquidation risk: {e}")
            return {'liquidation_risk': 'unknown', 'error': str(e)}
    
    async def calculate_safe_order_size(self, desired_size_usdc: float) -> float:
        """Calculate safe order size considering risk limits and collateral"""
        try:
            # Check risk limits
            max_order = self.config['risk_limits']['max_order_size_usdc']
            if desired_size_usdc > max_order:
                logger.warning(f"⚠️ Order size {desired_size_usdc} exceeds limit {max_order}")
                desired_size_usdc = max_order
            
            # Check collateral - use netEquityAvailable for trading power
            collateral_info = await self.get_collateral_info()
            if collateral_info:
                # Use netEquityAvailable instead of availableBalance for collateral-based trading
                net_equity_available = float(collateral_info.get('netEquityAvailable', 0))
                available_balance = float(collateral_info.get('availableBalance', 0))
                
                # Use the higher of the two (collateral allows more trading power)
                trading_power = max(net_equity_available, available_balance)
                
                logger.info(f"💰 Trading power: ${trading_power:.2f} (Net Equity Available: ${net_equity_available:.2f}, Available Balance: ${available_balance:.2f})")
                
                # Ensure we don't use more than 80% of trading power
                max_safe_size = trading_power * 0.8
                if desired_size_usdc > max_safe_size:
                    logger.warning(f"⚠️ Order size {desired_size_usdc} exceeds safe limit {max_safe_size}")
                    desired_size_usdc = max_safe_size
                
                # If we have trading power, allow smaller minimum orders
                if trading_power > 0:
                    min_order = 5.0  # Lower minimum when using collateral
                else:
                    min_order = 10.0  # Original minimum
            else:
                min_order = 10.0
            
            # Check minimum order size
            if desired_size_usdc < min_order:
                logger.warning(f"⚠️ Order size {desired_size_usdc} below minimum {min_order}")
                return 0
            
            logger.info(f"✅ Safe order size calculated: ${desired_size_usdc:.2f}")
            return desired_size_usdc
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate safe order size: {e}")
            return 0
    
    # Trading Methods
    
    async def market_buy_btc_futures(self, usdc_amount: float, reason: str = None) -> Optional[Dict]:
        """Execute market buy order for BTC futures (perpetual)"""
        try:
            # Safety checks
            safe_amount = await self.calculate_safe_order_size(usdc_amount)
            if safe_amount <= 0:
                logger.error("❌ Order rejected by safety checks")
                return None
            
            # Check collateral availability
            collateral_info = await self.get_collateral_info()
            if not collateral_info:
                logger.error("❌ Could not get collateral information")
                return None
                
            net_equity_available = float(collateral_info.get('netEquityAvailable', 0))
            if net_equity_available < safe_amount:
                logger.error(f"❌ Insufficient collateral: ${net_equity_available:.2f} < ${safe_amount:.2f}")
                return None
            
            # Get current futures price for quantity calculation
            btc_price = await self.get_mark_price('BTC_USDC_PERP')
            if not btc_price:
                logger.error("❌ Could not get BTC futures price for order")
                return None
            
            quantity = safe_amount / btc_price
            
            # Round quantity to appropriate precision for Backpack (try 4 decimal places)
            # Some exchanges have strict decimal precision requirements
            quantity = round(quantity, 4)
            
            # Prepare futures order parameters  
            order_params = {
                'symbol': 'BTC_USDC_PERP',  # BTC Perpetual Futures symbol
                'side': 'Bid',              # Buy futures position
                'orderType': 'Market',
                'quantity': f"{quantity:.4f}",  # BTC quantity for futures (4 decimals)
                'timeInForce': self.config['order_defaults']['time_in_force']
            }
            
            # Add clientId as integer if reason provided
            if reason:
                order_params['clientId'] = int(time.time())  # Integer timestamp
            
            logger.info(f"🚀 Placing BTC FUTURES market buy order:")
            logger.info(f"   Collateral Used: ${safe_amount:.2f} USDC")
            logger.info(f"   BTC Quantity: {quantity:.4f} BTC")
            logger.info(f"   Mark Price: ${btc_price:.2f}")
            logger.info(f"   Symbol: BTC_USDC_PERP (Perpetual Futures)")
            
            # Execute futures order
            response = await self._make_request(
                'POST', 
                '/api/v1/order', 
                order_params, 
                signed=True, 
                instruction='orderExecute'
            )
            
            logger.info(f"✅ Order executed successfully:")
            logger.info(f"   Order ID: {response.get('id', 'N/A')}")
            logger.info(f"   Status: {response.get('status', 'N/A')}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to execute market buy: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status"""
        try:
            params = {'orderId': order_id}
            response = await self._make_request(
                'GET', 
                '/api/v1/order', 
                params, 
                signed=True, 
                instruction='orderQuery'
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get order status: {e}")
            return None
    
    # ===========================
    # TRADE EXECUTION TRACKING (CRITICAL)
    # ===========================
    
    async def get_fills(self, symbol: str = 'BTC_USDC', limit: int = 100) -> List[Dict]:
        """Get recent fill history for execution tracking"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            response = await self._make_request(
                'GET', 
                '/api/v1/fills', 
                params, 
                signed=True, 
                instruction='fillHistoryQueryAll'
            )
            return response if response else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get fills: {e}")
            return []
    
    async def get_trade_history(self, symbol: str = 'BTC_USDC', limit: int = 100) -> List[Dict]:
        """Get detailed trade history"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            response = await self._make_request(
                'GET', 
                '/api/v1/wsTrades', 
                params, 
                signed=True, 
                instruction='tradeHistoryQueryAll'
            )
            return response if response else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get trade history: {e}")
            return []
    
    async def get_order_history(self, symbol: str = 'BTC_USDC', limit: int = 100) -> List[Dict]:
        """Get complete order history"""
        try:
            params = {
                'symbol': symbol,
                'limit': limit
            }
            response = await self._make_request(
                'GET', 
                '/api/v1/allOrders', 
                params, 
                signed=True, 
                instruction='orderHistoryQueryAll'
            )
            return response if response else []
            
        except Exception as e:
            logger.error(f"❌ Failed to get order history: {e}")
            return []
    
    # Utility Methods
    
    async def test_connection(self) -> bool:
        """Test API connection"""
        try:
            await self.get_btc_price()
            logger.info("✅ Backpack connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Backpack connection test failed: {e}")
            return False
    
    async def test_authentication(self) -> bool:
        """Test API authentication"""
        try:
            await self.get_balances()
            logger.info("✅ Backpack authentication test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Backpack authentication test failed: {e}")
            return False
    
    def get_client_info(self) -> Dict:
        """Get client information"""
        return {
            'api_key': f"{self.api_key[:8]}...",
            'base_url': self.base_url,
            'rate_limit': self.rate_limiter['requests_per_minute'],
            'cache_ttl': self.cache_ttl,
            'session_active': self.session is not None
        }
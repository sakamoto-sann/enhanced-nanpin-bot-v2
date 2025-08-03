#!/usr/bin/env python3
"""
🌸 Backpack Exchange Client for Nanpin Strategy
100% Official Documentation Compliant Implementation
永久ナンピン (Permanent DCA) Specialized Client
"""

import asyncio
import aiohttp
import base64
import json
import logging
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import hashlib
import hmac

logger = logging.getLogger(__name__)

class BackpackNanpinClient:
    """
    🌸 Specialized Backpack Exchange client for 永久ナンピン strategy
    
    Features:
    - 100% official Backpack API compliance
    - ED25519 authentication per official docs
    - Optimized for permanent DCA accumulation strategy
    - Single position scaling for BTC_USDC
    - Advanced risk management and liquidation protection
    """
    
    def __init__(self, api_key: str, secret_key: str, config_path: str = None):
        """
        Initialize Backpack Nanpin Client
        
        Args:
            api_key: Base64 encoded public key
            secret_key: Base64 encoded private key
            config_path: Path to configuration file
        """
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
            
        # API settings
        self.base_url = self.config['api']['base_url']
        self.timeout = self.config['api']['timeout']
        self.max_retries = self.config['api']['max_retries']
        
        # Initialize session
        self.session = None
        
        # Initialize ED25519 private key
        self._init_private_key()
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_reset = time.time() + 60  # Reset every minute
        
        # Position tracking
        self.current_position = None
        self.average_entry_price = 0.0
        self.total_invested = 0.0
        
        logger.info("🌸 Backpack Nanpin Client initialized")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Target Symbol: {self.config['orders']['default_symbol']}")
        logger.info(f"   Strategy: 永久ナンピン (Permanent DCA)")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'api': {
                'base_url': 'https://api.backpack.exchange',
                'timeout': 30.0,
                'max_retries': 3,
            },
            'orders': {
                'default_symbol': 'BTC_USDC',
                'default_side': 'Bid',
                'default_type': 'Market',
                'default_tif': 'IOC',
            },
            'authentication': {
                'default_window': 5000,
                'max_window': 60000,
            }
        }
    
    def _init_private_key(self):
        """Initialize ED25519 private key for signing"""
        try:
            # Try base64 decoding first
            private_key_bytes = base64.b64decode(self.secret_key)
            if len(private_key_bytes) == 32:
                self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                logger.info("✅ ED25519 private key initialized from base64")
                return
        except Exception:
            pass
        
        try:
            # Try hex decoding
            if len(self.secret_key) == 64:
                private_key_bytes = bytes.fromhex(self.secret_key)
                self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
                logger.info("✅ ED25519 private key initialized from hex")
                return
        except Exception:
            pass
        
        # Fallback: hash the secret
        try:
            private_key_bytes = hashlib.sha256(self.secret_key.encode('utf-8')).digest()
            self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
            logger.info("✅ ED25519 private key initialized from hash")
        except Exception as e:
            raise Exception(f"❌ Failed to initialize ED25519 private key: {e}")
    
    async def _init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    keepalive_timeout=30
                )
            )
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _create_signature(self, instruction: str, params: Dict = None, 
                         timestamp: int = None, window: int = None) -> Dict[str, str]:
        """
        Create ED25519 signature for Backpack API request
        
        Args:
            instruction: API instruction type
            params: Request parameters
            timestamp: Unix timestamp in milliseconds
            window: Request validity window
            
        Returns:
            Dictionary with authentication headers
        """
        if timestamp is None:
            timestamp = int(time.time() * 1000)
        
        if window is None:
            window = self.config['authentication']['default_window']
        
        # Build signature string according to Backpack docs
        sign_str = f"instruction={instruction}"
        
        # Add sorted parameters
        if params:
            sorted_params = []
            for key in sorted(params.keys()):
                value = params[key]
                # Convert boolean to lowercase string
                if isinstance(value, bool):
                    value = str(value).lower()
                sorted_params.append(f"{key}={value}")
            
            if sorted_params:
                sign_str += "&" + "&".join(sorted_params)
        
        # Add timestamp and window
        sign_str += f"&timestamp={timestamp}&window={window}"
        
        # Create signature
        signature_bytes = self.private_key.sign(sign_str.encode('utf-8'))
        signature_b64 = base64.b64encode(signature_bytes).decode('utf-8')
        
        # Return headers
        headers = {
            'X-API-Key': self.api_key,
            'X-Signature': signature_b64,
            'X-Timestamp': str(timestamp),
            'X-Window': str(window),
            'Content-Type': 'application/json'
        }
        
        logger.debug(f"🔐 Created signature for instruction: {instruction}")\n        \n        return headers\n    \n    async def _rate_limit_check(self):\n        \"\"\"Check and enforce rate limiting\"\"\"\n        now = time.time()\n        \n        # Reset counter every minute\n        if now > self.rate_limit_reset:\n            self.request_count = 0\n            self.rate_limit_reset = now + 60\n        \n        # Check rate limit\n        max_requests = self.config.get('rate_limits', {}).get('requests_per_minute', 100)\n        if self.request_count >= max_requests:\n            wait_time = self.rate_limit_reset - now\n            logger.warning(f\"⏱️ Rate limit reached, waiting {wait_time:.1f}s\")\n            await asyncio.sleep(wait_time)\n            self.request_count = 0\n            self.rate_limit_reset = time.time() + 60\n        \n        # Minimum time between requests\n        min_interval = 60 / max_requests\n        time_since_last = now - self.last_request_time\n        if time_since_last < min_interval:\n            await asyncio.sleep(min_interval - time_since_last)\n        \n        self.last_request_time = time.time()\n        self.request_count += 1\n    \n    async def _make_request(self, method: str, endpoint: str, instruction: str,\n                          params: Dict = None, data: Dict = None) -> Dict:\n        \"\"\"\n        Make authenticated request to Backpack API\n        \n        Args:\n            method: HTTP method (GET, POST, DELETE)\n            endpoint: API endpoint\n            instruction: Backpack instruction type\n            params: Query parameters\n            data: Request body data\n            \n        Returns:\n            API response as dictionary\n        \"\"\"\n        await self._init_session()\n        await self._rate_limit_check()\n        \n        # Prepare request parameters\n        request_params = params or {}\n        request_data = data or {}\n        \n        # Create signature\n        if data:\n            headers = self._create_signature(instruction, data)\n        else:\n            headers = self._create_signature(instruction, params)\n        \n        url = f\"{self.base_url}{endpoint}\"\n        \n        for attempt in range(self.max_retries + 1):\n            try:\n                if method.upper() == 'GET':\n                    async with self.session.get(url, headers=headers, params=request_params) as response:\n                        return await self._handle_response(response, endpoint)\n                \n                elif method.upper() == 'POST':\n                    async with self.session.post(url, headers=headers, json=request_data) as response:\n                        return await self._handle_response(response, endpoint)\n                \n                elif method.upper() == 'DELETE':\n                    if data:\n                        async with self.session.delete(url, headers=headers, json=request_data) as response:\n                            return await self._handle_response(response, endpoint)\n                    else:\n                        async with self.session.delete(url, headers=headers, params=request_params) as response:\n                            return await self._handle_response(response, endpoint)\n                \n                else:\n                    raise ValueError(f\"Unsupported HTTP method: {method}\")\n                    \n            except aiohttp.ClientError as e:\n                if attempt < self.max_retries:\n                    wait_time = (2 ** attempt) * self.config['api'].get('retry_delay', 1.0)\n                    logger.warning(f\"⚠️ Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}\")\n                    await asyncio.sleep(wait_time)\n                    continue\n                else:\n                    logger.error(f\"❌ Request failed after {self.max_retries + 1} attempts: {e}\")\n                    raise\n        \n        raise Exception(\"Should not reach here\")\n    \n    async def _handle_response(self, response: aiohttp.ClientResponse, endpoint: str) -> Dict:\n        \"\"\"Handle API response\"\"\"\n        try:\n            response_data = await response.json()\n        except Exception:\n            response_text = await response.text()\n            logger.error(f\"❌ Failed to parse JSON response from {endpoint}: {response_text}\")\n            raise Exception(f\"Invalid JSON response: {response_text}\")\n        \n        if response.status == 200:\n            logger.debug(f\"✅ Successful request to {endpoint}\")\n            return response_data\n        \n        elif response.status == 429:\n            logger.warning(f\"⏱️ Rate limited on {endpoint}\")\n            await asyncio.sleep(5)  # Wait 5 seconds on rate limit\n            raise aiohttp.ClientError(f\"Rate limited: {response_data}\")\n        \n        elif response.status in [401, 403]:\n            logger.error(f\"🔐 Authentication failed on {endpoint}: {response_data}\")\n            raise Exception(f\"Authentication error: {response_data}\")\n        \n        elif response.status >= 400:\n            error_msg = response_data.get('error', response_data)\n            logger.error(f\"❌ API error on {endpoint}: {error_msg}\")\n            raise Exception(f\"API error ({response.status}): {error_msg}\")\n        \n        return response_data\n    \n    # ===========================\n    # ACCOUNT MANAGEMENT METHODS\n    # ===========================\n    \n    async def get_balances(self) -> Dict:\n        \"\"\"Get account balances\"\"\"\n        try:\n            response = await self._make_request('GET', '/api/v1/capital', 'balanceQuery')\n            logger.info(\"📊 Retrieved account balances\")\n            return response\n        except Exception as e:\n            logger.error(f\"❌ Failed to get balances: {e}\")\n            raise\n    \n    async def get_collateral_info(self) -> Dict:\n        \"\"\"Get collateral information\"\"\"\n        try:\n            response = await self._make_request('GET', '/api/v1/capital/collateral', 'collateralQuery')\n            logger.info(\"💰 Retrieved collateral information\")\n            return response\n        except Exception as e:\n            logger.error(f\"❌ Failed to get collateral info: {e}\")\n            raise\n    \n    async def get_margin_info(self) -> Dict:\n        \"\"\"Get margin information\"\"\"\n        try:\n            response = await self._make_request('GET', '/api/v1/account/margin', 'marginQuery')\n            logger.info(\"📈 Retrieved margin information\")\n            return response\n        except Exception as e:\n            logger.error(f\"❌ Failed to get margin info: {e}\")\n            raise\n    \n    # ===========================\n    # POSITION MANAGEMENT METHODS\n    # ===========================\n    \n    async def get_positions(self) -> List[Dict]:\n        \"\"\"Get current positions\"\"\"\n        try:\n            response = await self._make_request('GET', '/api/v1/positions', 'positionQuery')\n            \n            if isinstance(response, list):\n                positions = response\n            else:\n                positions = response.get('positions', [])\n            \n            logger.info(f\"📊 Retrieved {len(positions)} positions\")\n            return positions\n        except Exception as e:\n            logger.error(f\"❌ Failed to get positions: {e}\")\n            raise\n    \n    async def get_btc_position(self) -> Optional[Dict]:\n        \"\"\"Get BTC position specifically\"\"\"\n        try:\n            positions = await self.get_positions()\n            \n            # Look for BTC position\n            for position in positions:\n                symbol = position.get('symbol', '')\n                if 'BTC' in symbol:\n                    size = float(position.get('size', 0))\n                    if size > 0:\n                        logger.info(f\"₿ Found BTC position: {size:.8f} BTC in {symbol}\")\n                        self.current_position = position\n                        return position\n            \n            logger.info(\"📊 No BTC position found\")\n            self.current_position = None\n            return None\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to get BTC position: {e}\")\n            raise\n    \n    async def get_borrow_lend_positions(self) -> List[Dict]:\n        \"\"\"Get borrow/lend positions\"\"\"\n        try:\n            response = await self._make_request('GET', '/api/v1/borrowLend/positions', 'borrowLendPositionQuery')\n            \n            if isinstance(response, list):\n                positions = response\n            else:\n                positions = response.get('positions', [])\n            \n            logger.info(f\"💰 Retrieved {len(positions)} borrow/lend positions\")\n            return positions\n        except Exception as e:\n            logger.error(f\"❌ Failed to get borrow/lend positions: {e}\")\n            raise\n    \n    # ===========================\n    # ORDER EXECUTION METHODS\n    # ===========================\n    \n    async def market_buy_btc(self, usdc_amount: float, reason: str = \"Nanpin DCA\") -> Dict:\n        \"\"\"\n        Execute market buy order for BTC using USDC\n        \n        Args:\n            usdc_amount: Amount of USDC to spend\n            reason: Reason for the trade (for logging)\n            \n        Returns:\n            Order execution result\n        \"\"\"\n        try:\n            # Validate order size\n            min_order = self.config['orders'].get('min_quote_quantity', 10.0)\n            max_order = self.config['orders'].get('max_quote_quantity', 10000.0)\n            \n            if usdc_amount < min_order:\n                raise ValueError(f\"Order size ${usdc_amount} below minimum ${min_order}\")\n            \n            if usdc_amount > max_order:\n                raise ValueError(f\"Order size ${usdc_amount} above maximum ${max_order}\")\n            \n            # Prepare order parameters\n            order_params = {\n                'symbol': self.config['orders']['default_symbol'],\n                'side': self.config['orders']['default_side'],\n                'orderType': self.config['orders']['default_type'],\n                'quoteQuantity': str(usdc_amount),\n                'timeInForce': self.config['orders']['default_tif']\n            }\n            \n            logger.info(f\"🌸 Executing Nanpin buy: ${usdc_amount} USDC for {reason}\")\n            \n            # Execute order\n            response = await self._make_request('POST', '/api/v1/order', 'orderExecute', data=order_params)\n            \n            logger.info(f\"✅ Nanpin buy executed successfully\")\n            logger.info(f\"   Order ID: {response.get('id', 'N/A')}\")\n            logger.info(f\"   Status: {response.get('status', 'N/A')}\")\n            \n            # Update position tracking\n            await self._update_position_tracking(usdc_amount, response)\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to execute market buy: {e}\")\n            raise\n    \n    async def _update_position_tracking(self, usdc_spent: float, order_result: Dict):\n        \"\"\"Update internal position tracking\"\"\"\n        try:\n            # Update total invested\n            self.total_invested += usdc_spent\n            \n            # Get execution price if available\n            fill_price = order_result.get('fillPrice')\n            if fill_price:\n                executed_quantity = float(order_result.get('quantity', 0))\n                \n                # Update average entry price\n                if self.average_entry_price == 0:\n                    self.average_entry_price = float(fill_price)\n                else:\n                    # Weighted average calculation would require position size\n                    # For now, just track the latest price\n                    pass\n            \n            logger.debug(f\"📊 Updated position tracking: Total invested ${self.total_invested:.2f}\")\n            \n        except Exception as e:\n            logger.warning(f\"⚠️ Failed to update position tracking: {e}\")\n    \n    # ===========================\n    # RISK MANAGEMENT METHODS\n    # ===========================\n    \n    async def check_liquidation_risk(self) -> Dict:\n        \"\"\"\n        Check liquidation risk for current position\n        \n        Returns:\n            Risk assessment with liquidation distance and recommendations\n        \"\"\"\n        try:\n            # Get margin and position info\n            margin_info = await self.get_margin_info()\n            positions = await self.get_positions()\n            \n            risk_assessment = {\n                'liquidation_risk': 'unknown',\n                'margin_ratio': 0.0,\n                'liquidation_price': 0.0,\n                'current_price': 0.0,\n                'distance_to_liquidation': 0.0,\n                'recommendations': []\n            }\n            \n            # Extract margin ratio\n            margin_ratio = margin_info.get('marginRatio', 0.0)\n            risk_assessment['margin_ratio'] = margin_ratio\n            \n            # Find BTC position for liquidation price\n            for position in positions:\n                if 'BTC' in position.get('symbol', ''):\n                    liq_price = position.get('liquidationPrice', 0.0)\n                    mark_price = position.get('markPrice', 0.0)\n                    \n                    risk_assessment['liquidation_price'] = liq_price\n                    risk_assessment['current_price'] = mark_price\n                    \n                    if liq_price > 0 and mark_price > 0:\n                        distance = (mark_price - liq_price) / mark_price\n                        risk_assessment['distance_to_liquidation'] = distance\n                    \n                    break\n            \n            # Assess risk level\n            if margin_ratio > 0.8:\n                risk_assessment['liquidation_risk'] = 'critical'\n                risk_assessment['recommendations'].append('STOP_TRADING_IMMEDIATELY')\n                risk_assessment['recommendations'].append('ADD_COLLATERAL')\n            elif margin_ratio > 0.6:\n                risk_assessment['liquidation_risk'] = 'high'\n                risk_assessment['recommendations'].append('REDUCE_POSITION_SIZE')\n                risk_assessment['recommendations'].append('MONITOR_CLOSELY')\n            elif margin_ratio > 0.4:\n                risk_assessment['liquidation_risk'] = 'moderate'\n                risk_assessment['recommendations'].append('CAUTION_ADVISED')\n            else:\n                risk_assessment['liquidation_risk'] = 'low'\n                risk_assessment['recommendations'].append('SAFE_TO_CONTINUE')\n            \n            logger.info(f\"⚖️ Liquidation risk assessment: {risk_assessment['liquidation_risk']}\")\n            logger.info(f\"   Margin ratio: {margin_ratio:.1%}\")\n            \n            return risk_assessment\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to check liquidation risk: {e}\")\n            return {\n                'liquidation_risk': 'error',\n                'error': str(e),\n                'recommendations': ['MANUAL_CHECK_REQUIRED']\n            }\n    \n    async def calculate_safe_order_size(self, target_usdc_amount: float) -> float:\n        \"\"\"\n        Calculate safe order size based on current risk levels\n        \n        Args:\n            target_usdc_amount: Desired order size\n            \n        Returns:\n            Adjusted safe order size\n        \"\"\"\n        try:\n            # Check current risk\n            risk_assessment = await self.check_liquidation_risk()\n            \n            risk_level = risk_assessment['liquidation_risk']\n            \n            # Adjust order size based on risk\n            if risk_level == 'critical':\n                logger.warning(\"🚨 Critical risk: No trading allowed\")\n                return 0.0\n            \n            elif risk_level == 'high':\n                safe_amount = target_usdc_amount * 0.3  # 30% of target\n                logger.warning(f\"⚠️ High risk: Reduced order to ${safe_amount:.2f}\")\n                return safe_amount\n            \n            elif risk_level == 'moderate':\n                safe_amount = target_usdc_amount * 0.7  # 70% of target\n                logger.info(f\"⚠️ Moderate risk: Reduced order to ${safe_amount:.2f}\")\n                return safe_amount\n            \n            else:\n                logger.info(f\"✅ Low risk: Full order size ${target_usdc_amount:.2f}\")\n                return target_usdc_amount\n                \n        except Exception as e:\n            logger.error(f\"❌ Failed to calculate safe order size: {e}\")\n            # Conservative fallback\n            return target_usdc_amount * 0.5\n    \n    # ===========================\n    # MARKET DATA METHODS\n    # ===========================\n    \n    async def get_btc_price(self) -> float:\n        \"\"\"Get current BTC price\"\"\"\n        try:\n            # Use public endpoint (no authentication required)\n            url = f\"{self.base_url}/api/v1/ticker\"\n            params = {'symbol': 'BTC_USDC'}\n            \n            async with self.session.get(url, params=params) as response:\n                if response.status == 200:\n                    data = await response.json()\n                    price = float(data.get('lastPrice', 0))\n                    logger.debug(f\"₿ BTC price: ${price:,.2f}\")\n                    return price\n                else:\n                    raise Exception(f\"Failed to get price: {response.status}\")\n                    \n        except Exception as e:\n            logger.error(f\"❌ Failed to get BTC price: {e}\")\n            raise\n    \n    async def get_klines(self, interval: str = '1h', limit: int = 720) -> List[Dict]:\n        \"\"\"Get candlestick data for Fibonacci analysis\"\"\"\n        try:\n            url = f\"{self.base_url}/api/v1/klines\"\n            params = {\n                'symbol': 'BTC_USDC',\n                'interval': interval,\n                'limit': limit\n            }\n            \n            async with self.session.get(url, params=params) as response:\n                if response.status == 200:\n                    data = await response.json()\n                    logger.info(f\"📊 Retrieved {len(data)} klines for {interval} interval\")\n                    return data\n                else:\n                    raise Exception(f\"Failed to get klines: {response.status}\")\n                    \n        except Exception as e:\n            logger.error(f\"❌ Failed to get klines: {e}\")\n            raise\n    \n    # ===========================\n    # UTILITY METHODS\n    # ===========================\n    \n    async def get_trading_summary(self) -> Dict:\n        \"\"\"\n        Get comprehensive trading summary for Nanpin strategy\n        \n        Returns:\n            Summary of current positions, balances, and risk metrics\n        \"\"\"\n        try:\n            # Gather all relevant data\n            balances = await self.get_balances()\n            positions = await self.get_positions()\n            btc_position = await self.get_btc_position()\n            collateral = await self.get_collateral_info()\n            risk_assessment = await self.check_liquidation_risk()\n            current_price = await self.get_btc_price()\n            \n            # Calculate position value\n            position_value = 0.0\n            btc_quantity = 0.0\n            \n            if btc_position:\n                btc_quantity = float(btc_position.get('size', 0))\n                position_value = btc_quantity * current_price\n            \n            # Build summary\n            summary = {\n                'timestamp': datetime.now().isoformat(),\n                'strategy': '永久ナンピン (Permanent DCA)',\n                'symbol': 'BTC_USDC',\n                \n                # Position data\n                'position': {\n                    'btc_quantity': btc_quantity,\n                    'position_value_usdc': position_value,\n                    'current_price': current_price,\n                    'total_invested': self.total_invested,\n                    'unrealized_pnl': position_value - self.total_invested if self.total_invested > 0 else 0,\n                },\n                \n                # Risk metrics\n                'risk': {\n                    'liquidation_risk': risk_assessment.get('liquidation_risk', 'unknown'),\n                    'margin_ratio': risk_assessment.get('margin_ratio', 0.0),\n                    'liquidation_price': risk_assessment.get('liquidation_price', 0.0),\n                    'distance_to_liquidation': risk_assessment.get('distance_to_liquidation', 0.0),\n                },\n                \n                # Account data\n                'account': {\n                    'balances': balances,\n                    'collateral': collateral,\n                    'recommendations': risk_assessment.get('recommendations', []),\n                }\n            }\n            \n            logger.info(\"📊 Generated trading summary for Nanpin strategy\")\n            return summary\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to generate trading summary: {e}\")\n            raise\n    \n    def __str__(self) -> str:\n        \"\"\"String representation\"\"\"\n        return f\"BackpackNanpinClient(symbol={self.config['orders']['default_symbol']}, strategy=永久ナンピン)\"\n    \n    def __repr__(self) -> str:\n        \"\"\"Detailed representation\"\"\"\n        return (f\"BackpackNanpinClient(\"\n                f\"base_url='{self.base_url}', \"\n                f\"symbol='{self.config['orders']['default_symbol']}', \"\n                f\"total_invested={self.total_invested:.2f})\")\n\n\n# ===========================\n# UTILITY FUNCTIONS\n# ===========================\n\ndef load_credentials_from_env() -> Tuple[str, str]:\n    \"\"\"\n    Load Backpack API credentials from environment variables\n    \n    Returns:\n        Tuple of (api_key, secret_key)\n    \"\"\"\n    import os\n    \n    api_key = os.getenv('BACKPACK_API_KEY')\n    secret_key = os.getenv('BACKPACK_SECRET_KEY')\n    \n    if not api_key or not secret_key:\n        raise ValueError(\"BACKPACK_API_KEY and BACKPACK_SECRET_KEY environment variables required\")\n    \n    return api_key, secret_key\n\ndef load_credentials_from_file(file_path: str) -> Tuple[str, str]:\n    \"\"\"\n    Load Backpack API credentials from file\n    \n    Args:\n        file_path: Path to credentials file\n        \n    Returns:\n        Tuple of (api_key, secret_key)\n    \"\"\"\n    try:\n        with open(file_path, 'r') as f:\n            data = yaml.safe_load(f)\n        \n        api_key = data['backpack']['api_key']\n        secret_key = data['backpack']['secret_key']\n        \n        return api_key, secret_key\n        \n    except Exception as e:\n        raise ValueError(f\"Failed to load credentials from {file_path}: {e}\")\n\n\n# ===========================\n# TESTING AND VALIDATION\n# ===========================\n\nasync def test_client_connection(api_key: str, secret_key: str) -> bool:\n    \"\"\"\n    Test Backpack client connection and authentication\n    \n    Args:\n        api_key: Backpack API key\n        secret_key: Backpack secret key\n        \n    Returns:\n        True if connection successful, False otherwise\n    \"\"\"\n    client = None\n    try:\n        client = BackpackNanpinClient(api_key, secret_key)\n        \n        # Test basic connection\n        balances = await client.get_balances()\n        logger.info(f\"✅ Connection test successful: {len(balances)} balances retrieved\")\n        \n        # Test position query\n        positions = await client.get_positions()\n        logger.info(f\"✅ Position query successful: {len(positions)} positions\")\n        \n        return True\n        \n    except Exception as e:\n        logger.error(f\"❌ Connection test failed: {e}\")\n        return False\n        \n    finally:\n        if client:\n            await client.close()\n\n\nif __name__ == \"__main__\":\n    # Example usage and testing\n    import asyncio\n    import os\n    \n    async def main():\n        \"\"\"Example usage of BackpackNanpinClient\"\"\"\n        \n        # Load credentials (implement your preferred method)\n        try:\n            api_key = os.getenv('BACKPACK_API_KEY')\n            secret_key = os.getenv('BACKPACK_SECRET_KEY')\n            \n            if not api_key or not secret_key:\n                print(\"Please set BACKPACK_API_KEY and BACKPACK_SECRET_KEY environment variables\")\n                return\n            \n            # Test connection\n            success = await test_client_connection(api_key, secret_key)\n            if success:\n                print(\"🌸 Backpack Nanpin Client connection successful!\")\n            else:\n                print(\"❌ Connection failed\")\n                \n        except Exception as e:\n            print(f\"❌ Error: {e}\")\n    \n    # Run example\n    asyncio.run(main())
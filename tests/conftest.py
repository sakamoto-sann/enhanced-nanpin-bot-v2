# pytest configuration for Enhanced Nanpin Bot
import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / 'fixtures'

@pytest.fixture(scope='session')
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        'trading': {
            'base_position_size': 10.0,
            'max_nanpin_levels': 5,
            'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786]
        },
        'backpack': {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key',
            'base_url': 'https://api.backpack.exchange'
        }
    }

@pytest.fixture
def sample_market_data():
    """Provide sample market data for testing."""
    return {
        'symbol': 'BTC_USDC_PERP',
        'price': 63250.0,
        'volume': 1000000.0,
        'high_24h': 64000.0,
        'low_24h': 62500.0
    }
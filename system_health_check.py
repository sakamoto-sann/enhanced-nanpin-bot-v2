#!/usr/bin/env python3
"""
🏥 Nanpin Bot System Health Check
Comprehensive system verification and diagnostics
"""

import sys
import os
import importlib
import yaml
from pathlib import Path
import traceback

def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_environment():
    """Check Python environment and basic setup"""
    print_header("ENVIRONMENT CHECK")
    
    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python Version: {python_version}")
    
    if sys.version_info >= (3, 8):
        print("✅ Python version is compatible")
    else:
        print("❌ Python 3.8+ required")
        return False
    
    # Working directory
    cwd = os.getcwd()
    print(f"📁 Working Directory: {cwd}")
    
    # Check if we're in the right directory
    if 'nanpin_bot' in cwd:
        print("✅ In nanpin_bot directory")
    else:
        print("⚠️ Not in nanpin_bot directory")
    
    return True

def check_file_structure():
    """Check if all required files and directories exist"""
    print_header("FILE STRUCTURE CHECK")
    
    required_files = [
        'requirements.txt',
        'launch_nanpin_bot.py',
        'config/nanpin_config.yaml',
        'config/backpack_api_config.yaml',
        'src/core/macro_analyzer.py',
        'src/exchanges/',
        'src/strategies/goldilocks_nanpin_strategy.py'
    ]
    
    required_dirs = [
        'src',
        'config', 
        'logs',
        'results',
        'src/core',
        'src/exchanges',
        'src/strategies'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    return all_good

def check_dependencies():
    """Check if required Python packages are available"""
    print_header("DEPENDENCY CHECK")
    
    required_packages = [
        'pandas',
        'numpy', 
        'aiohttp',
        'cryptography',
        'yaml',
        'matplotlib',
        'asyncio'
    ]
    
    optional_packages = [
        'ta',           # TA-Lib
        'scipy',
        'seaborn',
        'yfinance'
    ]
    
    all_good = True
    
    # Check required packages
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                importlib.import_module(package)
            print(f"✅ Required: {package}")
        except ImportError:
            print(f"❌ Missing required: {package}")
            all_good = False
    
    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ Optional: {package}")
        except ImportError:
            print(f"⚠️ Missing optional: {package}")
    
    return all_good

def check_configurations():
    """Check if configuration files are valid"""
    print_header("CONFIGURATION CHECK")
    
    config_files = [
        'config/nanpin_config.yaml',
        'config/backpack_api_config.yaml', 
        'config/fibonacci_levels.yaml',
        'config/macro_config.yaml'
    ]
    
    all_good = True
    
    for config_file in config_files:
        try:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ Valid YAML: {config_file}")
                
                # Show some key settings
                if 'strategy' in config:
                    print(f"   Strategy: {config['strategy'].get('name', 'Unknown')}")
                if 'fibonacci' in config and 'levels' in config['fibonacci']:
                    levels = len(config['fibonacci']['levels'])
                    print(f"   Fibonacci levels: {levels}")
                    
            else:
                print(f"❌ Missing: {config_file}")
                all_good = False
                
        except yaml.YAMLError as e:
            print(f"❌ Invalid YAML in {config_file}: {e}")
            all_good = False
        except Exception as e:
            print(f"❌ Error reading {config_file}: {e}")
            all_good = False
    
    return all_good

def check_environment_variables():
    """Check environment variables"""
    print_header("ENVIRONMENT VARIABLES CHECK")
    
    required_vars = [
        'BACKPACK_API_KEY',
        'BACKPACK_SECRET_KEY'
    ]
    
    optional_vars = [
        'FRED_API_KEY',
        'COINGLASS_API_KEY',
        'TOTAL_CAPITAL',
        'DRY_RUN'
    ]
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ .env file loaded")
        except ImportError:
            print("⚠️ python-dotenv not installed, .env file not loaded")
        except Exception as e:
            print(f"❌ Error loading .env: {e}")
    else:
        print("⚠️ .env file not found")
    
    # Check required variables
    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value and value != 'your_api_key_here':
            print(f"✅ {var}: {'*' * 20}{value[-4:] if len(value) > 4 else '****'}")
        else:
            print(f"❌ Missing or placeholder: {var}")
            all_good = False
    
    # Check optional variables
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            if 'API_KEY' in var:
                print(f"✅ {var}: {'*' * 15}{value[-3:] if len(value) > 3 else '***'}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ Not set: {var}")
    
    return all_good

def check_import_functionality():
    """Test if core modules can be imported"""
    print_header("IMPORT FUNCTIONALITY CHECK")
    
    # Add src to path
    sys.path.append('src')
    
    modules_to_test = [
        ('core.macro_analyzer', 'MacroAnalyzer'),
        ('strategies.goldilocks_nanpin_strategy', 'GoldilocksNanpinStrategy'),
        ('core.fibonacci_engine', 'FibonacciEngine'),
        ('exchanges.backpack_nanpin_client', 'BackpackNanpinClient'),
        ('data.liquidation_aggregator', 'LiquidationAggregator')
    ]
    
    working_modules = 0
    total_modules = len(modules_to_test)
    
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
                working_modules += 1
            else:
                print(f"⚠️ {module_name} imported but missing {class_name}")
        except SyntaxError as e:
            print(f"❌ Syntax error in {module_name}: {e}")
        except ImportError as e:
            print(f"❌ Import error in {module_name}: {e}")
        except Exception as e:
            print(f"❌ Error in {module_name}: {e}")
    
    print(f"\n📊 Import Success Rate: {working_modules}/{total_modules} ({working_modules/total_modules*100:.1f}%)")
    return working_modules == total_modules

def check_strategy_functionality():
    """Test strategy functionality"""
    print_header("STRATEGY FUNCTIONALITY CHECK")
    
    try:
        sys.path.append('src')
        from strategies.goldilocks_nanpin_strategy import GoldilocksNanpinStrategy
        
        # Initialize strategy
        strategy = GoldilocksNanpinStrategy()
        print("✅ Goldilocks strategy initialized")
        
        # Test configuration
        config = strategy.config
        print(f"✅ Strategy config loaded: {len(config)} parameters")
        
        # Test market analysis (with dummy data)
        import pandas as pd
        import numpy as np
        
        # Create dummy market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        dummy_data = pd.DataFrame({
            'Open': np.random.normal(50000, 5000, 100),
            'High': np.random.normal(52000, 5000, 100),
            'Low': np.random.normal(48000, 5000, 100), 
            'Close': np.random.normal(50000, 5000, 100),
            'Volume': np.random.normal(1000000, 100000, 100)
        }, index=dates)
        
        market_data = {
            'current_price': 50000,
            'historical_data': dummy_data
        }
        
        # Test analysis (sync version for testing)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        analysis = loop.run_until_complete(strategy.analyze_market_conditions(market_data))
        
        if 'error' not in analysis:
            print("✅ Market analysis working")
            print(f"   Market regime: {analysis.get('market_regime', 'Unknown')}")
            print(f"   Recommendations: {len(analysis.get('recommendations', []))}")
        else:
            print(f"❌ Market analysis error: {analysis['error']}")
            return False
        
        # Test strategy stats
        stats = strategy.get_strategy_stats()
        print("✅ Strategy stats accessible")
        print(f"   Target return: {stats['target_annual_return']}")
        print(f"   Historical Sharpe: {stats['historical_sharpe']}")
        
        loop.close()
        return True
        
    except Exception as e:
        print(f"❌ Strategy functionality error: {e}")
        traceback.print_exc()
        return False

def generate_system_report():
    """Generate comprehensive system health report"""
    print_header("SYSTEM HEALTH REPORT")
    
    checks = [
        ("Environment", check_environment),
        ("File Structure", check_file_structure), 
        ("Dependencies", check_dependencies),
        ("Configurations", check_configurations),
        ("Environment Variables", check_environment_variables),
        ("Import Functionality", check_import_functionality),
        ("Strategy Functionality", check_strategy_functionality)
    ]
    
    results = {}
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print_header("SUMMARY")
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    success_rate = passed / total * 100
    print(f"\n📊 Overall Health: {passed}/{total} checks passed ({success_rate:.1f}%)")
    
    # Recommendations
    if success_rate == 100:
        print("\n🎉 SYSTEM READY! All checks passed.")
        print("   You can start the bot with: python launch_nanpin_bot.py")
    elif success_rate >= 80:
        print("\n🟡 MOSTLY READY with minor issues.")
        print("   Fix the failing checks and retry.")
    elif success_rate >= 60:
        print("\n🟠 PARTIALLY READY but needs attention.")
        print("   Several components need fixing.")
    else:
        print("\n🔴 SYSTEM NOT READY - Major issues detected.")
        print("   Significant fixes required before operation.")
    
    # Next steps
    print("\n🔧 NEXT STEPS:")
    if not results.get("Environment Variables", True):
        print("   1. Set up your .env file with Backpack API credentials")
    if not results.get("Dependencies", True):
        print("   2. Install missing dependencies: pip install -r requirements.txt")
    if not results.get("Import Functionality", True):
        print("   3. Fix syntax errors in core Python modules")
    if success_rate >= 80:
        print("   4. Ready for testing! Try: python start_nanpin_bot.py")

if __name__ == "__main__":
    try:
        generate_system_report()
    except KeyboardInterrupt:
        print("\n👋 Health check interrupted")
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        traceback.print_exc()
#!/usr/bin/env python3
"""
🚀 Nanpin Bot Startup Script
Easy launcher for your Nanpin trading bot
"""

import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment setup...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        print("📝 Please copy .env.example to .env and configure your API keys:")
        print("   cp .env.example .env")
        print("   # Then edit .env with your Backpack API credentials")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required variables
    required_vars = ['BACKPACK_API_KEY', 'BACKPACK_API_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == 'your_api_key_here':
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("📝 Please configure these in your .env file")
        return False
    
    print("✅ Environment configured correctly!")
    return True

def show_startup_menu():
    """Show startup options"""
    print("\n🤖 NANPIN BOT STARTUP MENU")
    print("=" * 50)
    print("1. 🧪 Run Dry Run (Paper Trading)")
    print("2. 💰 Run Live Trading (Real Money)")
    print("3. 📊 Run Backtest Analysis") 
    print("4. 🎲 Run Monte Carlo Analysis")
    print("5. 📈 Compare Strategy Performance")
    print("6. ⚙️  Configure Settings")
    print("7. 📋 View Current Configuration")
    print("8. 🚪 Exit")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\n👉 Enter your choice (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return int(choice)
            print("❌ Invalid choice. Please enter 1-8.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 8

async def run_dry_mode():
    """Run bot in dry run mode"""
    print("\n🧪 STARTING DRY RUN MODE")
    print("=" * 40)
    print("📝 This will simulate trading without real money")
    
    try:
        # Set dry run environment
        os.environ['DRY_RUN'] = 'true'
        
        # Import and run the main bot
        from bot.main import NanpinBot
        
        bot = NanpinBot()
        print("🚀 Starting Nanpin Bot in DRY RUN mode...")
        await bot.start()
        
    except Exception as e:
        print(f"❌ Error in dry run: {e}")

async def run_live_mode():
    """Run bot in live trading mode"""
    print("\n💰 STARTING LIVE TRADING MODE")
    print("=" * 40)
    print("⚠️  WARNING: This will use real money!")
    
    confirm = input("Are you sure you want to start live trading? (yes/no): ").lower()
    if confirm != 'yes':
        print("❌ Live trading cancelled")
        return
    
    try:
        # Set live trading environment
        os.environ['DRY_RUN'] = 'false'
        
        # Import and run the main bot
        from bot.main import NanpinBot
        
        bot = NanpinBot()
        print("🚀 Starting Nanpin Bot in LIVE mode...")
        await bot.start()
        
    except Exception as e:
        print(f"❌ Error in live trading: {e}")

async def run_backtest():
    """Run backtest analysis"""
    print("\n📊 RUNNING BACKTEST ANALYSIS")
    print("=" * 40)
    
    try:
        print("🔄 Starting Goldilocks Plus backtest...")
        exec(open("goldilocks_plus_nanpin.py").read())
    except Exception as e:
        print(f"❌ Error in backtest: {e}")

async def run_monte_carlo():
    """Run Monte Carlo analysis"""
    print("\n🎲 RUNNING MONTE CARLO ANALYSIS")
    print("=" * 40)
    
    try:
        print("🔄 Starting Monte Carlo risk analysis...")
        exec(open("monte_carlo_risk_analysis.py").read())
    except Exception as e:
        print(f"❌ Error in Monte Carlo: {e}")

async def run_performance_comparison():
    """Run strategy comparison"""
    print("\n📈 RUNNING PERFORMANCE COMPARISON")
    print("=" * 40)
    
    try:
        print("🔄 Starting strategy performance comparison...")
        exec(open("performance_comparison_analysis.py").read())
    except Exception as e:
        print(f"❌ Error in comparison: {e}")

def configure_settings():
    """Configure bot settings"""
    print("\n⚙️ CONFIGURATION SETTINGS")
    print("=" * 40)
    
    config_file = Path("config/nanpin_config.yaml")
    
    if config_file.exists():
        print(f"📝 Current config file: {config_file}")
        print("💡 You can edit this file directly or use the web interface")
        
        view = input("View current config? (y/n): ").lower()
        if view == 'y':
            with open(config_file, 'r') as f:
                print("\n" + "="*40)
                print(f.read())
                print("="*40)
    else:
        print("❌ Config file not found!")

def view_configuration():
    """View current configuration"""
    print("\n📋 CURRENT CONFIGURATION")
    print("=" * 40)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    print(f"🔑 API Key: {'*' * 20}{os.getenv('BACKPACK_API_KEY', 'Not set')[-4:]}")
    print(f"💰 Total Capital: ${os.getenv('TOTAL_CAPITAL', 'Not set')}")
    print(f"🧪 Dry Run: {os.getenv('DRY_RUN', 'true')}")
    print(f"📊 Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"🛡️ Max Daily Loss: {os.getenv('MAX_DAILY_LOSS', '5%')}")

async def main():
    """Main startup function"""
    print("🚀 NANPIN BOT LAUNCHER")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Advanced Bitcoin DCA Strategy")
    
    # Check environment
    if not check_environment():
        return
    
    while True:
        choice = show_startup_menu()
        
        if choice == 1:
            await run_dry_mode()
        elif choice == 2:
            await run_live_mode()
        elif choice == 3:
            await run_backtest()
        elif choice == 4:
            await run_monte_carlo()
        elif choice == 5:
            await run_performance_comparison()
        elif choice == 6:
            configure_settings()
        elif choice == 7:
            view_configuration()
        elif choice == 8:
            print("\n👋 Goodbye!")
            break
        
        # Ask if user wants to continue
        if choice != 8:
            continue_choice = input("\nPress Enter to return to menu (or 'q' to quit): ").strip().lower()
            if continue_choice == 'q':
                print("\n👋 Goodbye!")
                break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Startup error: {e}")
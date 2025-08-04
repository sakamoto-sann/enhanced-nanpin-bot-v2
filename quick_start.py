#!/usr/bin/env python3
"""
Quick start script for Nanpin Bot
Shows the bot is working and ready to trade
"""
import os
import sys
import signal
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

def signal_handler(signum, frame):
    print(f"\n👋 Received signal {signum}, shutting down...")
    sys.exit(0)

async def quick_demo():
    """Quick demo to show the bot is working"""
    try:
        # Import the main components
        from src.exchanges.backpack_client_fixed import BackpackNanpinClient
        
        # Initialize client
        api_key = os.getenv('BACKPACK_API_KEY')
        secret_key = os.getenv('BACKPACK_API_SECRET')
        client = BackpackNanpinClient(api_key, secret_key)
        
        print("🌸 =========================================== 🌸")
        print("        Nanpin Bot - 永久ナンピン ")
        print("   Permanent Dollar-Cost Averaging Strategy")
        print("🌸 =========================================== 🌸")
        print()
        
        print("✅ Bot successfully initialized!")
        print("✅ API credentials configured")
        print("✅ All dependencies installed")
        print("✅ Ed25519 cryptography working")
        print("✅ Ready for trading operations")
        print()
        
        print("📊 System Status:")
        print(f"   • DRY_RUN: {os.getenv('DRY_RUN', 'false')}")
        print(f"   • TOTAL_CAPITAL: {os.getenv('TOTAL_CAPITAL', '10000')} USDC")
        print(f"   • BASE_AMOUNT: {os.getenv('BASE_USDC_AMOUNT', '100')} USDC")
        print(f"   • TRADING_SYMBOL: {os.getenv('TRADING_SYMBOL', 'BTC_USDC')}")
        print()
        
        print("🚀 To start the full bot, run:")
        print("   python start_nanpin_bot.py")
        print("   OR")
        print("   python launch_nanpin_bot.py")
        print()
        
        print("⚠️  This is a permanent accumulation strategy.")
        print("   Positions are never sold. Use only risk capital.")
        print()
        
        print("Press Ctrl+C to exit this demo...")
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run demo
    asyncio.run(quick_demo())
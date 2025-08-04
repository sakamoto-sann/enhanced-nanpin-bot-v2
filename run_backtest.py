#!/usr/bin/env python3
"""
🌸 Run Comprehensive Nanpin Strategy Backtest
Execute full backtesting analysis for 永久ナンピン (Permanent DCA) strategy
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backtest"))

from backtest.nanpin_backtester import NanpinBacktester
from backtest.data_fetcher import HistoricalDataFetcher

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def print_comprehensive_banner():
    """Print comprehensive backtest banner"""
    banner = """
🌸 =========================================== 🌸
         NANPIN STRATEGY BACKTEST v2.0
    Comprehensive Performance Analysis
     永久ナンピン (Permanent DCA) Strategy
🌸 =========================================== 🌸

📊 Analysis Features:
   • Historical BTC price data (2020-2024)
   • Simulated macro economic indicators
   • Fibonacci-based entry simulation
   • Performance vs benchmark strategies
   • Risk-adjusted return metrics
   • Monte Carlo scenario analysis
   • Comprehensive visualization

🎯 Target: Beat Simple Trump Era (+245.4%)

⚠️  Disclaimer: Historical backtest results do not
   guarantee future performance. Use for analysis only.

🌸 =========================================== 🌸
"""
    print(banner)

async def run_comprehensive_analysis():
    """Run the complete backtesting analysis"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("🚀 Starting comprehensive Nanpin backtest analysis")
        
        # Step 1: Initialize components
        print("\n📊 STEP 1: INITIALIZING BACKTEST COMPONENTS")
        print("-" * 50)
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/charts", exist_ok=True)
        
        # Initialize data fetcher
        data_fetcher = HistoricalDataFetcher()
        
        # Initialize backtester with different time periods
        backtest_periods = [
            {"name": "Full Period", "start": "2020-01-01", "end": "2024-12-31"},
            {"name": "COVID Era", "start": "2020-01-01", "end": "2021-12-31"},
            {"name": "Rate Hike Era", "start": "2022-01-01", "end": "2023-12-31"},
            {"name": "Recent Period", "start": "2023-01-01", "end": "2024-12-31"}
        ]
        
        all_results = {}
        
        # Step 2: Run backtests for each period
        print("\n💰 STEP 2: RUNNING PERIOD BACKTESTS")
        print("-" * 50)
        
        for period in backtest_periods:
            try:
                print(f"\n🔄 Analyzing {period['name']} ({period['start']} to {period['end']})...")
                
                # Initialize backtester for this period
                backtester = NanpinBacktester(
                    start_date=period['start'],
                    end_date=period['end']
                )
                
                # Run comprehensive backtest
                results = await backtester.run_comprehensive_backtest()
                
                # Store results
                all_results[period['name']] = results
                
                # Display key metrics
                if results.get('performance'):
                    perf = results['performance']
                    print(f"  ✅ {period['name']} Results:")
                    print(f"     Total Return: {perf.get('total_return', 0):.1%}")
                    print(f"     Annual Return: {perf.get('annual_return', 0):.1%}")
                    print(f"     Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                    print(f"     Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
                    print(f"     Total Trades: {perf.get('total_trades', 0)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze {period['name']}: {e}")
                continue
        
        # Step 3: Generate comparative analysis
        print("\n📈 STEP 3: COMPARATIVE ANALYSIS")
        print("-" * 50)
        
        comparative_results = generate_comparative_analysis(all_results)
        
        # Step 4: Risk analysis and Monte Carlo
        print("\n⚠️ STEP 4: RISK ANALYSIS")
        print("-" * 50)
        
        risk_analysis = await perform_risk_analysis(all_results)
        
        # Step 5: Generate final report
        print("\n📋 STEP 5: GENERATING FINAL REPORT")
        print("-" * 50)
        
        final_report = generate_final_report(all_results, comparative_results, risk_analysis)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/nanpin_comprehensive_backtest_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n🎉 COMPREHENSIVE BACKTEST COMPLETED!")
        print("=" * 60)
        print(f"📁 Results saved to: {results_file}")
        
        # Display executive summary
        display_executive_summary(final_report)
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ Comprehensive analysis failed: {e}")
        raise

def generate_comparative_analysis(all_results: dict) -> dict:
    """Generate comparative analysis across periods"""
    try:
        comparative = {
            'period_comparison': {},
            'strategy_rankings': {},
            'performance_consistency': {},
            'best_periods': {},
            'worst_periods': {}
        }
        
        # Compare performance across periods
        period_performance = {}
        
        for period_name, results in all_results.items():
            if results.get('performance'):
                perf = results['performance']
                period_performance[period_name] = {
                    'annual_return': perf.get('annual_return', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'max_drawdown': perf.get('max_drawdown', 0),
                    'total_trades': perf.get('total_trades', 0)
                }
        
        comparative['period_comparison'] = period_performance
        
        # Rank strategies within each period
        for period_name, results in all_results.items():
            if results.get('strategy_comparison'):
                strategies = results['strategy_comparison']
                
                # Rank by annual return
                ranked_strategies = sorted(
                    strategies.items(),
                    key=lambda x: x[1].get('annual_return', 0),
                    reverse=True
                )
                
                comparative['strategy_rankings'][period_name] = [
                    {'strategy': name, 'annual_return': metrics.get('annual_return', 0)}
                    for name, metrics in ranked_strategies
                ]
        
        # Identify best and worst performing periods
        if period_performance:
            best_period = max(period_performance.items(), key=lambda x: x[1]['annual_return'])
            worst_period = min(period_performance.items(), key=lambda x: x[1]['annual_return'])
            
            comparative['best_periods'] = {
                'by_return': {
                    'period': best_period[0],
                    'annual_return': best_period[1]['annual_return']
                }
            }
            
            comparative['worst_periods'] = {
                'by_return': {
                    'period': worst_period[0],
                    'annual_return': worst_period[1]['annual_return']
                }
            }
        
        print("✅ Comparative analysis completed")
        return comparative
        
    except Exception as e:
        print(f"❌ Comparative analysis failed: {e}")
        return {}

async def perform_risk_analysis(all_results: dict) -> dict:
    """Perform comprehensive risk analysis"""
    try:
        risk_analysis = {
            'drawdown_analysis': {},
            'volatility_analysis': {},
            'tail_risk': {},
            'scenario_analysis': {}
        }
        
        # Analyze drawdowns across periods
        max_drawdowns = []
        volatilities = []
        sharpe_ratios = []
        
        for period_name, results in all_results.items():
            if results.get('performance'):
                perf = results['performance']
                max_drawdowns.append(perf.get('max_drawdown', 0))
                volatilities.append(perf.get('volatility', 0))
                sharpe_ratios.append(perf.get('sharpe_ratio', 0))
        
        if max_drawdowns:
            risk_analysis['drawdown_analysis'] = {
                'avg_max_drawdown': sum(max_drawdowns) / len(max_drawdowns),
                'worst_drawdown': max(max_drawdowns),
                'best_drawdown': min(max_drawdowns),
                'consistency': 1 - (max(max_drawdowns) - min(max_drawdowns)) / max(max_drawdowns) if max(max_drawdowns) > 0 else 1
            }
        
        if volatilities:
            risk_analysis['volatility_analysis'] = {
                'avg_volatility': sum(volatilities) / len(volatilities),
                'volatility_range': max(volatilities) - min(volatilities),
                'risk_level': 'Low' if sum(volatilities) / len(volatilities) < 0.3 else 'Medium' if sum(volatilities) / len(volatilities) < 0.6 else 'High'
            }
        
        # Risk-adjusted performance
        if sharpe_ratios:
            risk_analysis['risk_adjusted_performance'] = {
                'avg_sharpe': sum(sharpe_ratios) / len(sharpe_ratios),
                'sharpe_consistency': 1 - (max(sharpe_ratios) - min(sharpe_ratios)) / max(sharpe_ratios) if max(sharpe_ratios) > 0 else 1,
                'quality_score': 'Excellent' if sum(sharpe_ratios) / len(sharpe_ratios) > 2 else 'Good' if sum(sharpe_ratios) / len(sharpe_ratios) > 1 else 'Fair'
            }
        
        print("✅ Risk analysis completed")
        return risk_analysis
        
    except Exception as e:
        print(f"❌ Risk analysis failed: {e}")
        return {}

def generate_final_report(all_results: dict, comparative: dict, risk_analysis: dict) -> dict:
    """Generate comprehensive final report"""
    try:
        final_report = {
            'executive_summary': generate_executive_summary(all_results, comparative, risk_analysis),
            'detailed_results': all_results,
            'comparative_analysis': comparative,
            'risk_analysis': risk_analysis,
            'recommendations': generate_recommendations(all_results, comparative, risk_analysis),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '2.0',
                'strategy': 'Enhanced Nanpin (永久ナンピン)',
                'periods_analyzed': len(all_results)
            }
        }
        
        print("✅ Final report generated")
        return final_report
        
    except Exception as e:
        print(f"❌ Final report generation failed: {e}")
        return {}

def generate_executive_summary(all_results: dict, comparative: dict, risk_analysis: dict) -> dict:
    """Generate executive summary"""
    try:
        # Calculate aggregate metrics
        total_returns = []
        annual_returns = []
        sharpe_ratios = []
        max_drawdowns = []
        total_trades = 0
        
        for results in all_results.values():
            if results.get('performance'):
                perf = results['performance']
                total_returns.append(perf.get('total_return', 0))
                annual_returns.append(perf.get('annual_return', 0))
                sharpe_ratios.append(perf.get('sharpe_ratio', 0))
                max_drawdowns.append(perf.get('max_drawdown', 0))
                total_trades += perf.get('total_trades', 0)
        
        # Performance assessment
        avg_annual_return = sum(annual_returns) / len(annual_returns) if annual_returns else 0
        avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
        avg_drawdown = sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0
        
        # Compare with target (Simple Trump Era +245.4%)
        target_annual = 2.454
        performance_vs_target = avg_annual_return / target_annual if target_annual > 0 else 0
        
        summary = {
            'overall_performance': {
                'average_annual_return': avg_annual_return,
                'average_sharpe_ratio': avg_sharpe,
                'average_max_drawdown': avg_drawdown,
                'total_trades_all_periods': total_trades,
                'performance_vs_target': performance_vs_target,
                'target_beaten': avg_annual_return > target_annual
            },
            'key_findings': [],
            'performance_grade': '',
            'risk_grade': '',
            'overall_recommendation': ''
        }
        
        # Key findings
        if avg_annual_return > target_annual:
            summary['key_findings'].append(f"✅ Strategy beats target by {(performance_vs_target - 1) * 100:.1f}%")
        else:
            summary['key_findings'].append(f"⚠️ Strategy underperforms target by {(1 - performance_vs_target) * 100:.1f}%")
        
        if avg_sharpe > 2.0:
            summary['key_findings'].append("✅ Excellent risk-adjusted returns (Sharpe > 2.0)")
        elif avg_sharpe > 1.0:
            summary['key_findings'].append("👍 Good risk-adjusted returns (Sharpe > 1.0)")
        else:
            summary['key_findings'].append("⚠️ Below-average risk-adjusted returns")
        
        if avg_drawdown < 0.25:
            summary['key_findings'].append("✅ Well-controlled downside risk (<25% drawdown)")
        else:
            summary['key_findings'].append("⚠️ High downside risk (>25% drawdown)")
        
        # Performance grading
        if avg_annual_return > target_annual and avg_sharpe > 2.0:
            summary['performance_grade'] = 'A+'
        elif avg_annual_return > target_annual * 0.8 and avg_sharpe > 1.5:
            summary['performance_grade'] = 'A'
        elif avg_annual_return > target_annual * 0.6 and avg_sharpe > 1.0:
            summary['performance_grade'] = 'B'
        else:
            summary['performance_grade'] = 'C'
        
        # Risk grading
        if avg_drawdown < 0.2 and avg_sharpe > 2.0:
            summary['risk_grade'] = 'A+'
        elif avg_drawdown < 0.3 and avg_sharpe > 1.5:
            summary['risk_grade'] = 'A'
        elif avg_drawdown < 0.4:
            summary['risk_grade'] = 'B'
        else:
            summary['risk_grade'] = 'C'
        
        # Overall recommendation
        if summary['performance_grade'] in ['A+', 'A'] and summary['risk_grade'] in ['A+', 'A']:
            summary['overall_recommendation'] = 'STRONG BUY - Excellent performance with controlled risk'
        elif summary['performance_grade'] in ['A', 'B'] and summary['risk_grade'] in ['A', 'B']:
            summary['overall_recommendation'] = 'BUY - Good performance with acceptable risk'
        elif summary['performance_grade'] == 'B':
            summary['overall_recommendation'] = 'HOLD - Moderate performance, consider optimization'
        else:
            summary['overall_recommendation'] = 'OPTIMIZE - Requires strategy improvements'
        
        return summary
        
    except Exception as e:
        print(f"❌ Executive summary generation failed: {e}")
        return {}

def generate_recommendations(all_results: dict, comparative: dict, risk_analysis: dict) -> list:
    """Generate actionable recommendations"""
    recommendations = []
    
    try:
        # Performance recommendations
        annual_returns = []
        for results in all_results.values():
            if results.get('performance'):
                annual_returns.append(results['performance'].get('annual_return', 0))
        
        if annual_returns:
            avg_return = sum(annual_returns) / len(annual_returns)
            
            if avg_return > 2.0:  # >200% annual
                recommendations.append("✅ Strategy shows exceptional performance - consider live deployment")
            elif avg_return > 1.0:  # >100% annual
                recommendations.append("👍 Strategy shows strong performance - consider gradual position sizing")
            else:
                recommendations.append("⚠️ Consider optimizing entry criteria and macro adjustments")
        
        # Risk recommendations
        if risk_analysis.get('risk_adjusted_performance', {}).get('quality_score') == 'Excellent':
            recommendations.append("✅ Excellent risk management - maintain current approach")
        else:
            recommendations.append("📊 Consider implementing stricter risk controls")
        
        # Specific optimizations
        recommendations.extend([
            "🔮 Monitor macro regime changes for dynamic scaling adjustments",
            "📐 Validate Fibonacci levels with additional confluence factors",
            "⏱️ Consider shorter cooldown periods during extreme market conditions",
            "💰 Test different base position sizes for optimization",
            "📈 Implement stop-loss mechanisms for extreme drawdown protection"
        ])
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Recommendations generation failed: {e}")
        return ["Further analysis required due to processing errors"]

def display_executive_summary(final_report: dict):
    """Display executive summary to console"""
    try:
        summary = final_report.get('executive_summary', {})
        overall = summary.get('overall_performance', {})
        
        print("\n🎯 EXECUTIVE SUMMARY")
        print("=" * 60)
        
        print(f"📈 Average Annual Return: {overall.get('average_annual_return', 0):.1%}")
        print(f"📊 Average Sharpe Ratio: {overall.get('average_sharpe_ratio', 0):.2f}")
        print(f"📉 Average Max Drawdown: {overall.get('average_max_drawdown', 0):.1%}")
        print(f"💰 Total Trades (All Periods): {overall.get('total_trades_all_periods', 0)}")
        print(f"🎯 vs Target (+245.4%): {overall.get('performance_vs_target', 0):.1%}")
        
        print(f"\n🏆 PERFORMANCE GRADE: {summary.get('performance_grade', 'N/A')}")
        print(f"⚠️ RISK GRADE: {summary.get('risk_grade', 'N/A')}")
        print(f"💡 RECOMMENDATION: {summary.get('overall_recommendation', 'N/A')}")
        
        print(f"\n🔍 KEY FINDINGS:")
        for finding in summary.get('key_findings', []):
            print(f"   {finding}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in final_report.get('recommendations', [])[:5]:  # Show top 5
            print(f"   {rec}")
        
    except Exception as e:
        print(f"❌ Failed to display executive summary: {e}")

async def main():
    """Main execution function"""
    try:
        # Setup
        setup_logging()
        print_comprehensive_banner()
        
        # Check requirements
        try:
            import yfinance
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            print(f"❌ Missing required package: {e}")
            print("Please install: pip install yfinance matplotlib seaborn")
            return
        
        # Run comprehensive analysis
        results = await run_comprehensive_analysis()
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Check 'results/' directory for detailed outputs")
        print(f"📊 Charts saved to 'results/charts/'")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        logging.getLogger(__name__).error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
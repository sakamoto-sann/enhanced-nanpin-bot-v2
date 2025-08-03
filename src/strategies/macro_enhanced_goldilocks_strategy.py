"""
🎯 Macro-Enhanced Goldilocks Nanpin Strategy
Integrates all sophisticated components for optimal trading decisions
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .goldilocks_nanpin_strategy import GoldilocksNanpinStrategy
from ..core.macro_analyzer import MacroAnalyzer, MacroAnalysis
from ..data.liquidation_aggregator_fixed import LiquidationAggregator
from ..data.flipside_client_fixed import FlipsideClient

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMarketAnalysis:
    """Enhanced market analysis with all data sources"""
    basic_analysis: Dict
    macro_analysis: Optional[MacroAnalysis]
    liquidation_intelligence: Dict
    flipside_metrics: Dict
    integrated_signals: Dict
    confidence_score: float
    timestamp: datetime

class MacroEnhancedGoldilocksStrategy(GoldilocksNanpinStrategy):
    """
    🎯 Macro-Enhanced Goldilocks Nanpin Strategy
    
    Integrates all sophisticated components:
    - Macro economic analysis (FRED + Polymarket)
    - Liquidation intelligence (multi-source)
    - On-chain metrics (Flipside)
    - Enhanced Fibonacci levels
    - Multi-API price validation
    """
    
    def __init__(self, config: Dict = None, macro_analyzer: MacroAnalyzer = None, 
                 liquidation_aggregator: LiquidationAggregator = None):
        """Initialize enhanced strategy"""
        super().__init__(config)
        
        self.macro_analyzer = macro_analyzer
        self.liquidation_aggregator = liquidation_aggregator
        self.flipside_client = FlipsideClient()
        
        # Enhanced configuration
        self.enhanced_config = self._get_enhanced_config()
        
        # Ensure update_frequencies exists in config
        if 'update_frequencies' not in self.enhanced_config:
            self.enhanced_config['update_frequencies'] = {
                'macro_analysis': 1800,      # 30 minutes
                'liquidation_intel': 300,    # 5 minutes  
                'flipside_metrics': 3600,    # 1 hour
                'price_validation': 60       # 1 minute
            }
        
        # State tracking
        self.last_macro_update = None
        self.last_liquidation_update = None
        self.last_flipside_update = None
        self.current_macro_analysis = None
        self.current_liquidation_heatmap = None
        self.current_flipside_metrics = None
        
        logger.info("🎯 Macro-Enhanced Goldilocks Strategy initialized")
        logger.info("   🔮 Macro analyzer integration: ✅")
        logger.info("   🔥 Liquidation intelligence: ✅") 
        logger.info("   🔗 Flipside on-chain data: ✅")
        logger.info("   📊 Multi-source validation: ✅")
    
    def _get_enhanced_config(self) -> Dict:
        """Get enhanced configuration with macro integration"""
        base_config = super()._get_default_config()
        
        enhanced_config = {
            **base_config,
            
            # Macro regime adjustments
            'macro_regime_multipliers': {
                'CRISIS': {'leverage': 2.5, 'position_size': 1.8, 'confidence': 0.9},
                'BEAR': {'leverage': 1.8, 'position_size': 1.4, 'confidence': 0.8},
                'NEUTRAL': {'leverage': 1.0, 'position_size': 1.0, 'confidence': 0.7},
                'BULL': {'leverage': 0.6, 'position_size': 0.7, 'confidence': 0.6}
            },
            
            # Liquidation intelligence weights
            'liquidation_weights': {
                'cluster_proximity': 0.3,    # How close to liquidation clusters
                'volume_significance': 0.25, # Size of liquidations
                'cascade_risk': 0.2,         # Risk of cascading liquidations
                'flipside_metrics': 0.15,    # On-chain validation
                'cross_validation': 0.1      # Multi-source confirmation
            },
            
            # Enhanced Fear & Greed inputs
            'fear_greed_inputs': {
                'macro_regime': 0.35,        # Macro economic regime
                'liquidation_stress': 0.25,  # Liquidation market stress
                'flipside_sentiment': 0.15,  # On-chain sentiment
                'polymarket_sentiment': 0.15, # Prediction market sentiment
                'basic_indicators': 0.1      # Traditional price-based
            },
            
            # Multi-source validation
            'validation_sources': {
                'min_sources': 3,            # Minimum data sources required
                'confidence_threshold': 0.10, # Minimum confidence for trades (TEST MODE)
                'macro_weight': 0.4,         # Weight of macro analysis
                'liquidation_weight': 0.35,  # Weight of liquidation intel
                'onchain_weight': 0.25       # Weight of on-chain data
            },
            
            # Update frequencies (seconds)
            'update_frequencies': {
                'macro_analysis': 1800,      # 30 minutes
                'liquidation_intel': 300,    # 5 minutes  
                'flipside_metrics': 3600,    # 1 hour
                'price_validation': 60       # 1 minute
            }
        }
        
        return enhanced_config
    
    async def initialize(self):
        """Initialize all enhanced components"""
        try:
            logger.info("🚀 Initializing enhanced strategy components...")
            
            # Initialize Flipside client
            if self.flipside_client:
                await self.flipside_client.initialize()
                logger.info("   ✅ Flipside client ready")
            
            # Perform initial data updates
            await self._update_all_analysis()
            
            logger.info("🎉 Enhanced strategy fully initialized!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing enhanced strategy: {e}")
            raise
    
    async def _update_all_analysis(self):
        """Update all analysis components"""
        try:
            now = datetime.now()
            
            # Update macro analysis
            if (not self.last_macro_update or 
                (now - self.last_macro_update).total_seconds() >= self.enhanced_config['update_frequencies']['macro_analysis']):
                
                if self.macro_analyzer:
                    logger.info("🔮 Updating macro analysis...")
                    self.current_macro_analysis = await self.macro_analyzer.update_macro_analysis()
                    self.last_macro_update = now
                    logger.info("   ✅ Macro analysis updated")
            
            # Update liquidation intelligence
            if (not self.last_liquidation_update or 
                (now - self.last_liquidation_update).total_seconds() >= self.enhanced_config['update_frequencies']['liquidation_intel']):
                
                if self.liquidation_aggregator:
                    logger.info("🔥 Updating liquidation intelligence...")
                    self.current_liquidation_heatmap = await self.liquidation_aggregator.generate_liquidation_heatmap('BTC')
                    self.last_liquidation_update = now
                    logger.info("   ✅ Liquidation intelligence updated")
            
            # Update Flipside metrics
            if (not self.last_flipside_update or 
                (now - self.last_flipside_update).total_seconds() >= self.enhanced_config['update_frequencies']['flipside_metrics']):
                
                logger.info("🔗 Updating Flipside metrics...")
                self.current_flipside_metrics = await self.flipside_client.get_liquidation_metrics('BTC')
                self.last_flipside_update = now
                logger.info("   ✅ Flipside metrics updated")
                
        except Exception as e:
            logger.error(f"❌ Error updating analysis: {e}")
    
    async def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Enhanced market analysis with all data sources"""
        try:
            # Update all analysis first
            await self._update_all_analysis()
            
            # Get basic analysis from parent class
            basic_analysis = await super().analyze_market_conditions(market_data)
            
            if 'error' in basic_analysis:
                return basic_analysis
            
            # Create enhanced analysis
            enhanced_analysis = EnhancedMarketAnalysis(
                basic_analysis=basic_analysis,
                macro_analysis=self.current_macro_analysis,
                liquidation_intelligence=self._process_liquidation_intelligence(),
                flipside_metrics=self._process_flipside_metrics(),
                integrated_signals={},
                confidence_score=0.0,
                timestamp=datetime.now()
            )
            
            # Generate integrated signals
            enhanced_analysis.integrated_signals = self._generate_integrated_signals(enhanced_analysis)
            
            # Calculate overall confidence
            enhanced_analysis.confidence_score = self._calculate_confidence_score(enhanced_analysis)
            
            # Generate enhanced recommendations
            enhanced_recommendations = self._generate_enhanced_recommendations(enhanced_analysis)
            
            # Return enhanced analysis
            return {
                'timestamp': enhanced_analysis.timestamp,
                'current_price': market_data.get('current_price'),
                'basic_analysis': basic_analysis,
                'macro_analysis': self._serialize_macro_analysis(enhanced_analysis.macro_analysis),
                'liquidation_intelligence': enhanced_analysis.liquidation_intelligence,
                'flipside_metrics': enhanced_analysis.flipside_metrics,
                'integrated_signals': enhanced_analysis.integrated_signals,
                'confidence_score': enhanced_analysis.confidence_score,
                'recommendations': enhanced_recommendations,
                'market_regime': self._determine_enhanced_market_regime(enhanced_analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced market analysis: {e}")
            # Fallback to basic analysis
            return await super().analyze_market_conditions(market_data)
    
    def _process_liquidation_intelligence(self) -> Dict:
        """Process liquidation intelligence into actionable signals"""
        try:
            if not self.current_liquidation_heatmap:
                return {'available': False, 'reason': 'No liquidation data'}
            
            heatmap = self.current_liquidation_heatmap
            
            # Process clusters
            cluster_analysis = {
                'total_clusters': len(heatmap.clusters),
                'high_risk_clusters': len([c for c in heatmap.clusters if hasattr(c, 'confidence') and c.confidence > 0.7]),
                'total_volume': sum(getattr(c, 'volume_usdc', 0) for c in heatmap.clusters),
                'price_levels': [getattr(c, 'price_level', 0) for c in heatmap.clusters[:10]],  # Top 10 levels
                'cascade_risk': getattr(heatmap, 'cascade_risk_score', 5.0),
                'market_stress': getattr(heatmap, 'overall_sentiment', 'NEUTRAL')
            }
            
            return {
                'available': True,
                'cluster_analysis': cluster_analysis,
                'signals': {
                    'liquidation_pressure': 'HIGH' if cluster_analysis['cascade_risk'] > 7.0 else 'MEDIUM' if cluster_analysis['cascade_risk'] > 4.0 else 'LOW',
                    'entry_opportunity': cluster_analysis['cascade_risk'] > 6.0,  # High liquidation = opportunity
                    'position_scaling': min(cluster_analysis['cascade_risk'] / 10.0, 1.0)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing liquidation intelligence: {e}")
            return {'available': False, 'error': str(e)}
    
    def _process_flipside_metrics(self) -> Dict:
        """Process Flipside on-chain metrics"""
        try:
            if not self.current_flipside_metrics:
                return {'available': False, 'reason': 'No Flipside data'}
            
            metrics = self.current_flipside_metrics
            
            # Calculate signals from metrics
            whale_pressure = 'HIGH' if metrics.whale_activity_score > 7.0 else 'MEDIUM' if metrics.whale_activity_score > 4.0 else 'LOW'
            liquidation_risk = 'HIGH' if metrics.liquidation_cascade_risk > 6.0 else 'MEDIUM' if metrics.liquidation_cascade_risk > 3.0 else 'LOW'
            market_stress = 'HIGH' if metrics.market_stress_indicator > 70 else 'MEDIUM' if metrics.market_stress_indicator > 40 else 'LOW'
            
            # Calculate net exchange flows
            exchange_flows = metrics.exchange_flows
            net_flow = exchange_flows.get('outflow', 0) - exchange_flows.get('inflow', 0)
            flow_sentiment = 'BULLISH' if net_flow > 0 else 'BEARISH' if net_flow < -10_000_000 else 'NEUTRAL'
            
            return {
                'available': True,
                'raw_metrics': {
                    'liquidation_volume_24h': metrics.liquidation_volume_24h,
                    'whale_activity_score': metrics.whale_activity_score,
                    'liquidation_cascade_risk': metrics.liquidation_cascade_risk,
                    'market_stress_indicator': metrics.market_stress_indicator,
                    'net_exchange_flow': net_flow
                },
                'signals': {
                    'whale_pressure': whale_pressure,
                    'liquidation_risk': liquidation_risk,
                    'market_stress': market_stress,
                    'flow_sentiment': flow_sentiment,
                    'entry_multiplier': self._calculate_flipside_entry_multiplier(metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing Flipside metrics: {e}")
            return {'available': False, 'error': str(e)}
    
    def _calculate_flipside_entry_multiplier(self, metrics) -> float:
        """Calculate entry position multiplier from Flipside metrics"""
        try:
            # Base multiplier
            multiplier = 1.0
            
            # Whale activity (higher activity = more opportunity)
            if metrics.whale_activity_score > 7.0:
                multiplier += 0.3
            elif metrics.whale_activity_score > 4.0:
                multiplier += 0.15
            
            # Liquidation cascade risk (higher risk = more opportunity)
            if metrics.liquidation_cascade_risk > 6.0:
                multiplier += 0.4
            elif metrics.liquidation_cascade_risk > 3.0:
                multiplier += 0.2
            
            # Market stress (higher stress = more opportunity)
            stress_factor = metrics.market_stress_indicator / 100.0
            multiplier += stress_factor * 0.5
            
            # Exchange flows (outflows bullish for price)
            net_flow = metrics.exchange_flows.get('outflow', 0) - metrics.exchange_flows.get('inflow', 0)
            if net_flow > 20_000_000:  # Large outflow
                multiplier += 0.2
            elif net_flow < -20_000_000:  # Large inflow
                multiplier -= 0.2
            
            return max(0.5, min(multiplier, 2.5))  # Clamp between 0.5x and 2.5x
            
        except Exception as e:
            logger.error(f"❌ Error calculating Flipside multiplier: {e}")
            return 1.0
    
    def _generate_integrated_signals(self, analysis: EnhancedMarketAnalysis) -> Dict:
        """Generate integrated signals from all data sources"""
        try:
            signals = {
                'macro_signal': 'NEUTRAL',
                'liquidation_signal': 'NEUTRAL', 
                'onchain_signal': 'NEUTRAL',
                'combined_signal': 'NEUTRAL',
                'entry_strength': 0.5,
                'position_multiplier': 1.0,
                'confidence_factors': {}
            }
            
            # Macro signal
            if analysis.macro_analysis:
                macro = analysis.macro_analysis
                if macro.overall_signal == 'STRONG_BUY':
                    signals['macro_signal'] = 'STRONG_BUY'
                    signals['entry_strength'] += 0.3
                elif macro.overall_signal == 'BUY':
                    signals['macro_signal'] = 'BUY'
                    signals['entry_strength'] += 0.2
                elif macro.overall_signal == 'SELL':
                    signals['macro_signal'] = 'SELL'
                    signals['entry_strength'] -= 0.2
                
                # Apply macro regime multiplier
                regime_multiplier = self.enhanced_config['macro_regime_multipliers'].get(
                    macro.regime.value, {'leverage': 1.0, 'position_size': 1.0}
                )
                signals['position_multiplier'] *= regime_multiplier['position_size']
                
                signals['confidence_factors']['macro'] = macro.regime_confidence
            
            # Liquidation signal
            if analysis.liquidation_intelligence.get('available'):
                liq_signals = analysis.liquidation_intelligence['signals']
                if liq_signals['entry_opportunity']:
                    signals['liquidation_signal'] = 'BUY'
                    signals['entry_strength'] += 0.25
                    signals['position_multiplier'] *= (1 + liq_signals['position_scaling'])
                
                signals['confidence_factors']['liquidation'] = liq_signals['position_scaling']
            
            # On-chain signal  
            if analysis.flipside_metrics.get('available'):
                flipside_signals = analysis.flipside_metrics['signals']
                if flipside_signals['liquidation_risk'] == 'HIGH':
                    signals['onchain_signal'] = 'BUY'
                    signals['entry_strength'] += 0.2
                    signals['position_multiplier'] *= flipside_signals['entry_multiplier']
                
                signals['confidence_factors']['onchain'] = min(
                    analysis.flipside_metrics['raw_metrics']['liquidation_cascade_risk'] / 10.0, 1.0
                )
            
            # Combined signal
            if signals['entry_strength'] > 0.8:
                signals['combined_signal'] = 'STRONG_BUY'
            elif signals['entry_strength'] > 0.6:
                signals['combined_signal'] = 'BUY'
            elif signals['entry_strength'] < 0.3:
                signals['combined_signal'] = 'SELL'
            else:
                signals['combined_signal'] = 'NEUTRAL'
            
            # Clamp position multiplier
            signals['position_multiplier'] = max(0.5, min(signals['position_multiplier'], 3.0))
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generating integrated signals: {e}")
            return {'combined_signal': 'NEUTRAL', 'entry_strength': 0.5, 'position_multiplier': 1.0}
    
    def _calculate_confidence_score(self, analysis: EnhancedMarketAnalysis) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_components = []
            
            # Basic analysis confidence
            basic_conf = analysis.basic_analysis.get('entry_assessment', {}).get('conditions_met', False)
            confidence_components.append(0.8 if basic_conf else 0.3)
            
            # Macro analysis confidence
            if analysis.macro_analysis:
                confidence_components.append(analysis.macro_analysis.regime_confidence)
            
            # Liquidation intelligence confidence
            if analysis.liquidation_intelligence.get('available'):
                liq_conf = min(analysis.liquidation_intelligence['cluster_analysis']['cascade_risk'] / 10.0, 1.0)
                confidence_components.append(liq_conf)
            
            # Flipside metrics confidence
            if analysis.flipside_metrics.get('available'):
                flipside_conf = min(analysis.flipside_metrics['raw_metrics']['market_stress_indicator'] / 100.0, 1.0)
                confidence_components.append(flipside_conf)
            
            # Calculate weighted average
            if confidence_components:
                return sum(confidence_components) / len(confidence_components)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_enhanced_recommendations(self, analysis: EnhancedMarketAnalysis) -> List[Dict]:
        """Generate enhanced trading recommendations"""
        try:
            # Get basic recommendations
            basic_recommendations = analysis.basic_analysis.get('recommendations', [])
            
            if not basic_recommendations or analysis.integrated_signals['combined_signal'] == 'SELL':
                return [{
                    'action': 'WAIT',
                    'reason': 'Enhanced analysis suggests waiting',
                    'confidence': analysis.confidence_score,
                    'integrated_signals': analysis.integrated_signals
                }]
            
            enhanced_recommendations = []
            
            for basic_rec in basic_recommendations:
                if basic_rec.get('action') == 'BUY':
                    # Enhance the recommendation
                    enhanced_rec = basic_rec.copy()
                    
                    # Apply position multiplier
                    original_size = enhanced_rec.get('position_size_usdc', 0)
                    enhanced_size = original_size * analysis.integrated_signals['position_multiplier']
                    enhanced_rec['position_size_usdc'] = enhanced_size
                    
                    # Apply macro regime leverage adjustment
                    if analysis.macro_analysis:
                        regime_multiplier = self.enhanced_config['macro_regime_multipliers'].get(
                            analysis.macro_analysis.regime.value, {'leverage': 1.0}
                        )
                        enhanced_rec['leverage'] = enhanced_rec.get('leverage', 1.0) * regime_multiplier['leverage']
                    
                    # Update confidence with integrated score
                    enhanced_rec['confidence'] = analysis.confidence_score
                    
                    # Add enhanced reasoning
                    enhanced_rec['enhanced_reasoning'] = self._create_enhanced_reasoning(analysis)
                    
                    # Add risk assessment
                    enhanced_rec['risk_assessment'] = self._assess_enhanced_risk(analysis)
                    
                    enhanced_recommendations.append(enhanced_rec)
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced recommendations: {e}")
            return analysis.basic_analysis.get('recommendations', [])
    
    def _create_enhanced_reasoning(self, analysis: EnhancedMarketAnalysis) -> str:
        """Create enhanced reasoning for trade recommendations"""
        try:
            reasons = []
            
            # Macro reasoning
            if analysis.macro_analysis:
                regime = analysis.macro_analysis.regime.value
                confidence = analysis.macro_analysis.regime_confidence
                reasons.append(f"Macro regime: {regime} (confidence: {confidence:.1%})")
            
            # Liquidation reasoning
            if analysis.liquidation_intelligence.get('available'):
                cascade_risk = analysis.liquidation_intelligence['cluster_analysis']['cascade_risk']
                reasons.append(f"Liquidation cascade risk: {cascade_risk:.1f}/10 (opportunity)")
            
            # On-chain reasoning
            if analysis.flipside_metrics.get('available'):
                stress = analysis.flipside_metrics['raw_metrics']['market_stress_indicator']
                reasons.append(f"Market stress indicator: {stress:.1f}% (elevated)")
            
            # Combined signal
            combined = analysis.integrated_signals['combined_signal']
            strength = analysis.integrated_signals['entry_strength']
            reasons.append(f"Integrated signal: {combined} (strength: {strength:.2f})")
            
            return " | ".join(reasons)
            
        except Exception as e:
            logger.error(f"❌ Error creating enhanced reasoning: {e}")
            return "Enhanced analysis with multi-source validation"
    
    def _assess_enhanced_risk(self, analysis: EnhancedMarketAnalysis) -> Dict:
        """Assess risk with enhanced data"""
        try:
            risk_factors = []
            overall_risk = 'MEDIUM'
            
            # Macro risk
            if analysis.macro_analysis:
                if analysis.macro_analysis.regime.value in ['CRISIS', 'BEAR']:
                    risk_factors.append('High macro economic stress')
                elif analysis.macro_analysis.regime.value == 'BULL':
                    risk_factors.append('Potential market overheating')
            
            # Liquidation risk
            if analysis.liquidation_intelligence.get('available'):
                cascade_risk = analysis.liquidation_intelligence['cluster_analysis']['cascade_risk']
                if cascade_risk > 7.0:
                    risk_factors.append('High liquidation cascade risk')
                    overall_risk = 'HIGH'
                elif cascade_risk > 4.0:
                    risk_factors.append('Moderate liquidation pressure')
            
            # On-chain risk
            if analysis.flipside_metrics.get('available'):
                whale_activity = analysis.flipside_metrics['raw_metrics']['whale_activity_score']
                if whale_activity > 8.0:
                    risk_factors.append('Extreme whale activity')
                    overall_risk = 'HIGH'
            
            # Confidence risk
            if analysis.confidence_score < 0.6:
                risk_factors.append('Low signal confidence')
                overall_risk = 'HIGH'
            elif analysis.confidence_score > 0.8 and overall_risk != 'HIGH':
                overall_risk = 'LOW'
            
            return {
                'overall_risk': overall_risk,
                'risk_factors': risk_factors,
                'confidence_score': analysis.confidence_score,
                'position_multiplier': analysis.integrated_signals['position_multiplier']
            }
            
        except Exception as e:
            logger.error(f"❌ Error assessing enhanced risk: {e}")
            return {'overall_risk': 'MEDIUM', 'risk_factors': ['Analysis error'], 'confidence_score': 0.5}
    
    def _determine_enhanced_market_regime(self, analysis: EnhancedMarketAnalysis) -> str:
        """Determine market regime with enhanced data"""
        try:
            # Start with macro regime if available
            if analysis.macro_analysis:
                base_regime = analysis.macro_analysis.regime.value
            else:
                # Fallback to basic regime
                basic_indicators = analysis.basic_analysis.get('indicators', {})
                drawdown = basic_indicators.get('drawdown', 0)
                fear_greed = basic_indicators.get('fear_greed', 50)
                
                if fear_greed < 20 and drawdown < -25:
                    base_regime = 'CRISIS'
                elif fear_greed < 40 and drawdown < -15:
                    base_regime = 'BEAR'
                elif fear_greed > 70 and drawdown > -5:
                    base_regime = 'BULL'
                else:
                    base_regime = 'NEUTRAL'
            
            # Adjust with liquidation and on-chain data
            adjustments = []
            
            if analysis.liquidation_intelligence.get('available'):
                cascade_risk = analysis.liquidation_intelligence['cluster_analysis']['cascade_risk']
                if cascade_risk > 7.0:
                    adjustments.append('STRESSED')
                elif cascade_risk > 4.0:
                    adjustments.append('VOLATILE')
            
            if analysis.flipside_metrics.get('available'):
                stress = analysis.flipside_metrics['raw_metrics']['market_stress_indicator']
                if stress > 80:
                    adjustments.append('EXTREME')
                elif stress > 60:
                    adjustments.append('ELEVATED')
            
            # Combine regime with adjustments
            if adjustments:
                return f"{base_regime}_{'_'.join(adjustments)}"
            else:
                return base_regime
                
        except Exception as e:
            logger.error(f"❌ Error determining enhanced market regime: {e}")
            return 'UNKNOWN'
    
    def _serialize_macro_analysis(self, macro_analysis: Optional[MacroAnalysis]) -> Dict:
        """Serialize macro analysis for JSON response"""
        if not macro_analysis:
            return {'available': False}
        
        try:
            return {
                'available': True,
                'regime': macro_analysis.regime.value,
                'regime_confidence': macro_analysis.regime_confidence,
                'overall_signal': macro_analysis.overall_signal,
                'fear_greed_index': macro_analysis.fear_greed_index,
                'bitcoin_sentiment': macro_analysis.bitcoin_sentiment,
                'position_scaling_factor': macro_analysis.position_scaling_factor,
                'risk_adjustment': macro_analysis.risk_adjustment,
                'analysis_timestamp': macro_analysis.analysis_timestamp.isoformat(),
                'indicator_count': len(macro_analysis.indicators)
            }
        except Exception as e:
            logger.error(f"❌ Error serializing macro analysis: {e}")
            return {'available': False, 'error': str(e)}
    
    def update_position_parameters(self, recommendation):
        """Update strategy position parameters from dynamic position sizer"""
        try:
            logger.info("🧮 Updating strategy position parameters from dynamic sizer...")
            
            # Ensure config sections exist
            if 'trading' not in self.config:
                self.config['trading'] = {}
            if 'risk_management' not in self.config:
                self.config['risk_management'] = {}
            
            # Update base configuration with dynamic values using correct config paths
            self.config['trading']['base_position_size'] = recommendation.base_margin
            self.config['trading']['nanpin_scaling_factor'] = recommendation.scaling_multiplier
            self.config['trading']['max_nanpin_levels'] = recommendation.max_levels
            
            # Override leverage if dynamic leverage is available
            if hasattr(recommendation, 'leverage'):
                self.config['trading']['leverage'] = recommendation.leverage
            
            # Update risk parameters
            self.config['trading']['max_position_size'] = recommendation.base_margin / 1000  # As fraction
            
            # Also update inherited goldilocks config
            if hasattr(self, 'base_investment'):
                self.base_investment = recommendation.base_margin
            if hasattr(self, 'scaling_multiplier'):
                self.scaling_multiplier = recommendation.scaling_multiplier
            if hasattr(self, 'max_levels'):
                self.max_levels = recommendation.max_levels
            
            logger.info(f"✅ Strategy updated with dynamic parameters:")
            logger.info(f"   Base Investment: ${recommendation.base_margin:.2f}")
            logger.info(f"   Scaling Multiplier: {recommendation.scaling_multiplier:.2f}")
            logger.info(f"   Max Levels: {recommendation.max_levels}")
            logger.info(f"   Dynamic Leverage: {recommendation.leverage}x")
            
        except Exception as e:
            logger.error(f"❌ Failed to update strategy position parameters: {e}")
    
    def _calculate_position_size(self, leverage: float, remaining_capital: float = None) -> float:
        """Override position size calculation to use dynamic values"""
        try:
            # Use dynamic position sizer values if available
            if hasattr(self, 'base_investment') and self.base_investment:
                base_margin = self.base_investment
                logger.debug(f"🧮 Using dynamic base margin: ${base_margin:.2f}")
                return base_margin
            
            # Fallback to config values if dynamic values not set
            base_position_size = self.config.get('trading', {}).get('base_position_size', 100)
            logger.debug(f"🧮 Using config base position: ${base_position_size:.2f}")
            return base_position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating dynamic position size: {e}")
            # Ultimate fallback to small safe value
            return 50.0
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.flipside_client:
                await self.flipside_client.close()
            logger.info("✅ Enhanced strategy resources cleaned up")
        except Exception as e:
            logger.error(f"❌ Error closing enhanced strategy: {e}")
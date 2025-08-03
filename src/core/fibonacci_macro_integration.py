#!/usr/bin/env python3
"""
🌸 Fibonacci Engine Macro Integration Methods
Extends FibonacciEngine with macro-aware position scaling
"""

def add_macro_methods_to_fibonacci_engine():
    """
    Methods to be added to FibonacciEngine class for macro integration
    """
    
    def _apply_macro_adjustments(self, base_multiplier: float, level_name: str, 
                               current_price: float, target_price: float) -> float:
        """
        Apply macro-informed adjustments to position scaling
        
        Args:
            base_multiplier: Base Fibonacci multiplier
            level_name: Fibonacci level name (e.g., '61.8%')
            current_price: Current BTC price
            target_price: Fibonacci level target price
            
        Returns:
            Macro-adjusted multiplier
        """
        try:
            if not self.macro_analyzer or not self.macro_analyzer.current_analysis:
                return base_multiplier
            
            analysis = self.macro_analyzer.current_analysis
            
            # Get macro scaling factors
            macro_scaling = analysis.position_scaling_factor
            risk_adjustment = analysis.risk_adjustment
            regime = analysis.regime
            fear_greed = analysis.fear_greed_index
            
            # Apply base macro scaling
            adjusted_multiplier = base_multiplier * macro_scaling * risk_adjustment
            
            # Apply regime-specific adjustments
            regime_adjustments = {
                'crisis': 1.5,      # Crisis = opportunity
                'recession': 1.3,   # Recession risk = opportunity
                'recovery': 1.1,    # Recovery = modest increase
                'expansion': 1.0,   # Normal expansion
                'stagflation': 1.2, # Inflation hedge
                'bubble': 0.6       # Bubble = extreme caution
            }
            
            regime_multiplier = regime_adjustments.get(regime.value, 1.0)
            adjusted_multiplier *= regime_multiplier
            
            # Apply fear/greed adjustments
            if fear_greed < 20:  # Extreme fear
                adjusted_multiplier *= 1.8
            elif fear_greed < 40:  # Fear
                adjusted_multiplier *= 1.4
            elif fear_greed > 80:  # Extreme greed
                adjusted_multiplier *= 0.5
            elif fear_greed > 70:  # Greed
                adjusted_multiplier *= 0.7
            
            # Golden ratio (61.8%) special treatment during macro stress
            if level_name == '61.8%' and fear_greed < 30:
                adjusted_multiplier *= 1.3  # Extra boost for golden ratio during fear
            
            # Deep levels (78.6%) get massive boosts during crisis
            if level_name == '78.6%' and (regime.value in ['crisis', 'recession'] or fear_greed < 25):
                adjusted_multiplier *= 2.0  # Maximum opportunity
            
            # Apply limits
            adjusted_multiplier = min(adjusted_multiplier, 20.0)  # Max 20x during extreme conditions
            adjusted_multiplier = max(adjusted_multiplier, 0.1)   # Min 0.1x
            
            logger.debug(f"🔮 Macro adjustments for {level_name}: "
                        f"{base_multiplier:.1f}x → {adjusted_multiplier:.1f}x "
                        f"(regime: {regime.value}, F&G: {fear_greed:.0f})")
            
            return adjusted_multiplier
            
        except Exception as e:
            logger.warning(f"Failed to apply macro adjustments: {e}")
            return base_multiplier
    
    def get_macro_enhanced_recommendations(self, current_price: float) -> Dict[str, Dict]:
        """
        Get position scaling recommendations with enhanced macro intelligence
        
        Args:
            current_price: Current BTC price
            
        Returns:
            Dictionary with macro-enhanced scaling recommendations
        """
        try:
            if not self.current_levels:
                logger.warning("⚠️ No Fibonacci levels calculated")
                return {}
            
            recommendations = {}
            
            # Get macro context
            macro_context = self._get_macro_context()
            
            for level_name, fib_level in self.current_levels.items():
                target_price = fib_level.price
                distance = (current_price - target_price) / target_price
                
                # Base Fibonacci multipliers
                base_multipliers = {
                    '23.6%': 1, '38.2%': 2, '50.0%': 3, '61.8%': 5, '78.6%': 8
                }
                base_multiplier = base_multipliers.get(level_name, 1)
                
                # Technical confluence adjustment
                confluence_boost = fib_level.confluence_score / 10.0
                technical_multiplier = base_multiplier * (1 + confluence_boost)
                
                # Apply macro adjustments
                macro_multiplier = self._apply_macro_adjustments(
                    technical_multiplier, level_name, current_price, target_price
                )
                
                # Determine action and urgency with macro context
                action, urgency = self._determine_macro_action(distance, macro_context)
                
                # Enhanced reasoning with macro intelligence
                reasoning = self._generate_macro_reasoning(
                    fib_level, distance, action, macro_context
                )
                
                recommendations[level_name] = {
                    'target_price': target_price,
                    'current_distance_pct': distance * 100,
                    'action': action,
                    'urgency': urgency,
                    'base_multiplier': base_multiplier,
                    'technical_multiplier': technical_multiplier,
                    'macro_multiplier': macro_multiplier,
                    'confidence': fib_level.confidence,
                    'confluence_score': fib_level.confluence_score,
                    'volume_confirmation': fib_level.volume_confirmation,
                    'macro_context': macro_context,
                    'reasoning': reasoning
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get macro-enhanced recommendations: {e}")
            return {}
    
    def _get_macro_context(self) -> Dict:
        """Get current macro context summary"""
        try:
            if not self.macro_analyzer or not self.macro_analyzer.current_analysis:
                return {'macro_available': False}
            
            analysis = self.macro_analyzer.current_analysis
            
            return {
                'macro_available': True,
                'regime': analysis.regime.value,
                'regime_confidence': analysis.regime_confidence,
                'overall_signal': analysis.overall_signal,
                'fear_greed_index': analysis.fear_greed_index,
                'bitcoin_sentiment': analysis.bitcoin_sentiment,
                'position_scaling_factor': analysis.position_scaling_factor,
                'risk_adjustment': analysis.risk_adjustment,
                'timestamp': analysis.analysis_timestamp.isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Failed to get macro context: {e}")
            return {'macro_available': False}
    
    def _determine_macro_action(self, distance: float, macro_context: Dict) -> Tuple[str, str]:
        """Determine action and urgency with macro context"""
        try:
            # Base action from price distance
            if distance <= -0.005:
                base_action = "BUY"
                base_urgency = "HIGH" if distance <= -0.02 else "MEDIUM"
            elif distance <= 0.005:
                base_action = "PREPARE"
                base_urgency = "MEDIUM"
            else:
                base_action = "WAIT"
                base_urgency = "LOW"
            
            if not macro_context.get('macro_available'):
                return base_action, base_urgency
            
            # Enhance with macro intelligence
            fear_greed = macro_context.get('fear_greed_index', 50)
            regime = macro_context.get('regime', 'expansion')
            overall_signal = macro_context.get('overall_signal', 'neutral')
            
            # Upgrade urgency during macro opportunities
            if base_action == "BUY":
                if fear_greed < 20 or regime in ['crisis', 'recession']:
                    base_urgency = "EXTREME"  # New urgency level for extreme macro opportunities
                elif fear_greed < 35 and overall_signal == 'bullish':
                    base_urgency = "HIGH"
            
            # Downgrade during macro caution periods
            elif regime == 'bubble' or fear_greed > 85:
                if base_action == "BUY":
                    base_action = "PREPARE"
                    base_urgency = "LOW"
                elif base_action == "PREPARE":
                    base_action = "WAIT"
                    base_urgency = "LOW"
            
            return base_action, base_urgency
            
        except Exception as e:
            logger.warning(f"Failed to determine macro action: {e}")
            return "WAIT", "LOW"
    
    def _generate_macro_reasoning(self, fib_level: FibonacciRetracement, distance: float, 
                                action: str, macro_context: Dict) -> str:
        """Generate enhanced reasoning with macro context"""
        try:
            reasoning_parts = []
            
            # Technical analysis
            reasoning_parts.append(f"{fib_level.percentage} Fibonacci level ({fib_level.strength} strength)")
            
            # Price action
            if distance <= -0.02:
                reasoning_parts.append("price significantly below level")
            elif distance <= -0.005:
                reasoning_parts.append("price below level")
            elif abs(distance) <= 0.005:
                reasoning_parts.append("price at level")
            else:
                reasoning_parts.append("price above level")
            
            # Technical confluence
            if fib_level.confluence_score > 6:
                reasoning_parts.append("high technical confluence")
            elif fib_level.confluence_score > 4:
                reasoning_parts.append("moderate confluence")
            
            # Volume confirmation
            if fib_level.volume_confirmation:
                reasoning_parts.append("volume confirmed")
            
            # Macro context
            if macro_context.get('macro_available'):
                regime = macro_context.get('regime')
                fear_greed = macro_context.get('fear_greed_index', 50)
                
                if regime in ['crisis', 'recession']:
                    reasoning_parts.append(f"{regime} regime (high opportunity)")
                elif regime == 'bubble':
                    reasoning_parts.append("bubble regime (extreme caution)")
                
                if fear_greed < 20:
                    reasoning_parts.append("extreme fear (maximum opportunity)")
                elif fear_greed < 35:
                    reasoning_parts.append("fear conditions (opportunity)")
                elif fear_greed > 80:
                    reasoning_parts.append("extreme greed (high caution)")
                elif fear_greed > 70:
                    reasoning_parts.append("greed conditions (caution)")
                
                overall_signal = macro_context.get('overall_signal')
                if overall_signal == 'bullish':
                    reasoning_parts.append("macro bullish")
                elif overall_signal == 'bearish':
                    reasoning_parts.append("macro bearish")
            
            return ", ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"Failed to generate macro reasoning: {e}")
            return "Standard analysis"
    
    def get_extreme_fibonacci_levels(self, reference_high: float, reference_low: float) -> Dict[str, float]:
        """
        Calculate extreme Fibonacci levels for crisis conditions
        
        Args:
            reference_high: Swing high price
            reference_low: Swing low price
            
        Returns:
            Dictionary of extreme Fibonacci levels
        """
        try:
            if not self.macro_analyzer or not self.macro_analyzer.current_analysis:
                return {}
            
            analysis = self.macro_analyzer.current_analysis
            
            # Only calculate extreme levels during crisis/recession
            if analysis.regime.value not in ['crisis', 'recession'] and analysis.fear_greed_index > 30:
                return {}
            
            price_range = reference_high - reference_low
            extreme_levels = {}
            
            # 88.6% Fibonacci level (extreme retracement)
            extreme_levels['88.6%'] = reference_high - (price_range * 0.886)
            
            # 100% retracement (full cycle)
            extreme_levels['100%'] = reference_low
            
            # 127.2% extension (beyond full retracement)
            extreme_levels['127.2%'] = reference_high - (price_range * 1.272)
            
            logger.info(f"🚨 Extreme Fibonacci levels calculated for {analysis.regime.value} conditions:")
            for level, price in extreme_levels.items():
                logger.info(f"   {level}: ${price:,.2f}")
            
            return extreme_levels
            
        except Exception as e:
            logger.warning(f"Failed to calculate extreme levels: {e}")
            return {}
    
    def calculate_macro_opportunity_score(self, level_name: str, current_price: float) -> float:
        """
        Calculate macro opportunity score for a Fibonacci level
        
        Args:
            level_name: Fibonacci level name
            current_price: Current BTC price
            
        Returns:
            Opportunity score (0-100)
        """
        try:
            if not self.macro_analyzer or not self.macro_analyzer.current_analysis:
                return 50.0  # Neutral
            
            analysis = self.macro_analyzer.current_analysis
            score = 50.0  # Start neutral
            
            # Regime scoring
            regime_scores = {
                'crisis': 90,      # Maximum opportunity
                'recession': 80,   # High opportunity  
                'recovery': 60,    # Moderate opportunity
                'expansion': 50,   # Neutral
                'stagflation': 55, # Slight opportunity (inflation hedge)
                'bubble': 20       # High caution
            }
            
            regime_score = regime_scores.get(analysis.regime.value, 50)
            score = (score + regime_score) / 2
            
            # Fear/Greed scoring
            if analysis.fear_greed_index < 20:  # Extreme fear
                score += 25
            elif analysis.fear_greed_index < 35:  # Fear
                score += 15
            elif analysis.fear_greed_index > 80:  # Extreme greed
                score -= 25
            elif analysis.fear_greed_index > 70:  # Greed
                score -= 15
            
            # Level-specific adjustments
            level_multipliers = {
                '23.6%': 0.8,  # Shallow level, less opportunity
                '38.2%': 0.9,  # Moderate level
                '50.0%': 1.0,  # Psychological level
                '61.8%': 1.2,  # Golden ratio, high opportunity
                '78.6%': 1.4   # Deep level, maximum opportunity
            }
            
            multiplier = level_multipliers.get(level_name, 1.0)
            score *= multiplier
            
            # Bitcoin sentiment adjustment
            if analysis.bitcoin_sentiment > 70:
                score += 10
            elif analysis.bitcoin_sentiment < 30:
                score -= 5
            
            # Bounds
            score = max(0, min(100, score))
            
            return score
            
        except Exception as e:
            logger.warning(f"Failed to calculate opportunity score: {e}")
            return 50.0
    
    def get_macro_alert_conditions(self) -> Dict[str, bool]:
        """
        Check for macro alert conditions
        
        Returns:
            Dictionary of alert conditions
        """
        try:
            alerts = {
                'extreme_fear_opportunity': False,
                'crisis_regime_detected': False,
                'maximum_accumulation_signal': False,
                'bubble_warning': False,
                'regime_change_detected': False
            }
            
            if not self.macro_analyzer or not self.macro_analyzer.current_analysis:
                return alerts
            
            analysis = self.macro_analyzer.current_analysis
            
            # Extreme fear opportunity
            if analysis.fear_greed_index < 20:
                alerts['extreme_fear_opportunity'] = True
            
            # Crisis regime
            if analysis.regime.value == 'crisis':
                alerts['crisis_regime_detected'] = True
            
            # Maximum accumulation conditions
            if (analysis.regime.value in ['crisis', 'recession'] and 
                analysis.fear_greed_index < 25 and 
                analysis.overall_signal == 'bullish'):
                alerts['maximum_accumulation_signal'] = True
            
            # Bubble warning
            if analysis.regime.value == 'bubble' or analysis.fear_greed_index > 85:
                alerts['bubble_warning'] = True
            
            # Regime change (would need historical tracking)
            # This would require storing previous regime states
            
            return alerts
            
        except Exception as e:
            logger.warning(f"Failed to check alert conditions: {e}")
            return {}
    
    # Return the methods to be added to the FibonacciEngine class
    return {
        '_apply_macro_adjustments': _apply_macro_adjustments,
        'get_macro_enhanced_recommendations': get_macro_enhanced_recommendations,
        '_get_macro_context': _get_macro_context,
        '_determine_macro_action': _determine_macro_action,
        '_generate_macro_reasoning': _generate_macro_reasoning,
        'get_extreme_fibonacci_levels': get_extreme_fibonacci_levels,
        'calculate_macro_opportunity_score': calculate_macro_opportunity_score,
        'get_macro_alert_conditions': get_macro_alert_conditions
    }
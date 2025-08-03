#!/usr/bin/env python3
"""
🌸 Macro Update Method for Nanpin Bot
To be integrated into the main launcher
"""

async def _update_macro_analysis(self):
    """Update macro economic analysis periodically"""
    try:
        now = datetime.now()
        update_frequency = 1800  # 30 minutes default
        
        # Check if update is needed
        if (self.last_macro_update and 
            (now - self.last_macro_update).total_seconds() < update_frequency):
            return
        
        logger.info("🔮 Updating macro analysis...")
        
        # Update macro analysis
        analysis = await self.macro_analyzer.update_macro_analysis()
        
        if analysis:
            logger.info(f"✅ Macro analysis updated:")
            logger.info(f"   Regime: {analysis.regime.value.upper()} (confidence: {analysis.regime_confidence:.1%})")
            logger.info(f"   Signal: {analysis.overall_signal.upper()}")
            logger.info(f"   Fear/Greed: {analysis.fear_greed_index:.1f}")
            logger.info(f"   BTC Sentiment: {analysis.bitcoin_sentiment:.1f}")
            logger.info(f"   Position Scaling: {analysis.position_scaling_factor:.2f}x")
            logger.info(f"   Risk Adjustment: {analysis.risk_adjustment:.2f}x")
            
            # Display key macro indicators
            if analysis.indicators:
                logger.info("   📊 Key Indicators:")
                for name, indicator in list(analysis.indicators.items())[:5]:  # Show top 5
                    if indicator.source == 'fred':
                        logger.info(f"      {indicator.name}: {indicator.value:.2f} ({indicator.signal})")
            
            # Check for extreme conditions that warrant alerts
            if analysis.fear_greed_index < 20:
                logger.warning("🚨 EXTREME FEAR DETECTED - Maximum accumulation opportunity!")
            elif analysis.regime.value == 'crisis':
                logger.warning("🚨 CRISIS REGIME DETECTED - High accumulation opportunity!")
            elif analysis.fear_greed_index > 85:
                logger.warning("⚠️ EXTREME GREED DETECTED - Exercise caution!")
            
            self.last_macro_update = now
        else:
            logger.warning("⚠️ Failed to update macro analysis")
        
    except Exception as e:
        logger.error(f"❌ Failed to update macro analysis: {e}")


# Enhanced check_trading_opportunities method with macro intelligence
async def _check_trading_opportunities_enhanced(self):
    """Check for Nanpin trading opportunities with macro intelligence"""
    try:
        # Get current price
        current_price = await self.backpack_client.get_btc_price()
        if not current_price:
            logger.warning("⚠️ Could not get current BTC price")
            return
        
        # Get enhanced Fibonacci recommendations with macro context
        if not self.fibonacci_engine.current_levels:
            logger.debug("📐 No Fibonacci levels available yet")
            return
        
        # Use macro-enhanced recommendations if available
        if hasattr(self.fibonacci_engine, 'get_macro_enhanced_recommendations'):
            recommendations = self.fibonacci_engine.get_macro_enhanced_recommendations(current_price)
        else:
            recommendations = self.fibonacci_engine.get_position_scaling_recommendations(current_price)
        
        # Check for buy signals with enhanced macro context
        for level_name, rec in recommendations.items():
            if rec['action'] == 'BUY':
                # Enhanced urgency logic
                urgency = rec.get('urgency', 'LOW')
                
                # Upgrade urgency for extreme macro opportunities
                if urgency in ['HIGH', 'MEDIUM', 'EXTREME']:
                    await self._evaluate_buy_opportunity_enhanced(level_name, rec, current_price)
                elif urgency == 'LOW' and self.macro_analyzer and self.macro_analyzer.current_analysis:
                    # Check if macro conditions warrant action despite low technical urgency
                    macro_analysis = self.macro_analyzer.current_analysis
                    if (macro_analysis.fear_greed_index < 25 or 
                        macro_analysis.regime.value in ['crisis', 'recession']):
                        logger.info(f"🔮 Macro override: Considering {level_name} despite low technical urgency")
                        await self._evaluate_buy_opportunity_enhanced(level_name, rec, current_price)
        
    except Exception as e:
        logger.error(f"❌ Failed to check trading opportunities: {e}")


async def _evaluate_buy_opportunity_enhanced(self, level_name: str, recommendation: Dict, current_price: float):
    """Enhanced buy opportunity evaluation with macro context"""
    try:
        logger.info(f"🎯 Evaluating macro-enhanced buy opportunity at {level_name}")
        logger.info(f"   Target Price: ${recommendation['target_price']:,.2f}")
        logger.info(f"   Current Price: ${current_price:,.2f}")
        logger.info(f"   Distance: {recommendation['current_distance_pct']:+.2f}%")
        
        # Enhanced reasoning with macro context
        reasoning = recommendation.get('reasoning', 'Standard analysis')
        logger.info(f"   Reasoning: {reasoning}")
        
        # Display macro context if available
        macro_context = recommendation.get('macro_context', {})
        if macro_context.get('macro_available'):
            logger.info(f"   🔮 Macro Context:")
            logger.info(f"      Regime: {macro_context['regime']} (confidence: {macro_context['regime_confidence']:.1%})")
            logger.info(f"      Fear/Greed: {macro_context['fear_greed_index']:.1f}")
            logger.info(f"      Scaling Factor: {macro_context['position_scaling_factor']:.2f}x")
        
        # Check if we're at the level (enhanced with macro urgency)
        distance_threshold = -0.5  # Default
        urgency = recommendation.get('urgency', 'LOW')
        
        if urgency == 'EXTREME':
            distance_threshold = 0.5  # More aggressive entry during extreme macro opportunities
        elif urgency == 'HIGH':
            distance_threshold = -0.2
        elif urgency == 'MEDIUM':
            distance_threshold = -0.5
        
        if recommendation['current_distance_pct'] > distance_threshold:
            logger.info(f"   ⏳ Price not sufficiently below target level (threshold: {distance_threshold:.1f}%), waiting...")
            return
        
        # Calculate position size with macro adjustments
        base_amount = self.config['position_scaling']['base_usdc_amount']
        
        # Use macro-adjusted multiplier if available
        if 'macro_multiplier' in recommendation:
            multiplier = recommendation['macro_multiplier']
            logger.info(f"   📊 Using macro-adjusted multiplier: {multiplier:.2f}x")
        else:
            multiplier = recommendation.get('adjusted_multiplier', 1.0)
        
        target_usdc_amount = base_amount * multiplier
        
        # Enhanced risk management
        safe_amount = await self.backpack_client.calculate_safe_order_size(target_usdc_amount)
        
        if safe_amount <= 0:
            logger.warning("   🚨 Risk management prevented trade")
            return
        
        # Check cooldown (enhanced for extreme conditions)
        if not self._check_scaling_cooldown():
            # Override cooldown for extreme macro opportunities
            if urgency == 'EXTREME':
                logger.info("   🚨 Overriding cooldown for extreme macro opportunity")
            else:
                logger.info("   ⏱️ Scaling cooldown active, skipping trade")
                return
        
        # Execute the trade
        logger.info(f"   💰 Executing Macro-Enhanced Nanpin buy: ${safe_amount:.2f} USDC")
        
        reason = f"Nanpin {level_name} Fibonacci level (macro: {macro_context.get('regime', 'unknown')})"
        order_result = await self.backpack_client.market_buy_btc(safe_amount, reason)
        
        if order_result:
            logger.info(f"   ✅ Macro-enhanced Nanpin buy executed successfully!")
            logger.info(f"      Order ID: {order_result.get('id', 'N/A')}")
            logger.info(f"      Macro Signal: {recommendation.get('macro_context', {}).get('overall_signal', 'unknown')}")
            
            # Update position tracking
            self._update_position_tracking(safe_amount, order_result)
            
            # Enhanced trade logging with macro context
            await self._log_trade_enhanced(level_name, safe_amount, current_price, order_result, macro_context)
        else:
            logger.error("   ❌ Failed to execute macro-enhanced Nanpin buy")
        
    except Exception as e:
        logger.error(f"❌ Failed to evaluate enhanced buy opportunity: {e}")


async def _log_trade_enhanced(self, level_name: str, usdc_amount: float, price: float, 
                            order_result: Dict, macro_context: Dict):
    """Enhanced trade logging with macro context"""
    try:
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'strategy': '永久ナンピン Enhanced',
            'fibonacci_level': level_name,
            'usdc_amount': usdc_amount,
            'btc_price': price,
            'order_id': order_result.get('id'),
            'fill_price': order_result.get('fillPrice'),
            'quantity': order_result.get('quantity'),
            'status': order_result.get('status'),
            'position_tracker': self.position_tracker.copy(),
            'macro_context': macro_context
        }
        
        # Enhanced logging output
        logger.info(f"📝 Enhanced Trade Executed:")
        logger.info(f"   Level: {level_name} | Amount: ${usdc_amount:.2f}")
        logger.info(f"   Macro Regime: {macro_context.get('regime', 'unknown')}")
        logger.info(f"   Fear/Greed: {macro_context.get('fear_greed_index', 'unknown')}")
        logger.info(f"   Overall Signal: {macro_context.get('overall_signal', 'unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log enhanced trade: {e}")
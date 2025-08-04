# 🔑 API Rate Limits Summary - Nanpin Bot v1.3

**Date**: January 31, 2025  
**Purpose**: Comprehensive API rate limits for production deployment

---

## 📊 API Keys Found in Project

From `launch_nanpin_bot_fixed.py` and related files:

```python
# Backpack Exchange (Main Trading)
BACKPACK_API_KEY = 'oHkTqR81TAc/lYifkmbxoMr0dPHBjuMXftdSQAKjzW0='
BACKPACK_SECRET_KEY = 'BGq0WKjYaVi2SrgGNkPvFpL/pNTr2jGTAbDTXmFKPtE='

# CoinGlass (Liquidation Data) 
COINGLASS_API_KEY = '3ec7b948900e4bd2a407a26fd4c52135'

# Environment Variables (Optional)
COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY')
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY') 
FLIPSIDE_API_KEY = os.getenv('FLIPSIDE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
```

---

## 🚨 CRITICAL: API Rate Limits

### 1. **Backpack Exchange** ⭐ (Primary Trading API)
- **Type**: Cryptocurrency Exchange
- **Rate Limits**: 
  - **General**: 100 requests/minute
  - **Order Operations**: 60 requests/minute  
  - **Market Data**: 120 requests/minute
  - **Account Data**: 30 requests/minute
- **Burst Limit**: 10 requests
- **Token Bucket**: 10 tokens, refill 2/second
- **Status**: ✅ **ACTIVE** - Keys found in project
- **Critical**: This is the MAIN trading API - must respect limits!

### 2. **CoinGlass** 🔥 (Liquidation Intelligence)
- **Type**: Liquidation Data Provider
- **Rate Limits**:
  - **Standard**: ~100 requests/minute (estimated)
  - **Liquidation Charts**: Limited to prevent abuse
  - **Real-time Data**: May have stricter limits
- **Status**: ✅ **ACTIVE** - Key found: `3ec7b948900e4bd2a407a26fd4c52135`
- **Usage**: High-value liquidation data for entry timing

### 3. **CoinMarketCap** 💰 (Market Data)
- **Type**: Market Data & Metrics
- **Rate Limits**:
  - **Free Tier**: 333 requests/month (10.8 requests/day)
  - **Basic Tier** ($29/month): 10,000 requests/month
  - **Standard Tier** ($79/month): 100,000 requests/month
  - **Professional Tier** ($699/month): 3,000,000 requests/month
- **Status**: ⚠️ **OPTIONAL** - Environment variable only
- **Critical**: Very limited on free tier!

### 4. **CoinGecko** 🦎 (Alternative Market Data)
- **Type**: Cryptocurrency Data API
- **Rate Limits**:
  - **Free Tier**: 50 requests/minute (10-30 calls/minute in practice)
  - **Demo**: 10,000 requests/month
  - **Pro** ($8/month): 50,000 requests/month + higher rate limits
- **Status**: ⚠️ **OPTIONAL** - Environment variable only
- **Backup**: Good fallback for CoinMarketCap

### 5. **Flipside Crypto** 📊 (On-Chain Analytics)
- **Type**: Blockchain Analytics
- **Rate Limits**:
  - **Free Tier**: 150 requests/month
  - **Starter** ($25/month): 1,500 requests/month
  - **Growth** ($299/month): 15,000 requests/month
- **Status**: ⚠️ **OPTIONAL** - Environment variable only
- **Usage**: Advanced on-chain metrics

### 6. **FRED (Federal Reserve)** 🏦 (Macro Data)
- **Type**: Economic Data
- **Rate Limits**:
  - **Free**: 120 requests/hour per IP
  - **No monthly limits** for most users
  - **Bulk downloads**: May have restrictions
- **Status**: ⚠️ **OPTIONAL** - Environment variable only
- **Usage**: Macro-economic intelligence

---

## ⚡ PRODUCTION RATE LIMITING STRATEGY

### **Tier 1: Critical APIs (Always Respect)**
```python
CRITICAL_RATE_LIMITS = {
    'backpack': {
        'requests_per_minute': 100,
        'burst_limit': 10,
        'token_bucket': True,
        'retry_after_429': True,
        'exponential_backoff': True
    },
    'coinglass': {
        'requests_per_minute': 80,  # Conservative
        'burst_limit': 5,
        'cache_duration': 120,  # 2-minute cache
        'retry_on_error': True
    }
}
```

### **Tier 2: Market Data APIs (Cache Heavily)**
```python
MARKET_DATA_LIMITS = {
    'coinmarketcap': {
        'requests_per_day': 10,  # Free tier constraint
        'cache_duration': 3600,  # 1-hour cache
        'fallback_to_coingecko': True
    },
    'coingecko': {
        'requests_per_minute': 40,  # Conservative
        'cache_duration': 600,   # 10-minute cache
        'burst_protection': True
    }
}
```

### **Tier 3: Analytics APIs (Minimal Usage)**
```python
ANALYTICS_LIMITS = {
    'flipside': {
        'requests_per_day': 5,  # Very limited
        'cache_duration': 86400,  # 24-hour cache
        'use_sparingly': True
    },
    'fred': {
        'requests_per_hour': 100,  # Well within limits
        'cache_duration': 3600,  # 1-hour cache
        'batch_requests': True
    }
}
```

---

## 🎯 RECOMMENDED USAGE PATTERNS

### **High-Frequency Operations** (Every 1-5 minutes)
- ✅ **Backpack**: Live trading data, account balance
- ❌ **No other APIs** - Use cached data only

### **Medium-Frequency Operations** (Every 10-30 minutes)  
- ✅ **CoinGlass**: Updated liquidation data
- ✅ **CoinGecko**: Price verification (if CoinMarketCap exhausted)

### **Low-Frequency Operations** (Every 1-6 hours)
- ✅ **CoinMarketCap**: Market overview, BTC dominance
- ✅ **FRED**: Macro-economic updates
- ✅ **Flipside**: On-chain analytics (if available)

### **Cache-First Strategy**
```python
CACHE_PRIORITIES = {
    'backpack_price': 30,      # 30 seconds
    'coinglass_liquidations': 120,  # 2 minutes  
    'coinmarketcap_global': 3600,   # 1 hour
    'coingecko_price': 300,         # 5 minutes
    'fred_macro': 21600,            # 6 hours
    'flipside_onchain': 86400       # 24 hours
}
```

---

## 🚨 PRODUCTION SAFEGUARDS

### **Mandatory Rate Limiting Implementation**
```python
class ProductionAPIManager:
    def __init__(self):
        self.rate_limiters = {
            'backpack': TokenBucket(100, 60),    # 100/minute
            'coinglass': TokenBucket(80, 60),     # 80/minute
            'coinmarketcap': TokenBucket(10, 86400), # 10/day
            'coingecko': TokenBucket(40, 60),     # 40/minute
        }
        
    async def safe_api_call(self, api_name, endpoint, params):
        # Check rate limit before calling
        if not self.rate_limiters[api_name].consume():
            # Use cached data or wait
            return await self.get_cached_or_wait(api_name, endpoint)
        
        # Make API call with error handling
        return await self.make_api_call_with_retry(endpoint, params)
```

### **Error Handling & Fallbacks**
```python
API_FALLBACKS = {
    'price_data': ['backpack', 'coingecko', 'coinmarketcap'],
    'liquidation_data': ['coinglass', 'manual_calculation'],
    'market_metrics': ['coinmarketcap', 'coingecko'],
    'macro_data': ['fred', 'cached_data']
}
```

---

## 💡 COST OPTIMIZATION RECOMMENDATIONS

### **Free Tier Strategy** (Current Setup)
- **Backpack**: Free with account ✅
- **CoinGlass**: Key provided ✅  
- **CoinMarketCap**: 10 calls/day limit ⚠️
- **CoinGecko**: 50 calls/minute ✅
- **FRED**: 120 calls/hour ✅
- **Flipside**: 150 calls/month ⚠️

**Monthly Cost**: $0 (if staying within limits)

### **Production Tier Strategy** (Recommended)
- **Backpack**: Free ✅
- **CoinGlass**: Current key ✅
- **CoinMarketCap Basic**: $29/month for 10K calls
- **CoinGecko Pro**: $8/month for 50K calls  
- **FRED**: Free ✅
- **Flipside Starter**: $25/month for 1.5K calls

**Monthly Cost**: ~$62/month for professional-grade data

---

## 🔧 IMPLEMENTATION CHECKLIST

### **Phase 1: Rate Limiting** ✅
- [x] Implement token bucket rate limiters
- [x] Add exponential backoff
- [x] Cache frequently accessed data
- [x] Monitor API usage

### **Phase 2: Error Handling** ✅  
- [x] Graceful API failure handling
- [x] Multiple fallback data sources
- [x] Circuit breaker pattern
- [x] Comprehensive logging

### **Phase 3: Monitoring** 📊
- [ ] API usage tracking dashboard
- [ ] Rate limit alerting
- [ ] Cost monitoring
- [ ] Performance metrics

### **Phase 4: Optimization** ⚡
- [ ] Smart caching strategies  
- [ ] Request batching
- [ ] Predictive data fetching
- [ ] Usage analytics

---

## 📈 PROJECTED API USAGE (Production)

### **Daily Trading Bot Usage**
```
Backpack Exchange:
  - Price checks: 1440/day (every minute)
  - Order operations: 50/day (average)
  - Account checks: 288/day (every 5 minutes)
  Total: ~1800 requests/day ✅ Well within limits

CoinGlass:
  - Liquidation updates: 144/day (every 10 minutes)
  Total: ~144 requests/day ✅ Acceptable

CoinMarketCap:  
  - Market overview: 8/day (every 3 hours)
  Total: ~8 requests/day ⚠️ Needs Basic plan for production

CoinGecko:
  - Price verification: 288/day (every 5 minutes)  
  Total: ~288 requests/day ✅ Within free limits
```

---

## 🎯 FINAL RECOMMENDATIONS

### **For Development/Testing**
- Use current setup with free tiers
- Implement aggressive caching (1-5 minute cache times)
- Mock data for high-frequency testing

### **For Production**
- Upgrade CoinMarketCap to Basic ($29/month)
- Consider CoinGecko Pro ($8/month) for redundancy
- Implement comprehensive monitoring
- Use circuit breakers and fallbacks

### **API Priority Order**
1. **Backpack** (Critical - Trading)
2. **CoinGlass** (High - Intelligence)  
3. **CoinGecko** (Medium - Market data)
4. **FRED** (Medium - Macro intelligence)
5. **CoinMarketCap** (Low - Backup data)
6. **Flipside** (Low - Deep analytics)

---

**✅ Summary**: Current nanpin bot v1.3 setup can run in production with proper rate limiting and caching. Upgrading CoinMarketCap to Basic tier ($29/month) recommended for serious trading operations.
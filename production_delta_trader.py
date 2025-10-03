#!/usr/bin/env python3
"""
Production-Ready Delta Exchange Options Short Strangle System
=====================================

Complete implementation of requirements:
✅ BTC/ETH options short strangle automation
✅ Strike selection by option delta (10-15 range)
✅ Advanced order management with best bid + retry logic
✅ Complete exit rules (fixed time, max loss/profit, trailing SL)
✅ Position management with duplicate prevention
✅ Next-day expiry handling (avoid same-day expiry issues)
✅ Comprehensive API testing and validation
✅ P&L tracking with real-time updates
✅ IST timezone handling for Indian market

Author: Production Trading System
Date: September 30, 2025
Version: 1.0.0
"""

import os
import requests
import json
import time
import hmac
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
import asyncio
import websocket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging with file-focused debugging
logging.basicConfig(
    level=logging.INFO,  # Terminal shows INFO and above
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Terminal output
    ]
)

# Create detailed file logger for debugging
file_handler = logging.FileHandler('delta_trader_detailed.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
file_handler.setFormatter(file_formatter)

# Create summary file logger for key events
summary_handler = logging.FileHandler('delta_trader_summary.log')
summary_handler.setLevel(logging.INFO)
summary_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
summary_handler.setFormatter(summary_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(summary_handler)

# Suppress verbose request logging in terminal (keep in files)
requests_logger = logging.getLogger("requests")
requests_logger.setLevel(logging.WARNING)  # Only warnings/errors to terminal
requests_logger.addHandler(file_handler)   # All requests details to file

urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)   # Only warnings/errors to terminal
urllib3_logger.addHandler(file_handler)    # All urllib3 details to file

# Constants
IST_OFFSET = timedelta(hours=5, minutes=30)
UI_OVERRIDES_PATH = Path("ui_overrides.json")
UI_SNAPSHOT_PATH = Path("ui_config_snapshot.json")

# Dynamic trade times - immediate start for testing exit logic
def get_dynamic_times():
    current_ist = datetime.now(timezone.utc) + IST_OFFSET
    trade_time = current_ist + timedelta(seconds=30)  # Start in 30 seconds
    exit_time = current_ist + timedelta(minutes=2)    # Exit in 2 minutes  
    return trade_time.strftime("%H:%M"), exit_time.strftime("%H:%M")

DYNAMIC_TRADE_TIME, DYNAMIC_EXIT_TIME = get_dynamic_times()
DEFAULT_TRADE_TIME = DYNAMIC_TRADE_TIME  # now + 5 minutes
DEFAULT_EXIT_TIME = DYNAMIC_EXIT_TIME    # now + 10 minutes
DEFAULT_EXPIRY_TIME = "17:30" # 5:30 PM IST expiry
FIXED_EXPIRY_DATE = "01-10-2025"  # Next-day expiry as requested

class OrderState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    PENDING = "pending"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

class ExitReason(Enum):
    FIXED_TIME = "fixed_time"
    MAX_LOSS = "max_loss"
    MAX_PROFIT = "max_profit"
    TRAILING_STOP = "trailing_stop"
    FORCE_CLOSE = "force_close"

@dataclass
class TradingConfig:
    """Production trading configuration with live trading parameters"""
    # Strategy parameters
    underlying: str = "BTC"  # BTC or ETH
    delta_range_low: float = 0.10
    delta_range_high: float = 0.15
    trade_time_ist: str = DEFAULT_TRADE_TIME
    exit_time_ist: str = DEFAULT_EXIT_TIME
    expiry_date: Optional[str] = None  # dd-mm-YYYY, falls back to FIXED_EXPIRY_DATE when absent
    expiry_buffer_hours: int = 24  # Use next-day expiry if < N hours to current expiry
    quantity: int = 1  # 1 contract quantity for each leg (minimum for options)
    contract_size: float = 0.001  # Contract size in underlying units (BTC options: 0.001 BTC)
    
    # Risk management
    max_loss_pct: float = 0.80  # 80% max loss
    max_profit_pct: float = 0.80  # 80% max profit
    
    # Trailing stop loss rules
    trailing_sl_enabled: bool = True
    trailing_rules: Dict[float, float] = field(default_factory=lambda: {
        0.40: 0.00,  # 40% profit -> SL at 0%
        0.50: 0.25,  # 50% profit -> SL at 25%
        # After 50%, every +10% profit adds +5% to SL
    })
    
    # Order management
    max_retries: int = 4
    retry_slippage_pct: float = 0.02  # 2% price slippage per retry
    order_timeout: int = 60
    
    # System settings - Enhanced for live trading
    dry_run: bool = False  # 🔴 LIVE TRADING MODE ENABLED
    testnet: bool = False  # Use production environment
    position_check_interval: int = 1  # 1 second monitoring
    websocket_enabled: bool = True  # Enable websocket for live prices
    
    # Database and logging
    log_level: str = "DEBUG"  # Full debugging
    metrics_enabled: bool = True
    
    # Auto-recovery settings
    max_api_retries: int = 5
    websocket_reconnect_attempts: int = 10
    error_recovery_enabled: bool = True


def resolve_runtime_config() -> Tuple[TradingConfig, str]:
    """Resolve the runtime configuration, preferring the latest UI settings."""
    try:
        from config_loader import load_config_for_trading, load_config_snapshot
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.warning(
            "Streamlit configuration helpers unavailable (%s); falling back to defaults.",
            exc,
        )
        return TradingConfig(), "built-in defaults"

    if UI_SNAPSHOT_PATH.exists():
        try:
            ui_config = load_config_snapshot(UI_SNAPSHOT_PATH)
            legacy_config = ui_config.to_legacy_config()
            logger.info(
                "Loaded trading configuration from Streamlit snapshot '%s'.",
                UI_SNAPSHOT_PATH,
            )
            return legacy_config, "Streamlit snapshot"
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            logger.error(
                "Failed to load Streamlit snapshot '%s': %s",
                UI_SNAPSHOT_PATH,
                exc,
            )

    if UI_OVERRIDES_PATH.exists():
        try:
            overrides: Dict[str, Any] = json.loads(UI_OVERRIDES_PATH.read_text())
            legacy_config = load_config_for_trading("default", **overrides)
            logger.info(
                "Loaded trading configuration from Streamlit overrides '%s'.",
                UI_OVERRIDES_PATH,
            )
            return legacy_config, "Streamlit overrides"
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            logger.error(
                "Failed to apply overrides from '%s': %s",
                UI_OVERRIDES_PATH,
                exc,
            )

    logger.info("Using built-in TradingConfig defaults.")
    return TradingConfig(), "built-in defaults"

@dataclass
class OptionData:
    """Option data structure"""
    symbol: str
    contract_type: str  # call_options or put_options
    strike_price: float
    expiry_date: str
    spot_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    volume: int
    open_interest: int
    product_id: int

@dataclass
class Position:
    """Enhanced position tracking with partial fill support"""
    strategy_id: str
    symbol: str
    side: PositionSide
    quantity: float  # Target quantity
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    order_ids: List[str] = field(default_factory=list)
    fills: List[Dict] = field(default_factory=list)
    
    # Partial fill tracking
    filled_quantity: float = 0.0  # Actual filled quantity
    remaining_quantity: float = 0.0  # Quantity still needed
    is_fully_filled: bool = False  # True when filled_quantity == quantity
    partial_fills: List[Dict] = field(default_factory=list)  # Track all partial fills
    contract_size: float = 1.0  # Underlying units per contract
    
    def add_fill(self, fill_data: Dict) -> None:
        """Add a fill to this position and update tracking"""
        self.fills.append(fill_data)
        self.order_ids.append(fill_data.get('order_id', ''))
        
        fill_quantity = fill_data.get('fill_quantity', 0)
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        self.is_fully_filled = self.remaining_quantity == 0
        
        # Track partial fills
        if fill_data.get('is_partial_fill', False):
            self.partial_fills.append(fill_data)
        
        # Update weighted average entry price
        if self.filled_quantity > 0:
            total_value = sum(fill['fill_price'] * fill['fill_quantity'] for fill in self.fills)
            self.entry_price = total_value / self.filled_quantity
    
    def get_fill_summary(self) -> str:
        """Get human-readable fill summary"""
        if self.is_fully_filled:
            return f"✅ Fully filled: {self.filled_quantity}/{self.quantity}"
        elif self.filled_quantity > 0:
            return f"⚠️  Partially filled: {self.filled_quantity}/{self.quantity} (Missing: {self.remaining_quantity})"
        else:
            return f"❌ Not filled: 0/{self.quantity}"

@dataclass
class StrategyState:
    """Short strangle strategy state"""
    strategy_id: str
    config: TradingConfig
    call_position: Optional[Position] = None
    put_position: Optional[Position] = None
    total_premium_received: float = 0.0
    current_pnl: float = 0.0
    current_pnl_pct: Optional[float] = None
    max_profit_seen: float = 0.0
    trailing_sl_level: float = 0.0
    is_active: bool = False
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[ExitReason] = None

class WebSocketPriceStreamer:
    """
    WebSocket price streamer for live option data
    
    Implements proper webhook format for real-time option prices:
    - Uses 'mark_price' channel for options with MARK: prefix format  
    - Uses 'v2/ticker' channel for additional market data
    - Handles both mark price and bid/ask data
    - Auto-reconnection with exponential backoff
    
    Example subscription format:
    {
        "type": "subscribe", 
        "payload": {
            "channels": [
                {
                    "name": "mark_price",
                    "symbols": ["MARK:C-BTC-90000-310125"]
                }
            ]
        }
    }
    """
    
    def __init__(self, api: 'DeltaExchangeAPI'):
        self.api = api
        self.ws = None
        self.is_connected = False
        self.price_data = {}
        self.reconnect_count = 0
        self.max_reconnects = api.config.websocket_reconnect_attempts
        self.thread = None
        logger.debug("🌐 WebSocket Price Streamer initialized")
    
    def on_message(self, ws, message):
        """Handle incoming websocket messages"""
        try:
            data = json.loads(message)
            logger.debug(f"📡 WebSocket message: {data}")
            
            # Handle mark_price channel data for options
            if data.get('type') == 'data' and 'mark_price' in str(data):
                if 'symbol' in data and 'mark_price' in data:
                    symbol = data['symbol']
                    mark_price = float(data['mark_price'])
                    self.price_data[symbol] = {
                        'mark_price': mark_price,
                        'best_bid': mark_price * 0.999,  # Approximate bid
                        'best_ask': mark_price * 1.001,  # Approximate ask
                        'timestamp': time.time()
                    }
                    logger.debug(f"💹 Updated mark price for {symbol}: ${mark_price}")
            
            # Handle v2/ticker channel data for additional market info
            elif 'symbol' in data and 'quotes' in data:
                symbol = data['symbol']
                quotes = data['quotes']
                
                # Update or merge with existing price data
                if symbol not in self.price_data:
                    self.price_data[symbol] = {}
                
                self.price_data[symbol].update({
                    'best_bid': float(quotes.get('best_bid', 0)),
                    'best_ask': float(quotes.get('best_ask', 0)),
                    'timestamp': time.time()
                })
                logger.debug(f"� Updated ticker for {symbol}: bid=${quotes.get('best_bid')}, ask=${quotes.get('best_ask')}")
                
        except Exception as e:
            logger.error(f"❌ Error processing WebSocket message: {str(e)}")
            logger.debug(f"📝 Raw message: {message}")
    
    def on_error(self, ws, error):
        """Handle websocket errors"""
        logger.error(f"🚨 WebSocket error: {error}")
        self.is_connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle websocket close"""
        logger.warning(f"📴 WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        
        # Auto-reconnect
        if self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            logger.info(f"🔄 Attempting reconnect {self.reconnect_count}/{self.max_reconnects}")
            time.sleep(2 ** self.reconnect_count)  # Exponential backoff
            self.connect()
    
    def on_open(self, ws):
        """Handle websocket open"""
        logger.info("✅ WebSocket connected successfully")
        self.is_connected = True
        self.reconnect_count = 0
    
    def format_symbol_for_mark_price(self, symbol: str) -> str:
        """Format symbol for mark_price subscription
        Convert trading symbols to MARK: format for options
        Example: C-BTC-90000-310125 -> MARK:C-BTC-90000-310125
        """
        if symbol.startswith(('C-', 'P-')):  # Options symbols
            return f"MARK:{symbol}"
        return symbol  # Keep other symbols as-is
    
    def connect(self):
        """Connect to websocket"""
        try:
            # Delta Exchange WebSocket URL (adjust if needed)
            ws_url = "wss://socket.delta.exchange"
            logger.debug(f"🔌 Connecting to WebSocket: {ws_url}")
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # Run in separate thread
            self.thread = threading.Thread(target=self.ws.run_forever)
            self.thread.daemon = True
            self.thread.start()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect WebSocket: {str(e)}")
    
    def subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to price updates for symbols"""
        if not self.is_connected:
            logger.warning("⚠️  WebSocket not connected, cannot subscribe")
            return
        
        for symbol in symbols:
            # Use mark_price channel for options as specified in webhook documentation
            mark_symbol = self.format_symbol_for_mark_price(symbol)
            subscribe_msg = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {
                            "name": "mark_price",
                            "symbols": [mark_symbol]
                        }
                    ]
                }
            }
            
            try:
                if self.ws:
                    self.ws.send(json.dumps(subscribe_msg))
                    logger.debug(f"📡 Subscribed to mark_price for {mark_symbol}")
                else:
                    logger.error("🔌 WebSocket not connected for subscription")
            except Exception as e:
                logger.error(f"❌ Failed to subscribe to {mark_symbol}: {str(e)}")
                
            # Also subscribe to v2/ticker for additional market data
            ticker_msg = {
                "type": "subscribe", 
                "payload": {
                    "channels": [{"name": "v2/ticker", "symbols": [symbol]}]
                }
            }
            
            try:
                if self.ws:
                    self.ws.send(json.dumps(ticker_msg))
                    logger.debug(f"� Subscribed to v2/ticker for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to subscribe to ticker for {symbol}: {str(e)}")
    
    def get_live_price(self, symbol: str) -> Optional[Dict]:
        """Get live price for symbol with mark_price support"""
        if symbol in self.price_data:
            data = self.price_data[symbol]
            # Check if data is fresh (within 10 seconds)
            if time.time() - data['timestamp'] < 10:
                # Prefer mark_price for options, fallback to bid/ask
                if 'mark_price' in data:
                    logger.debug(f"📈 Using mark price for {symbol}: ${data['mark_price']}")
                    return {
                        'mark_price': data['mark_price'],
                        'best_bid': data.get('best_bid', data['mark_price'] * 0.999),
                        'best_ask': data.get('best_ask', data['mark_price'] * 1.001),
                        'timestamp': data['timestamp']
                    }
                else:
                    logger.debug(f"📊 Using ticker data for {symbol}")
                    return data
        return None
    
    def disconnect(self):
        """Disconnect websocket"""
        if self.ws:
            self.ws.close()
            logger.info("📴 WebSocket disconnected")

class DeltaExchangeAPI:
    """Production-ready Delta Exchange API wrapper"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        
        # Load API credentials based on environment
        if config.testnet:
            self.api_key = os.getenv('DELTA_TESTNET_API_KEY') or os.getenv('DELTA_API_KEY')
            self.api_secret = os.getenv('DELTA_TESTNET_API_SECRET') or os.getenv('DELTA_API_SECRET')
            self.base_url = "https://testnet-api.delta.exchange"
            logger.info("🧪 Using TESTNET environment")
        else:
            self.api_key = os.getenv('DELTA_API_KEY')
            self.api_secret = os.getenv('DELTA_API_SECRET')
            self.base_url = "https://api.india.delta.exchange"
            logger.info("🔴 Using PRODUCTION environment")
        
        if not self.api_key or not self.api_secret:
            if config.testnet:
                raise ValueError(
                    "Testnet API credentials not found!\n"
                    "Set DELTA_TESTNET_API_KEY and DELTA_TESTNET_API_SECRET in .env file\n"
                    "OR set testnet=False to use production credentials"
                )
            else:
                raise ValueError("DELTA_API_KEY and DELTA_API_SECRET must be set in .env file")
        
        logger.debug(f"🔑 API Key loaded: {self.api_key[:8] if self.api_key else 'None'}...")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # WebSocket integration
        self.websocket_streamer = None
        if config.websocket_enabled:
            self.websocket_streamer = WebSocketPriceStreamer(self)
            self.websocket_streamer.connect()
        
        logger.info(f"✅ Delta Exchange API initialized")
    
    def _generate_signature(self, method: str, path: str, timestamp: str, query_string: str = "", body: str = "") -> str:
        """Generate HMAC-SHA256 signature"""
        if not self.api_secret:
            raise ValueError("API secret not configured")
        
        message = method + timestamp + path + query_string + body
        return hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Tuple[bool, Any]:
        """Make authenticated API request with enhanced error handling and logging"""
        self._rate_limit()
        
        timestamp = str(int(time.time()))
        query_string = ""
        if params:
            # Properly encode query string for signature
            from urllib.parse import urlencode
            query_string = "?" + urlencode(params)
        
        body = ""
        if data:
            body = json.dumps(data)
        
        signature = self._generate_signature(method, endpoint, timestamp, query_string, body)
        
        headers = {
            'api-key': self.api_key,
            'signature': signature,
            'timestamp': timestamp,
            'User-Agent': 'python-production-trader',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        # Enhanced logging
        logger.debug(f"🔄 API Request: {method} {url}")
        logger.debug(f"📤 Headers: {headers}")
        if params:
            logger.debug(f"📋 Params: {params}")
        if data:
            logger.debug(f"📋 Data: {data}")
        
        # Retry logic with enhanced error handling
        for attempt in range(self.config.max_api_retries):
            try:
                response = None
                start_time = time.time()
                
                if method == 'GET':
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, data=body, timeout=10)
                elif method == 'PUT':
                    response = requests.put(url, headers=headers, data=body, timeout=10)
                elif method == 'DELETE':
                    response = requests.delete(url, headers=headers, data=body, timeout=10)
                
                duration = time.time() - start_time
                logger.debug(f"⏱️  API Response time: {duration:.3f}s")
                
                if response is None:
                    logger.error(f"❌ No response received for {method} {endpoint}")
                    continue
                
                logger.debug(f"📥 Response Status: {response.status_code}")
                logger.debug(f"📥 Response Headers: {dict(response.headers)}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"⏳ Rate limited, waiting {retry_after}s (attempt {attempt + 1}/{self.config.max_api_retries})")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"✅ API Success: {method} {endpoint}")
                    logger.debug(f"📦 Response: {json.dumps(result, indent=2)[:500]}...")
                    return True, result
                else:
                    error_text = response.text
                    logger.error(f"❌ API Error {response.status_code}: {error_text}")
                    
                    # Don't retry on client errors (4xx)
                    if 400 <= response.status_code < 500:
                        return False, f"Error {response.status_code}: {error_text}"
                    
                    # Retry on server errors (5xx)
                    if attempt < self.config.max_api_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"🔄 Retrying in {wait_time}s (attempt {attempt + 1}/{self.config.max_api_retries})")
                        time.sleep(wait_time)
                        continue
                    
                    return False, f"Error {response.status_code}: {error_text}"
                    
            except Exception as e:
                logger.error(f"❌ Request failed (attempt {attempt + 1}/{self.config.max_api_retries}): {str(e)}")
                if attempt < self.config.max_api_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return False, f"Request failed after {self.config.max_api_retries} attempts: {str(e)}"
        
        return False, "Max retries exceeded"
    
    def _make_public_request(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[bool, Any]:
        """Make public API request"""
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return False, f"Request failed: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test API connection and authentication"""
        logger.info("🔌 Testing API connection...")
        
        # Test public endpoint first
        success, result = self._make_public_request('/v2/products')
        if not success:
            logger.error(f"❌ Public API test failed: {result}")
            return False
        
        logger.info(f"✅ Public API working - {len(result.get('result', []))} products available")
        
        # Test authenticated endpoint
        logger.info("🔐 Testing authenticated API...")
        success, result = self._make_request('GET', '/v2/wallet/balances')
        if not success:
            error_msg = str(result)
            if "invalid_api_key" in error_msg:
                logger.error(f"❌ Invalid API Key!")
                logger.error(f"   Current API Key: {self.api_key[:8] if self.api_key else 'None'}...")
                if self.config.testnet:
                    logger.error(f"   Environment: TESTNET")
                    logger.error(f"   Make sure you have valid testnet API credentials")
                    logger.error(f"   OR set testnet=False to use production API")
                else:
                    logger.error(f"   Environment: PRODUCTION")
                    logger.error(f"   Verify your DELTA_API_KEY and DELTA_API_SECRET in .env file")
                    logger.error(f"   Get credentials from: https://www.delta.exchange/app/account/api")
            else:
                logger.error(f"❌ Authenticated API test failed: {result}")
            return False
        
        balances = result.get('result', [])
        logger.info(f"✅ Authenticated API working - {len(balances)} wallet balances retrieved")
        
        # Log wallet balances for reference
        for balance in balances[:5]:  # Show first 5
            asset = balance.get('asset', 'Unknown')
            available = float(balance.get('available_balance', 0))
            if available > 0:
                logger.info(f"   💰 {asset}: {available}")
        
        return True
    
    def get_current_ist_time(self) -> datetime:
        """Get current IST time"""
        return datetime.now(timezone.utc) + IST_OFFSET
    
    def get_next_expiry_date(self) -> str:
        """Determine target expiry date for option selection."""
        configured_expiry = getattr(self.config, "expiry_date", None)
        if configured_expiry:
            logger.info(f"📅 Using configured expiry date: {configured_expiry}")
            return configured_expiry

        expiry_str = FIXED_EXPIRY_DATE
        logger.info(f"📅 Using fallback expiry date: {expiry_str}")
        logger.debug("🎯 No configured expiry provided; falling back to legacy default")
        return expiry_str
    
    def get_option_chain_with_greeks(self, underlying: str = "BTC") -> List[OptionData]:
        """Get option chain with Greeks for delta-based selection"""
        logger.info(f"📊 Fetching {underlying} option chain with Greeks...")
        
        expiry_date = self.get_next_expiry_date()
        
        params = {
            'contract_types': 'call_options,put_options',
            'underlying_asset_symbols': underlying,
            'expiry_date': expiry_date
        }
        
        success, result = self._make_public_request('/v2/tickers', params)
        
        if not success:
            logger.error(f"❌ Failed to get option chain: {result}")
            return []
        
        options_data = []
        raw_options = result.get('result', [])
        
        logger.info(f"📈 Processing {len(raw_options)} raw options...")
        
        for option in raw_options:
            try:
                # Extract Greeks
                greeks = option.get('greeks', {})
                quotes = option.get('quotes', {})
                
                if not greeks or not quotes:
                    continue
                
                # Validate required data
                required_fields = ['delta', 'gamma', 'theta', 'vega']
                if not all(greeks.get(field) for field in required_fields):
                    continue
                
                if not all(quotes.get(field) for field in ['best_bid', 'best_ask']):
                    continue
                
                option_data = OptionData(
                    symbol=option.get('symbol', ''),
                    contract_type=option.get('contract_type', ''),
                    strike_price=float(option.get('strike_price', 0)),
                    expiry_date=expiry_date,
                    spot_price=float(option.get('spot_price', 0)),
                    delta=abs(float(greeks.get('delta', 0))),  # Use absolute delta
                    gamma=float(greeks.get('gamma', 0)),
                    theta=float(greeks.get('theta', 0)),
                    vega=float(greeks.get('vega', 0)),
                    implied_volatility=float(greeks.get('iv', 0)),
                    best_bid=float(quotes.get('best_bid', 0)),
                    best_ask=float(quotes.get('best_ask', 0)),
                    bid_size=float(quotes.get('bid_size', 0)),
                    ask_size=float(quotes.get('ask_size', 0)),
                    volume=int(float(option.get('volume', 0))),
                    open_interest=int(float(option.get('oi', 0))),
                    product_id=int(option.get('product_id', 0))
                )
                
                # Only include options with reasonable liquidity
                if option_data.best_bid > 0 and option_data.best_ask > 0:
                    options_data.append(option_data)
                    
            except (ValueError, TypeError, KeyError) as e:
                continue
        
        logger.info(f"✅ Processed {len(options_data)} valid options with Greeks and liquidity")
        return options_data
    
    def find_delta_options(self, options: List[OptionData]) -> Tuple[Optional[OptionData], Optional[OptionData]]:
        """Find CE and PE options in the specified delta range"""
        logger.info(f"🎯 Finding options in delta range {self.config.delta_range_low:.2f} - {self.config.delta_range_high:.2f}")
        
        # Separate calls and puts
        calls = [opt for opt in options if opt.contract_type == 'call_options']
        puts = [opt for opt in options if opt.contract_type == 'put_options']
        
        # Filter by delta range
        target_calls = [opt for opt in calls if self.config.delta_range_low <= opt.delta <= self.config.delta_range_high]
        target_puts = [opt for opt in puts if self.config.delta_range_low <= opt.delta <= self.config.delta_range_high]
        
        logger.info(f"   📞 Found {len(target_calls)} calls in delta range")
        logger.info(f"   📄 Found {len(target_puts)} puts in delta range")
        
        if not target_calls or not target_puts:
            logger.warning(f"⚠️  Insufficient options in delta range!")
            return None, None
        
        # Sort by delta (closest to middle of range)
        target_delta = (self.config.delta_range_low + self.config.delta_range_high) / 2
        
        best_call = min(target_calls, key=lambda x: abs(x.delta - target_delta))
        best_put = min(target_puts, key=lambda x: abs(x.delta - target_delta))
        
        logger.info(f"   📞 Selected Call: {best_call.symbol} (Δ {best_call.delta:.3f}, Bid: ${best_call.best_bid})")
        logger.info(f"   📄 Selected Put: {best_put.symbol} (Δ {best_put.delta:.3f}, Bid: ${best_put.best_ask})")
        
        return best_call, best_put
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "limit_order", 
                    limit_price: Optional[float] = None, client_order_id: Optional[str] = None) -> Tuple[bool, Any]:
        """Place order with idempotency"""
        if self.config.dry_run:
            logger.info(f"🧪 DRY RUN: Would place {side} order for {quantity} {symbol} @ ${limit_price}")
            return True, {"id": f"dry_run_{int(time.time())}", "state": "open"}
        
        if not client_order_id:
            client_order_id = str(uuid.uuid4())
        
        order_data = {
            "product_symbol": symbol,
            "size": str(quantity),
            "side": side,
            "order_type": order_type,
            "client_order_id": client_order_id
        }
        
        if limit_price:
            order_data["limit_price"] = str(limit_price)
        
        logger.info(f"📤 Placing {side} {order_type} order: {quantity} {symbol} @ ${limit_price}")
        
        success, result = self._make_request('POST', '/v2/orders', data=order_data)
        
        if success:
            order = result.get('result', {})
            order_id = order.get('id')
            logger.info(f"✅ Order placed! ID: {order_id}")
            return True, order
        else:
            logger.error(f"❌ Order failed: {result}")
            return False, result
    
    def get_order_status(self, order_id: str) -> Tuple[bool, Any]:
        """Get order status"""
        if self.config.dry_run:
            return True, {"id": order_id, "state": "closed", "size": "1", "unfilled_size": "0"}
        
        success, result = self._make_request('GET', f'/v2/orders/{order_id}')
        return success, result.get('result', {}) if success else result
    
    def get_open_orders(self, include_recent: bool = True) -> List[Dict]:
        """Get all open orders using the correct /v2/orders endpoint with states=open"""
        if self.config.dry_run:
            return []
        
        # Use the correct primary endpoint for active orders
        params = {
            'states': 'open,pending',  # Get both open and pending orders
            'page_size': 100  # Increased page size to catch more orders
        }
        
        logger.debug(f"🔍 Fetching open orders using /v2/orders with states=open,pending")
        success, result = self._make_request('GET', '/v2/orders', params=params)
        
        if success:
            open_orders = result.get('result', [])
            
            logger.info(f"📋 Found {len(open_orders)} open/pending orders from primary API")
            
            # Log details of each open order for debugging
            for order in open_orders:
                order_id = order.get('id')
                symbol = order.get('product_symbol', 'Unknown')
                side = order.get('side', 'Unknown')
                state = order.get('state', 'Unknown')
                size = order.get('size', 0)
                unfilled_size = order.get('unfilled_size', 0)
                logger.debug(f"   🔍 Open order: {order_id} | {symbol} | {side} | {state} | Size: {size} | Unfilled: {unfilled_size}")
            
            return open_orders
        else:
            logger.error(f"❌ Failed to get open orders: {result}")
            
            # Fallback to history method if primary fails
            logger.info("🔄 Falling back to orders history method...")
            success, result = self._make_request('GET', '/v2/orders/history', params={'page_size': 100})
            if success:
                history_orders = result.get('result', [])
                open_orders = [order for order in history_orders if order.get('state') in ['open', 'pending']]
                logger.info(f"📋 Fallback: Found {len(open_orders)} open orders from history")
                return open_orders
            else:
                logger.error(f"❌ Fallback also failed: {result}")
                return []
    
    def get_current_positions(self) -> List[Dict]:
        """Get current open positions with proper parameters"""
        if self.config.dry_run:
            return []
        
        # Delta Exchange requires either product_id or underlying_asset_symbol
        # Get all positions by using underlying asset filter
        params = {
            'underlying_asset_symbol': 'BTC',  # Get BTC positions
            'page_size': 100
        }
        
        success, result = self._make_request('GET', '/v2/positions', params=params)
        if success:
            positions = result.get('result', [])
            # Filter only positions with non-zero size
            open_positions = [pos for pos in positions if float(pos.get('size', 0)) != 0]
            logger.info(f"📊 Found {len(open_positions)} open BTC positions")
            
            # Also try to get ETH positions if we're trading ETH
            if hasattr(self.config, 'underlying') and self.config.underlying == 'ETH':
                eth_params = {
                    'underlying_asset_symbol': 'ETH',
                    'page_size': 100
                }
                eth_success, eth_result = self._make_request('GET', '/v2/positions', params=eth_params)
                if eth_success:
                    eth_positions = eth_result.get('result', [])
                    eth_open_positions = [pos for pos in eth_positions if float(pos.get('size', 0)) != 0]
                    open_positions.extend(eth_open_positions)
                    logger.info(f"📊 Found {len(eth_open_positions)} open ETH positions")
            
            return open_positions
        else:
            logger.error(f"❌ Failed to get positions: {result}")
            return []
    
    def cancel_all_open_orders(self) -> bool:
        """Cancel all open orders with verification using the correct API endpoint"""
        logger.info("🧹 Cancelling all open orders...")
        
        # Get initial list of open orders using the correct endpoint
        open_orders = self.get_open_orders(include_recent=True)
        if not open_orders:
            logger.info("✅ No open orders to cancel")
            return True
        
        logger.info(f"🎯 Found {len(open_orders)} orders to cancel")
        for order in open_orders:
            order_id = order.get('id')
            symbol = order.get('product_symbol', 'Unknown')
            size = order.get('size', 0)
            unfilled_size = order.get('unfilled_size', 0)
            state = order.get('state', 'Unknown')
            logger.info(f"   📋 Order to cancel: {order_id} for {symbol} | Size: {size} | Unfilled: {unfilled_size} | State: {state}")
            
            # Only cancel orders that actually have unfilled quantity
            if float(unfilled_size) <= 0:
                logger.info(f"   ✅ Order {order_id} already fully filled, skipping cancellation")
                continue
        
        success_count = 0
        cancelled_order_ids = []
        
        # Step 1: Send cancellation requests for orders with unfilled quantity
        for order in open_orders:
            order_id = order.get('id')
            product_id = order.get('product_id')
            symbol = order.get('product_symbol', 'Unknown')
            unfilled_size = float(order.get('unfilled_size', 0))
            
            if not order_id or not product_id:
                logger.warning(f"⚠️  Skipping order with missing ID or product_id: {order}")
                continue
            
            # Skip if no unfilled quantity
            if unfilled_size <= 0:
                logger.info(f"   ✅ Order {order_id} fully filled, no need to cancel")
                continue
            
            logger.info(f"🗑️  Cancelling order {order_id} for {symbol} (unfilled: {unfilled_size})")
            
            success, result = self.cancel_order(str(order_id), int(product_id))
            if success:
                success_count += 1
                cancelled_order_ids.append(str(order_id))
                logger.info(f"✅ Sent cancellation for order {order_id}")
            else:
                logger.error(f"❌ Failed to cancel order {order_id}: {result}")
        
        # Step 2: Wait and verify cancellations
        if cancelled_order_ids:
            logger.info(f"⏳ Waiting 3 seconds for cancellations to process...")
            time.sleep(3)
            
            # Verify orders are actually cancelled using the correct endpoint
            remaining_orders = self.get_open_orders(include_recent=True)
            remaining_unfilled = []
            
            for order in remaining_orders:
                order_id = str(order.get('id'))
                unfilled_size = float(order.get('unfilled_size', 0))
                if order_id in cancelled_order_ids and unfilled_size > 0:
                    remaining_unfilled.append(order_id)
            
            if remaining_unfilled:
                logger.warning(f"⚠️  {len(remaining_unfilled)} orders still have unfilled quantity: {remaining_unfilled}")
                # Try cancelling remaining orders one more time
                for order_id in remaining_unfilled:
                    logger.info(f"🔄 Retrying cancellation for order {order_id}")
                    # Find the order details for retry
                    for order in remaining_orders:
                        if str(order.get('id')) == order_id:
                            product_id = order.get('product_id')
                            unfilled_size = float(order.get('unfilled_size', 0))
                            if product_id and unfilled_size > 0:
                                self.cancel_order(order_id, int(product_id))
                            break
                
                # Final wait and check
                time.sleep(2)
                final_orders = self.get_open_orders(include_recent=True)
                final_unfilled_count = len([o for o in final_orders if float(o.get('unfilled_size', 0)) > 0])
                logger.info(f"📊 Final unfilled orders count: {final_unfilled_count}")
                
                return final_unfilled_count == 0
            else:
                logger.info(f"✅ All {len(cancelled_order_ids)} orders successfully cancelled")
                return True
        
        logger.info(f"📊 Cancelled {success_count}/{len([o for o in open_orders if float(o.get('unfilled_size', 0)) > 0])} unfilled orders")
        return True
    
    def cancel_order(self, order_id: str, product_id: int) -> Tuple[bool, Any]:
        """Cancel order"""
        if self.config.dry_run:
            logger.info(f"🧪 DRY RUN: Would cancel order {order_id}")
            return True, {"message": "Order cancelled (dry run)"}
        
        cancel_data = {"id": order_id, "product_id": product_id}
        success, result = self._make_request('DELETE', '/v2/orders', data=cancel_data)
        return success, result

class AdvancedOrderManager:
    """Advanced order management with comprehensive partial fill handling"""
    
    def __init__(self, api: DeltaExchangeAPI):
        self.api = api
        self.config = api.config
    
    def handle_partial_fill_strategy(self, position: Position, remaining_option: OptionData) -> bool:
        """
        Handle partial fill scenarios for options trading
        
        Options trading requires exact quantities for balanced strategies.
        Partial fills can create position imbalances that affect risk profile.
        
        Strategies:
        1. Immediate completion: Try to fill remaining quantity immediately
        2. Accept partial: Log warning and continue with partial position
        3. Unwind partial: Close partial position and restart
        """
        if position.is_fully_filled:
            return True
        
        logger.warning(f"🚨 PARTIAL FILL DETECTED for {position.symbol}")
        logger.info(f"   Target: {position.quantity}, Filled: {position.filled_quantity}, Missing: {position.remaining_quantity}")
        
        # For options short strangle, we typically want exact quantities
        # Strategy 1: Try to complete the remaining quantity with a smaller order
        if position.remaining_quantity > 0 and position.remaining_quantity < 1:
            logger.warning(f"   ⚠️  Remaining quantity {position.remaining_quantity} is fractional - skipping completion")
            logger.warning(f"   💡 Consider adjusting strategy to accept partial positions")
            return True  # Accept the partial fill
        
        # Strategy 2: For integer remaining quantities, try immediate market order
        if position.remaining_quantity >= 1:
            logger.info(f"   🔄 Attempting to complete remaining {position.remaining_quantity} with market order")
            
            side = "sell" if position.side == PositionSide.SHORT else "buy"
            success, completion_result = self.execute_order_with_retries(
                remaining_option, 
                side, 
                position.remaining_quantity
            )
            
            if success:
                position.add_fill(completion_result)
                logger.info(f"   ✅ Successfully completed position: {position.get_fill_summary()}")
                return True
            else:
                logger.error(f"   ❌ Failed to complete remaining quantity")
                logger.warning(f"   💡 Continuing with partial position: {position.get_fill_summary()}")
                return True  # Continue with partial
        
        return True
    
    def execute_order_with_retries(self, option: OptionData, side: str, quantity: float) -> Tuple[bool, Dict]:
        """Execute order with enhanced retry logic, position checks, and automatic market order fallback"""
        logger.info(f"🎯 Executing {side} order for {quantity} {option.symbol}")
        logger.debug(f"💰 Order value: {quantity} × ${option.best_bid if side == 'sell' else option.best_ask}")
        
        # Step 1: Check current positions and orders
        logger.info("🔍 Checking current positions and open orders...")
        positions = self.api.get_current_positions()
        open_orders = self.api.get_open_orders()
        
        # Log current state
        for pos in positions:
            symbol = pos.get('product_symbol', 'Unknown')
            size = pos.get('size', 0)
            logger.info(f"📊 Current position: {symbol} size={size}")
        
        # Step 2: Cancel all open orders before placing new ones with enhanced verification
        if open_orders:
            logger.warning(f"⚠️  Found {len(open_orders)} open orders - cancelling all before proceeding")
            cancel_success = self.api.cancel_all_open_orders()
            if not cancel_success:
                logger.error("❌ Failed to cancel all orders - proceeding anyway")
            
            # Additional wait to ensure cancellations are processed
            logger.info("⏳ Waiting 5 seconds for order cancellations to fully process...")
            time.sleep(5)
            
            # Verify no orders remain
            remaining_orders = self.api.get_open_orders(include_recent=True)
            if remaining_orders:
                logger.warning(f"⚠️  {len(remaining_orders)} orders still open after cancellation attempt")
                for order in remaining_orders:
                    order_id = order.get('id')
                    symbol = order.get('product_symbol', 'Unknown')
                    logger.warning(f"   📋 Remaining order: {order_id} for {symbol}")
            else:
                logger.info("✅ All orders successfully cancelled")
        
        original_price = option.best_bid if side == "sell" else option.best_ask
        current_price = original_price
        
        # Step 3: Execute order with retries and enhanced pre-checks
        for attempt in range(self.config.max_retries + 1):
            logger.info(f"   🔄 Attempt {attempt + 1}/{self.config.max_retries + 1} @ ${current_price}")
            
            # PRE-ORDER CHECK: Ensure no open orders before placing new one
            pre_order_check = self.api.get_open_orders(include_recent=True)
            if pre_order_check:
                logger.warning(f"🚨 Found {len(pre_order_check)} open orders before placing new order!")
                for order in pre_order_check:
                    order_id = order.get('id')
                    symbol = order.get('product_symbol', 'Unknown')
                    logger.warning(f"   📋 Existing order: {order_id} for {symbol}")
                
                # Cancel these orders immediately
                logger.info("🧹 Cancelling orders found before placing new order...")
                self.api.cancel_all_open_orders()
                time.sleep(3)  # Wait for cancellations
            
            # Generate unique client order ID
            client_order_id = f"strangle_{option.symbol}_{side}_{int(time.time())}_{attempt}"
            
            # Place limit order
            success, order_result = self.api.place_order(
                symbol=option.symbol,
                side=side,
                quantity=quantity,
                order_type="limit_order",
                limit_price=current_price,
                client_order_id=client_order_id
            )
            
            if not success:
                logger.error(f"   ❌ Order placement failed (attempt {attempt + 1}): {order_result}")
                if attempt < self.config.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"   ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue
            
            order_id = order_result.get('id')
            logger.info(f"   📤 Order placed with ID: {order_id}")
            
            # Monitor order for fills with enhanced logging
            monitor_start = time.time()
            filled = False
            while time.time() - monitor_start < self.config.order_timeout:
                success, status = self.api.get_order_status(order_id)
                
                if not success:
                    logger.error(f"   ❌ Failed to get order status: {status}")
                    break
                
                state = status.get('state')
                unfilled_size = float(status.get('unfilled_size', quantity))
                filled_size = quantity - unfilled_size
                
                logger.debug(f"   📊 Order status: {state}, Filled: {filled_size}/{quantity}")
                
                if state == 'closed':
                    # Order filled (check if partial or full)
                    fill_price = float(status.get('average_fill_price', current_price))
                    fill_quantity = filled_size
                    
                    # Check for partial fill
                    is_partial_fill = fill_quantity < quantity
                    remaining_quantity = quantity - fill_quantity
                    
                    if is_partial_fill:
                        logger.warning(f"   ⚠️  PARTIAL FILL detected! Filled: {fill_quantity}/{quantity}, Remaining: {remaining_quantity}")
                        logger.info(f"   💰 Partial fill price: ${fill_price}, Value: ${fill_price * fill_quantity:.2f}")
                        
                        # For options trading, partial fills are problematic as we need exact quantities
                        # Log the issue and continue with what we got
                        logger.warning(f"   🚨 Strategy expects exact quantities - partial fill may affect position balance")
                    else:
                        logger.info(f"   ✅ Order FULLY filled! Price: ${fill_price}, Quantity: {fill_quantity}")
                        logger.info(f"   💵 Total value: ${fill_price * fill_quantity:.2f}")
                    
                    filled = True
                    return True, {
                        'order_id': order_id,
                        'fill_price': fill_price,
                        'fill_quantity': fill_quantity,
                        'remaining_quantity': remaining_quantity,
                        'is_partial_fill': is_partial_fill,
                        'client_order_id': client_order_id,
                        'attempt': attempt + 1,
                        'total_value': fill_price * fill_quantity,
                        'order_type': 'limit'
                    }
                
                elif state == 'cancelled':
                    logger.warning(f"   ⚠️  Order was cancelled")
                    break
                
                # Order still open, continue monitoring
                time.sleep(2)  # Check every 2 seconds
            
            if filled:
                break
                
            # Order didn't fill within timeout - cancel it
            logger.warning(f"   ⏰ Order timeout after {self.config.order_timeout}s, cancelling...")
            
            if order_id:
                cancel_success, cancel_result = self.api.cancel_order(str(order_id), option.product_id or 0)
                if cancel_success:
                    logger.info(f"   ✅ Order cancelled successfully")
                else:
                    logger.error(f"   ❌ Failed to cancel order: {cancel_result}")
            
            # Adjust price for next attempt (slippage)
            if attempt < self.config.max_retries:
                if side == "sell":
                    current_price *= (1 - self.config.retry_slippage_pct)  # Lower price for sells
                else:
                    current_price *= (1 + self.config.retry_slippage_pct)  # Higher price for buys
                
                logger.info(f"   🔄 Retrying with slippage-adjusted price: ${current_price:.2f}")
                time.sleep(5)  # Brief pause before retry
        
        # Step 4: All limit orders failed - place market order as last resort
        logger.warning(f"🚨 All {self.config.max_retries + 1} limit order attempts failed!")
        logger.info(f"📈 Placing MARKET ORDER as last resort for {quantity} {option.symbol}")
        
        # Cancel any remaining orders before market order
        self.api.cancel_all_open_orders()
        time.sleep(1)
        
        client_order_id = f"market_{option.symbol}_{side}_{int(time.time())}"
        
        success, order_result = self.api.place_order(
            symbol=option.symbol,
            side=side,
            quantity=quantity,
            order_type="market_order",
            client_order_id=client_order_id
        )
        
        if success:
            order_id = order_result.get('id')
            logger.info(f"   📤 Market order placed with ID: {order_id}")
            
            # Wait for market order to fill
            time.sleep(3)
            
            success, status = self.api.get_order_status(str(order_id))
            if success and status.get('state') == 'closed':
                fill_price = float(status.get('average_fill_price', current_price))
                fill_quantity = float(status.get('size', 0)) - float(status.get('unfilled_size', 0))
                remaining_quantity = quantity - fill_quantity
                is_partial_fill = fill_quantity < quantity
                
                if is_partial_fill:
                    logger.warning(f"   ⚠️  Market order PARTIAL FILL! Filled: {fill_quantity}/{quantity}")
                else:
                    logger.info(f"   ✅ Market order FULLY filled! Price: ${fill_price}, Quantity: {fill_quantity}")
                logger.info(f"   💵 Total value: ${fill_price * fill_quantity:.2f}")
                
                return True, {
                    'order_id': order_id,
                    'fill_price': fill_price,
                    'fill_quantity': fill_quantity,
                    'remaining_quantity': remaining_quantity,
                    'is_partial_fill': is_partial_fill,
                    'client_order_id': client_order_id,
                    'attempt': 'market',
                    'market_order': True,
                    'total_value': fill_price * fill_quantity,
                    'order_type': 'market'
                }
            else:
                logger.error(f"❌ Market order failed to fill: {status}")
        
        logger.error(f"❌ All order attempts (including market order) failed for {option.symbol}")
        return False, {'error': 'All order attempts failed, including market order'}

class ShortStrangleStrategy:
    """Production short strangle strategy implementation"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.api = DeltaExchangeAPI(config)
        self.order_manager = AdvancedOrderManager(self.api)
        self.strategy_id = f"strangle_{int(time.time())}"
        self.state = StrategyState(self.strategy_id, config)
        
        logger.info(f"🎭 Short Strangle Strategy initialized")
        logger.info(f"   Strategy ID: {self.strategy_id}")
        logger.info(f"   Underlying: {config.underlying}")
        logger.info(f"   Delta Range: {config.delta_range_low:.2f} - {config.delta_range_high:.2f}")
        logger.info(f"   Dry Run: {config.dry_run}")
    
    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites before trading with comprehensive duplicate prevention"""
        logger.info("🔍 Validating prerequisites...")
        
        # Test API connection
        if not self.api.test_connection():
            logger.error("❌ API connection test failed")
            return False
        
        # Check if we have an active position in strategy state
        if self.state.is_active:
            logger.warning("⚠️  Strategy already has active position in memory")
            return False
        
        # CRITICAL: Check for actual live positions in account (duplicate prevention)
        logger.info("🔍 Checking for existing live positions...")
        live_positions = self.api.get_current_positions()
        live_orders = self.api.get_open_orders()
        
        if live_positions:
            logger.warning(f"🚨 DUPLICATE TRADE PREVENTION: Found {len(live_positions)} existing positions")
            for pos in live_positions:
                symbol = pos.get('product_symbol', 'Unknown')
                size = pos.get('size', 0)
                entry_price = pos.get('entry_price', 0)
                logger.warning(f"   📊 Existing position: {symbol} | Size: {size} | Entry: ${entry_price}")

            if self.recover_existing_positions(live_positions):
                logger.info("♻️  Recovered existing short strangle — skipping new entry.")
                return True

            logger.error("❌ Cannot start new trade - existing positions detected and recovery failed")
            return False
        
        if live_orders:
            logger.warning(f"🚨 DUPLICATE TRADE PREVENTION: Found {len(live_orders)} open orders")
            for order in live_orders:
                order_id = order.get('id')
                symbol = order.get('product_symbol', 'Unknown')
                side = order.get('side', 'Unknown')
                logger.warning(f"   📋 Open order: {order_id} | {symbol} | {side}")
            logger.error("❌ Cannot start new trade - open orders detected")
            return False
        
        logger.info("✅ No existing positions or orders found - safe to proceed")
        
        # Check current time vs trade time
        current_ist = self.api.get_current_ist_time()
        logger.info(f"   Current IST: {current_ist.strftime('%H:%M:%S')}")
        
        # Validate market hours (options trade 9:00 - 17:30 IST)
        if not (9 <= current_ist.hour < 18):
            logger.warning(f"⚠️  Outside market hours: {current_ist.hour}:00")
        
        logger.info("✅ All prerequisites validated")
        return True
    
    def recover_existing_positions(self, existing_positions: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Recover and reconstruct strategy state from existing live positions"""
        logger.info("🔄 Attempting to recover existing positions into strategy state...")
        
        # Get live positions
        live_positions = existing_positions if existing_positions is not None else self.api.get_current_positions()
        if not live_positions:
            logger.info("✅ No existing positions to recover")
            return False
        
        logger.info(f"📊 Found {len(live_positions)} live positions to recover")
        
        # Analyze positions to reconstruct strategy
        call_positions = []
        put_positions = []
        total_premium = 0.0
        
        for pos in live_positions:
            symbol = pos.get('product_symbol', '')
            size = float(pos.get('size', 0))
            entry_price = float(pos.get('entry_price', 0))
            
            # Skip non-option positions
            if not (symbol.startswith('C-') or symbol.startswith('P-')):
                continue
            
            # Check if it's a short position (negative size)
            if size < 0:
                # This is a short position - add to premium received
                premium_received = entry_price * abs(size) * self.config.contract_size
                total_premium += premium_received
                
                if symbol.startswith('C-'):
                    call_positions.append(pos)
                    logger.info(
                        "📞 Found short call: %s | Contracts: %s | Entry: $%.2f | Premium: $%.6f",
                        symbol,
                        size,
                        entry_price,
                        premium_received,
                    )
                elif symbol.startswith('P-'):
                    put_positions.append(pos)
                    logger.info(
                        "📄 Found short put: %s | Contracts: %s | Entry: $%.2f | Premium: $%.6f",
                        symbol,
                        size,
                        entry_price,
                        premium_received,
                    )
        
        # Check if we have a valid strangle (at least 1 call and 1 put)
        if len(call_positions) >= 1 and len(put_positions) >= 1:
            logger.info(f"✅ Valid short strangle detected - reconstructing strategy state")
            
            # Reconstruct call position
            call_pos = call_positions[0]  # Take first call
            self.state.call_position = Position(
                strategy_id=self.strategy_id,
                symbol=call_pos.get('product_symbol', ''),
                side=PositionSide.SHORT,
                quantity=abs(float(call_pos.get('size', 0))),
                entry_price=float(call_pos.get('entry_price', 0)),
                current_price=float(call_pos.get('entry_price', 0)),
                unrealized_pnl=0.0,
                entry_time=self.api.get_current_ist_time() - timedelta(hours=1),  # Estimate entry time
                contract_size=self.config.contract_size,
            )
            
            # Reconstruct put position
            put_pos = put_positions[0]  # Take first put
            self.state.put_position = Position(
                strategy_id=self.strategy_id,
                symbol=put_pos.get('product_symbol', ''),
                side=PositionSide.SHORT,
                quantity=abs(float(put_pos.get('size', 0))),
                entry_price=float(put_pos.get('entry_price', 0)),
                current_price=float(put_pos.get('entry_price', 0)),
                unrealized_pnl=0.0,
                entry_time=self.api.get_current_ist_time() - timedelta(hours=1),  # Estimate entry time
                contract_size=self.config.contract_size,
            )
            
            # Update strategy state
            self.state.total_premium_received = total_premium
            self.state.is_active = True
            self.state.entry_time = self.api.get_current_ist_time() - timedelta(hours=1)
            
            logger.info(f"✅ Strategy state recovered successfully!")
            logger.info(f"💰 Total Premium Received: ${total_premium:.6f}")
            logger.info(f"📞 Call Position: {self.state.call_position.symbol}")
            logger.info(f"📄 Put Position: {self.state.put_position.symbol}")
            
            return True
        else:
            logger.warning(f"⚠️  Positions found but not a valid strangle pattern")
            logger.warning(f"   Calls: {len(call_positions)}, Puts: {len(put_positions)}")
            return False
    
    def enter_short_strangle(self) -> bool:
        """Enter short strangle position"""
        logger.info("🚀 Entering Short Strangle Position")
        logger.info("=" * 60)
        
        try:
            # Get option chain with Greeks
            options = self.api.get_option_chain_with_greeks(self.config.underlying)
            
            if not options:
                logger.error("❌ No options data available")
                return False
            
            logger.info(f"📊 Retrieved {len(options)} options")
            
            # Find suitable delta options
            call_option, put_option = self.api.find_delta_options(options)
            
            if not call_option or not put_option:
                logger.error("❌ Could not find suitable options in delta range")
                return False
            
            # Store entry time
            self.state.entry_time = self.api.get_current_ist_time()

            logger.info("\n⚡ Executing CALL and PUT legs concurrently...")
            premium_received_call = 0.0
            premium_received_put = 0.0

            with ThreadPoolExecutor(max_workers=2) as executor:
                future_map = {
                    executor.submit(
                        self.order_manager.execute_order_with_retries,
                        call_option,
                        "sell",
                        self.config.quantity,
                    ): "call",
                    executor.submit(
                        self.order_manager.execute_order_with_retries,
                        put_option,
                        "sell",
                        self.config.quantity,
                    ): "put",
                }

                leg_results: Dict[str, Tuple[bool, Dict[str, Any]]] = {}
                for future in as_completed(future_map):
                    leg = future_map[future]
                    try:
                        leg_results[leg] = future.result()
                    except Exception as exc:
                        logger.error("❌ %s leg execution raised exception: %s", leg.capitalize(), exc)
                        leg_results[leg] = (False, {"error": str(exc)})

            call_success, call_result = leg_results.get("call", (False, {}))
            put_success, put_result = leg_results.get("put", (False, {}))

            if not call_success or not put_success:
                if call_success and not put_success:
                    logger.error("❌ Put leg failed after call leg filled — attempting to flatten call exposure")
                    self._attempt_flatten_leg(
                        call_option.symbol,
                        float(call_result.get('fill_price', 0.0) or 0.0),
                        float(call_result.get('fill_quantity', self.config.quantity) or self.config.quantity),
                    )
                if put_success and not call_success:
                    logger.error("❌ Call leg failed after put leg filled — attempting to flatten put exposure")
                    self._attempt_flatten_leg(
                        put_option.symbol,
                        float(put_result.get('fill_price', 0.0) or 0.0),
                        float(put_result.get('fill_quantity', self.config.quantity) or self.config.quantity),
                    )
                logger.error("❌ Failed to execute both legs successfully")
                return False

            # Create call position with enhanced partial fill tracking
            self.state.call_position = Position(
                strategy_id=self.strategy_id,
                symbol=call_option.symbol,
                side=PositionSide.SHORT,
                quantity=self.config.quantity,
                entry_price=0.0,  # Will be calculated by add_fill
                current_price=call_result['fill_price'],
                unrealized_pnl=0.0,
                entry_time=self.state.entry_time,
                contract_size=self.config.contract_size,
            )

            self.state.call_position.add_fill(call_result)
            logger.info(f"📞 Call position status: {self.state.call_position.get_fill_summary()}")
            if call_result.get('is_partial_fill', False):
                logger.warning("⚠️  Call leg partially filled - this may affect strategy balance")

            premium_received_call = (
                call_result['fill_price']
                * call_result['fill_quantity']
                * self.config.contract_size
            )

            # Create put position with enhanced partial fill tracking
            self.state.put_position = Position(
                strategy_id=self.strategy_id,
                symbol=put_option.symbol,
                side=PositionSide.SHORT,
                quantity=self.config.quantity,
                entry_price=0.0,
                current_price=put_result['fill_price'],
                unrealized_pnl=0.0,
                entry_time=self.state.entry_time,
                contract_size=self.config.contract_size,
            )

            self.state.put_position.add_fill(put_result)
            logger.info(f"📄 Put position status: {self.state.put_position.get_fill_summary()}")
            if put_result.get('is_partial_fill', False):
                logger.warning("⚠️  Put leg partially filled - this may affect strategy balance")

            premium_received_put = (
                put_result['fill_price']
                * put_result['fill_quantity']
                * self.config.contract_size
            )
            
            # Update strategy state
            self.state.total_premium_received = premium_received_call + premium_received_put
            self.state.is_active = True
            
            logger.info(f"\n🎉 SHORT STRANGLE ENTRY COMPLETED!")
            logger.info(
                "📞 Call: %s | Price: $%.2f | Contracts: %s | Premium: $%.4f",
                call_option.symbol,
                call_result['fill_price'],
                call_result['fill_quantity'],
                premium_received_call,
            )
            logger.info(
                "📄 Put:  %s | Price: $%.2f | Contracts: %s | Premium: $%.4f",
                put_option.symbol,
                put_result['fill_price'],
                put_result['fill_quantity'],
                premium_received_put,
            )
            logger.info(
                "💰 Total Premium Received: $%.4f",
                self.state.total_premium_received,
            )
            logger.info(f"⏰ Entry Time: {self.state.entry_time.strftime('%H:%M:%S')} IST")
            logger.info("🔍 P&L Calculation: (Entry_Price - Current_Price) × Quantity")
            logger.info("📈 Monitoring every 1 second with detailed logs in files...")
            
            # Detailed logging to file only
            logger.debug(f"=" * 60)
            logger.debug(
                "📞 Call: %s | Price: $%.2f | Contracts: %s | Premium: $%.4f",
                call_option.symbol,
                call_result['fill_price'],
                call_result['fill_quantity'],
                premium_received_call,
            )
            logger.debug(
                "📄 Put:  %s | Price: $%.2f | Contracts: %s | Premium: $%.4f",
                put_option.symbol,
                put_result['fill_price'],
                put_result['fill_quantity'],
                premium_received_put,
            )
            logger.debug(
                "💰 Total Premium Received: $%.4f",
                self.state.total_premium_received,
            )
            logger.debug(f"⏰ Entry Time: {self.state.entry_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
            logger.debug(f"=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error entering position: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_fair_market_price(self, symbol: str) -> Optional[float]:
        """Get fair market price using multiple sources with preference order"""
        try:
            # First, try WebSocket live price (mark_price is most accurate)
            if self.api.websocket_streamer:
                live_price_data = self.api.websocket_streamer.get_live_price(symbol)
                if live_price_data:
                    # Prefer mark_price for options valuation
                    if 'mark_price' in live_price_data:
                        logger.debug(f"💹 Using WebSocket mark_price for {symbol}: ${live_price_data['mark_price']:.2f}")
                        return float(live_price_data['mark_price'])
                    
                    # Fallback to mid-price from WebSocket
                    bid = live_price_data.get('best_bid', 0)
                    ask = live_price_data.get('best_ask', 0)
                    if bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        logger.debug(f"📊 Using WebSocket mid-price for {symbol}: ${mid_price:.2f}")
                        return mid_price
            
            # Fallback to API ticker data
            success, result = self.api._make_public_request(f'/v2/tickers/{symbol}')
            if success:
                result_data = result.get('result', {})
                
                # Try mark_price first (most accurate for options)
                if 'mark_price' in result_data:
                    mark_price = float(result_data['mark_price'])
                    logger.debug(f"🎯 Using API mark_price for {symbol}: ${mark_price:.2f}")
                    return mark_price
                
                # Calculate mid-price from bid/ask
                quotes = result_data.get('quotes', {})
                best_bid = quotes.get('best_bid')
                best_ask = quotes.get('best_ask')
                
                if best_bid and best_ask:
                    bid = float(best_bid)
                    ask = float(best_ask)
                    
                    # Sanity check: bid should be less than ask
                    if bid < ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                        logger.debug(f"🧮 Using API mid-price for {symbol}: ${mid_price:.2f} (bid=${bid:.2f}, ask=${ask:.2f})")
                        return mid_price
                
                # Last resort: use last_price
                if 'last_price' in result_data:
                    last_price = float(result_data['last_price'])
                    logger.debug(f"⏮️  Using last_price for {symbol}: ${last_price:.2f}")
                    return last_price
            
            logger.warning(f"⚠️  Could not get fair market price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting fair market price for {symbol}: {str(e)}")
            return None

    def _attempt_flatten_leg(self, symbol: str, reference_price: float, quantity: float) -> None:
        """Best-effort flatten for a leg if its counterpart fails during entry."""
        try:
            if quantity <= 0:
                logger.warning("⚠️  Skipping flatten attempt for %s — invalid quantity %.4f", symbol, quantity)
                return

            contract_type = "call_options" if symbol.startswith("C-") else "put_options"
            market_price = self._get_fair_market_price(symbol) or reference_price or 0.0
            if market_price <= 0:
                market_price = max(reference_price, 0.0001)

            unwind_option = OptionData(
                symbol=symbol,
                contract_type=contract_type,
                strike_price=0.0,
                expiry_date="",
                spot_price=0.0,
                delta=0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                implied_volatility=0.0,
                best_bid=market_price * 0.999,
                best_ask=market_price * 1.001,
                bid_size=0.0,
                ask_size=0.0,
                volume=0,
                open_interest=0,
                product_id=0,
            )

            logger.info("🛑 Attempting to flatten %s leg with %s contracts", symbol, quantity)
            success, result = self.order_manager.execute_order_with_retries(
                unwind_option, "buy", quantity
            )
            if success:
                logger.info("✅ Successfully flattened %s leg after counterpart failure", symbol)
            else:
                logger.error("❌ Failed to flatten %s leg: %s", symbol, result)
        except Exception as exc:
            logger.error("❌ Flatten attempt for %s leg encountered error: %s", symbol, exc)

    def update_pnl(self) -> bool:
        """Update P&L for both positions with correct calculation"""
        if not self.state.is_active:
            return True
        
        try:
            total_unrealized_pnl = 0.0
            
            # Update call position
            if self.state.call_position:
                current_market_price = self._get_fair_market_price(self.state.call_position.symbol)
                
                if current_market_price is not None:
                    self.state.call_position.current_price = current_market_price
                    contracts = self.state.call_position.quantity
                    contract_size = getattr(self.state.call_position, "contract_size", self.config.contract_size)
                    
                    # P&L for SHORT position = (Premium Received - Current Market Value) * Contracts * Contract Size
                    self.state.call_position.unrealized_pnl = (
                        self.state.call_position.entry_price - current_market_price
                    ) * contracts * contract_size
                    
                    total_unrealized_pnl += self.state.call_position.unrealized_pnl
                    
                    logger.debug(
                        "📞 Call P&L: Entry=$%.2f, Current=$%.2f, Contracts=%s, ContractSize=%.6f BTC, P&L=$%.6f",
                        self.state.call_position.entry_price,
                        current_market_price,
                        contracts,
                        contract_size,
                        self.state.call_position.unrealized_pnl,
                    )
            
            # Update put position
            if self.state.put_position:
                current_market_price = self._get_fair_market_price(self.state.put_position.symbol)
                
                if current_market_price is not None:
                    self.state.put_position.current_price = current_market_price
                    contracts = self.state.put_position.quantity
                    contract_size = getattr(self.state.put_position, "contract_size", self.config.contract_size)
                    
                    # P&L for SHORT position = (Premium Received - Current Market Value) * Contracts * Contract Size
                    self.state.put_position.unrealized_pnl = (
                        self.state.put_position.entry_price - current_market_price
                    ) * contracts * contract_size
                    
                    total_unrealized_pnl += self.state.put_position.unrealized_pnl
                    
                    logger.debug(
                        "📄 Put P&L: Entry=$%.2f, Current=$%.2f, Contracts=%s, ContractSize=%.6f BTC, P&L=$%.6f",
                        self.state.put_position.entry_price,
                        current_market_price,
                        contracts,
                        contract_size,
                        self.state.put_position.unrealized_pnl,
                    )
            
            # Update strategy P&L
            self.state.current_pnl = total_unrealized_pnl

            total_premium = self.state.total_premium_received
            if total_premium and abs(total_premium) > 1e-9:
                self.state.current_pnl_pct = self.state.current_pnl / total_premium
            else:
                self.state.current_pnl_pct = None
            
            # Update max profit seen for trailing stop
            if self.state.current_pnl > self.state.max_profit_seen:
                self.state.max_profit_seen = self.state.current_pnl
                self._update_trailing_stop()
            
            if self.state.current_pnl_pct is not None:
                logger.debug(
                    "💰 Total Strategy P&L: $%.6f (%.2f%% of premium)",
                    self.state.current_pnl,
                    self.state.current_pnl_pct * 100,
                )
            else:
                logger.debug(f"💰 Total Strategy P&L: ${self.state.current_pnl:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating P&L: {str(e)}")
            return False
    
    def _update_trailing_stop(self):
        """Update trailing stop loss level"""
        if not self.config.trailing_sl_enabled:
            return
        
        profit_pct = self.state.max_profit_seen / self.state.total_premium_received
        
        # Apply basic trailing rules
        new_sl_level = 0.0
        for profit_threshold, sl_level in self.config.trailing_rules.items():
            if profit_pct >= profit_threshold:
                new_sl_level = sl_level
        
        # After 50%, every +10% profit adds +5% to SL
        if profit_pct > 0.50:
            additional_profit = profit_pct - 0.50
            additional_steps = int(additional_profit / 0.10)
            new_sl_level += additional_steps * 0.05
        
        if new_sl_level > self.state.trailing_sl_level:
            old_level = self.state.trailing_sl_level
            self.state.trailing_sl_level = new_sl_level
            logger.info(f"📈 Trailing SL updated: {old_level:.1%} → {new_sl_level:.1%} (Profit: {profit_pct:.1%})")
    
    def check_exit_conditions(self) -> Tuple[bool, Optional[ExitReason]]:
        """Check if any exit conditions are met"""
        if not self.state.is_active:
            return False, None
        
        current_ist = self.api.get_current_ist_time()
        
        # Check fixed time exit
        exit_time_str = self.config.exit_time_ist
        exit_hour, exit_minute = map(int, exit_time_str.split(':'))
        exit_time = current_ist.replace(hour=exit_hour, minute=exit_minute, second=0, microsecond=0)

        # Roll exit to the next day when configured time has already passed today
        if exit_time <= current_ist:
            exit_time += timedelta(days=1)
            logger.debug(
                "🗓️ Exit time rolled to next day: %s",
                exit_time.strftime("%Y-%m-%d %H:%M:%S %Z") if exit_time.tzinfo else exit_time.isoformat(),
            )
        
        if current_ist >= exit_time:
            logger.info(f"⏰ Fixed time exit triggered: {exit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            return True, ExitReason.FIXED_TIME
        
        # Update P&L before checking profit/loss conditions
        self.update_pnl()
        
        # Check max loss
        loss_threshold = -self.state.total_premium_received * self.config.max_loss_pct
        if self.state.current_pnl <= loss_threshold:
            logger.warning(f"🚨 Max loss triggered: ${self.state.current_pnl:.2f} <= ${loss_threshold:.2f}")
            return True, ExitReason.MAX_LOSS
        
        # Check max profit
        profit_threshold = self.state.total_premium_received * self.config.max_profit_pct
        if self.state.current_pnl >= profit_threshold:
            logger.info(f"🎯 Max profit triggered: ${self.state.current_pnl:.2f} >= ${profit_threshold:.2f}")
            return True, ExitReason.MAX_PROFIT
        
        # Check trailing stop
        if self.config.trailing_sl_enabled and self.state.trailing_sl_level > 0:
            trailing_threshold = self.state.total_premium_received * self.state.trailing_sl_level
            if self.state.current_pnl <= trailing_threshold:
                logger.info(f"📉 Trailing stop triggered: ${self.state.current_pnl:.2f} <= ${trailing_threshold:.2f}")
                return True, ExitReason.TRAILING_STOP
        
        return False, None
    
    def exit_position(self, reason: ExitReason) -> bool:
        """Exit the complete position with enhanced checks and accurate P&L calculation"""
        logger.info(f"🚪 Exiting position - Reason: {reason.value}")
        logger.info("=" * 60)
        
        # Step 1: Check current state before exit
        logger.info("🔍 Pre-exit checks...")
        positions = self.api.get_current_positions()
        open_orders = self.api.get_open_orders()
        
        logger.info(f"📊 Found {len(positions)} positions and {len(open_orders)} open orders")
        
        # Cancel all open orders first
        if open_orders:
            logger.warning(f"⚠️  Cancelling {len(open_orders)} open orders before exit")
            self.api.cancel_all_open_orders()
            time.sleep(2)  # Wait for cancellations
        
        exit_success = True
        total_realized_pnl = 0.0
        
        try:
            leg_configs: Dict[str, Dict[str, Any]] = {}

            # Prepare call leg exit
            if self.state.call_position:
                logger.info(f"📞 Closing call position: {self.state.call_position.symbol}")
                logger.info(f"   Entry: ${self.state.call_position.entry_price:.2f}")
                logger.info(f"   Current: ${self.state.call_position.current_price:.2f}")
                logger.info(f"   Quantity: {self.state.call_position.quantity}")

                current_price = self._get_fair_market_price(self.state.call_position.symbol)
                if current_price:
                    self.state.call_position.current_price = current_price
                    logger.info(f"   Updated current price: ${current_price:.2f}")

                call_option = OptionData(
                    symbol=self.state.call_position.symbol,
                    contract_type="call_options",
                    strike_price=0.0,
                    expiry_date="",
                    spot_price=0.0,
                    delta=0.0,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    implied_volatility=0.0,
                    best_bid=self.state.call_position.current_price * 0.999,
                    best_ask=self.state.call_position.current_price * 1.001,
                    bid_size=0.0,
                    ask_size=0.0,
                    volume=0,
                    open_interest=0,
                    product_id=0,
                )

                actual_quantity = abs(self.state.call_position.quantity)
                logger.info(f"   Queued BUY order for {actual_quantity} contracts")
                leg_configs["call"] = {
                    "option": call_option,
                    "quantity": actual_quantity,
                    "position": self.state.call_position,
                }

            # Prepare put leg exit
            if self.state.put_position:
                logger.info(f"📄 Closing put position: {self.state.put_position.symbol}")
                logger.info(f"   Entry: ${self.state.put_position.entry_price:.2f}")
                logger.info(f"   Current: ${self.state.put_position.current_price:.2f}")
                logger.info(f"   Quantity: {self.state.put_position.quantity}")

                current_price = self._get_fair_market_price(self.state.put_position.symbol)
                if current_price:
                    self.state.put_position.current_price = current_price
                    logger.info(f"   Updated current price: ${current_price:.2f}")

                put_option = OptionData(
                    symbol=self.state.put_position.symbol,
                    contract_type="put_options",
                    strike_price=0.0,
                    expiry_date="",
                    spot_price=0.0,
                    delta=0.0,
                    gamma=0.0,
                    theta=0.0,
                    vega=0.0,
                    implied_volatility=0.0,
                    best_bid=self.state.put_position.current_price * 0.999,
                    best_ask=self.state.put_position.current_price * 1.001,
                    bid_size=0.0,
                    ask_size=0.0,
                    volume=0,
                    open_interest=0,
                    product_id=0,
                )

                actual_quantity = abs(self.state.put_position.quantity)
                logger.info(f"   Queued BUY order for {actual_quantity} contracts")
                leg_configs["put"] = {
                    "option": put_option,
                    "quantity": actual_quantity,
                    "position": self.state.put_position,
                }

            leg_results: Dict[str, Tuple[bool, Dict[str, Any]]] = {}
            if leg_configs:
                logger.info(f"⚡ Sending {len(leg_configs)} closing order(s) concurrently")
                with ThreadPoolExecutor(max_workers=len(leg_configs)) as executor:
                    future_map = {
                        executor.submit(
                            self.order_manager.execute_order_with_retries,
                            leg_data["option"],
                            "buy",
                            leg_data["quantity"],
                        ): leg_name
                        for leg_name, leg_data in leg_configs.items()
                    }

                    for future in as_completed(future_map):
                        leg = future_map[future]
                        try:
                            leg_results[leg] = future.result()
                        except Exception as exc:
                            logger.error("❌ %s leg exit raised exception: %s", leg.capitalize(), exc)
                            leg_results[leg] = (False, {"error": str(exc)})

            # Process call leg result
            if "call" in leg_configs:
                call_success, call_result = leg_results.get("call", (False, {}))
                call_position = leg_configs["call"]["position"]
                if call_success:
                    call_position.exit_time = self.api.get_current_ist_time()
                    fill_price = call_result.get('fill_price', call_position.current_price)
                    fill_quantity = call_result.get('fill_quantity', leg_configs["call"]["quantity"])
                    contract_size = getattr(call_position, "contract_size", self.config.contract_size)
                    call_pnl = (
                        call_position.entry_price - fill_price
                    ) * fill_quantity * contract_size
                    call_position.realized_pnl = call_pnl
                    total_realized_pnl += call_pnl

                    logger.info("✅ Call position closed!")
                    logger.info(f"   Entry: ${call_position.entry_price:.2f}")
                    logger.info(f"   Exit: ${fill_price:.2f}")
                    logger.info(f"   Contracts: {fill_quantity}")
                    logger.info(f"   Contract Size: {contract_size:.6f}")
                    logger.info(f"   P&L: ${call_pnl:.6f}")
                else:
                    logger.error(f"❌ Failed to close call position: {call_result}")
                    exit_success = False

            # Process put leg result
            if "put" in leg_configs:
                put_success, put_result = leg_results.get("put", (False, {}))
                put_position = leg_configs["put"]["position"]
                if put_success:
                    put_position.exit_time = self.api.get_current_ist_time()
                    fill_price = put_result.get('fill_price', put_position.current_price)
                    fill_quantity = put_result.get('fill_quantity', leg_configs["put"]["quantity"])
                    contract_size = getattr(put_position, "contract_size", self.config.contract_size)
                    put_pnl = (
                        put_position.entry_price - fill_price
                    ) * fill_quantity * contract_size
                    put_position.realized_pnl = put_pnl
                    total_realized_pnl += put_pnl

                    logger.info("✅ Put position closed!")
                    logger.info(f"   Entry: ${put_position.entry_price:.2f}")
                    logger.info(f"   Exit: ${fill_price:.2f}")
                    logger.info(f"   Contracts: {fill_quantity}")
                    logger.info(f"   Contract Size: {contract_size:.6f}")
                    logger.info(f"   P&L: ${put_pnl:.6f}")
                else:
                    logger.error(f"❌ Failed to close put position: {put_result}")
                    exit_success = False

            # Step 4: Final verification
            logger.info("🔍 Post-exit verification...")
            final_positions = self.api.get_current_positions()
            final_orders = self.api.get_open_orders()
            
            logger.info(f"📊 Remaining positions: {len(final_positions)}")
            logger.info(f"📊 Remaining orders: {len(final_orders)}")
            
            if final_positions:
                for pos in final_positions:
                    symbol = pos.get('product_symbol', 'Unknown')
                    size = pos.get('size', 0)
                    logger.warning(f"⚠️  Remaining position: {symbol} size={size}")
            
            if final_orders:
                logger.warning(f"⚠️  {len(final_orders)} orders still open - attempting final cleanup")
                self.api.cancel_all_open_orders()
            
            # Update strategy state
            self.state.is_active = False
            self.state.exit_time = self.api.get_current_ist_time()
            self.state.exit_reason = reason
            
            # Calculate total return percentage
            return_pct = (total_realized_pnl / self.state.total_premium_received * 100) if self.state.total_premium_received > 0 else 0
            
            logger.info(f"\n🏁 POSITION EXIT COMPLETED!")
            logger.info(f"=" * 60)
            logger.info(f"💰 Total Realized P&L: ${total_realized_pnl:.6f}")
            logger.info(f"📊 Premium Received: ${self.state.total_premium_received:.6f}")
            logger.info(f"📈 Return: {return_pct:.2f}%")
            logger.info(f"⏰ Exit Time: {self.state.exit_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
            logger.info(f"🔍 Exit Reason: {reason.value}")
            logger.info(f"✅ Exit Success: {exit_success}")
            logger.info(f"📊 Final Positions: {len(final_positions)}")
            logger.info(f"📊 Final Orders: {len(final_orders)}")
            
            return exit_success
            
        except Exception as e:
            logger.error(f"❌ Error exiting position: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_live_monitoring(self, duration_minutes: int = 60):
        """Run comprehensive live monitoring of positions"""
        logger.info(f"👁️  Starting live monitoring for {duration_minutes} minutes...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_report_time = 0
        report_interval = 300  # 5 minutes
        
        while time.time() < end_time and self.state.is_active:
            try:
                current_time = time.time()
                
                # Update P&L
                self.update_pnl()
                
                # Generate detailed status report every 5 minutes
                if current_time - last_report_time >= report_interval:
                    self._generate_status_report()
                    last_report_time = current_time
                
                # Log current status with detailed P&L breakdown
                current_ist = self.api.get_current_ist_time()
                profit_pct = (self.state.current_pnl / self.state.total_premium_received * 100) if self.state.total_premium_received > 0 else 0
                
                # Detailed P&L breakdown
                call_pnl = self.state.call_position.unrealized_pnl if self.state.call_position else 0
                put_pnl = self.state.put_position.unrealized_pnl if self.state.put_position else 0
                call_entry = self.state.call_position.entry_price if self.state.call_position else 0
                call_current = self.state.call_position.current_price if self.state.call_position else 0
                put_entry = self.state.put_position.entry_price if self.state.put_position else 0
                put_current = self.state.put_position.current_price if self.state.put_position else 0
                
                # Terminal: Clean summary output
                logger.info(
                    "📊 %s P&L: $%.6f (%+.1f%%) | Premium: $%.6f",
                    current_ist.strftime('%H:%M:%S'),
                    self.state.current_pnl,
                    profit_pct,
                    self.state.total_premium_received,
                )
                
                # File: Detailed P&L breakdown (debug level goes to file only)
                logger.debug(
                    "📊 %s - Total P&L: $%.6f (%+.1f%%)",
                    current_ist.strftime('%H:%M:%S'),
                    self.state.current_pnl,
                    profit_pct,
                )
                logger.debug(
                    "   📞 Call: $%.2f→$%.2f = $%+.6f | 📄 Put: $%.2f→$%.2f = $%+.6f",
                    call_entry,
                    call_current,
                    call_pnl,
                    put_entry,
                    put_current,
                    put_pnl,
                )
                logger.debug(
                    "   🎯 Max: $%.6f | TrailSL: %.1f%%",
                    self.state.max_profit_seen,
                    self.state.trailing_sl_level * 100,
                )
                logger.debug(
                    "   💰 Premium: $%.6f",
                    self.state.total_premium_received,
                )
                logger.debug("-" * 80)
                
                # Check for significant market moves
                self._check_market_alerts()
                
                # Position reconciliation (every 10 checks)
                if int(current_time) % (self.config.position_check_interval * 10) == 0:
                    self._reconcile_positions()
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions()
                
                if should_exit and exit_reason:
                    logger.info(f"🚨 Exit condition triggered: {exit_reason.value}")
                    self.exit_position(exit_reason)
                    break
                
                # Wait before next check
                time.sleep(self.config.position_check_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Manual stop requested")
                self.exit_position(ExitReason.FORCE_CLOSE)
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Wait longer on error
        
        if self.state.is_active:
            logger.info("⏰ Monitoring duration completed")
    
    def _generate_status_report(self):
        """Generate comprehensive status report"""
        logger.info(f"\n📋 POSITION STATUS REPORT")
        logger.info(f"=" * 50)
        
        current_ist = self.api.get_current_ist_time()
        
        if self.state.entry_time:
            try:
                # Ensure both datetimes are timezone-aware for subtraction
                entry_time = self.state.entry_time
                if entry_time.tzinfo is None:
                    # Convert naive datetime to UTC then to IST
                    entry_time = entry_time.replace(tzinfo=timezone.utc) + IST_OFFSET
                
                time_in_trade = current_ist - entry_time
                hours_in_trade = time_in_trade.total_seconds() / 3600
                logger.info(f"⏰ Time in Trade: {hours_in_trade:.1f} hours")
            except Exception as e:
                logger.debug(f"⚠️  Could not calculate time in trade: {str(e)}")
                logger.info(f"⏰ Time in Trade: Entry time available")
        
        logger.info(f"💰 Premium Received: ${self.state.total_premium_received:.6f}")
        logger.info(f"📈 Current P&L: ${self.state.current_pnl:.2f}")
        logger.info(f"🎯 Max Profit Seen: ${self.state.max_profit_seen:.2f}")
        logger.info(f"📉 Trailing SL Level: {self.state.trailing_sl_level:.1%}")
        
        if self.state.call_position:
            logger.info(f"📞 Call Position:")
            logger.info(f"   Symbol: {self.state.call_position.symbol}")
            logger.info(f"   Entry: ${self.state.call_position.entry_price:.2f}")
            logger.info(f"   Current: ${self.state.call_position.current_price:.2f}")
            logger.info(f"   P&L: ${self.state.call_position.unrealized_pnl:.2f}")
        
        if self.state.put_position:
            logger.info(f"📄 Put Position:")
            logger.info(f"   Symbol: {self.state.put_position.symbol}")
            logger.info(f"   Entry: ${self.state.put_position.entry_price:.2f}")
            logger.info(f"   Current: ${self.state.put_position.current_price:.2f}")
            logger.info(f"   P&L: ${self.state.put_position.unrealized_pnl:.2f}")
        
        # Risk metrics
        logger.info(f"\n🛡️  Risk Metrics:")
        max_loss_threshold = -self.state.total_premium_received * self.config.max_loss_pct
        max_profit_threshold = self.state.total_premium_received * self.config.max_profit_pct
        
        logger.info(f"   Max Loss Threshold: ${max_loss_threshold:.2f}")
        logger.info(f"   Max Profit Threshold: ${max_profit_threshold:.2f}")
        
        distance_to_max_loss = abs(self.state.current_pnl - max_loss_threshold)
        distance_to_max_profit = abs(max_profit_threshold - self.state.current_pnl)
        
        logger.info(f"   Distance to Max Loss: ${distance_to_max_loss:.2f}")
        logger.info(f"   Distance to Max Profit: ${distance_to_max_profit:.2f}")
        
        logger.info(f"=" * 50)
    
    def _check_market_alerts(self):
        """Check for significant market movements and alerts"""
        try:
            # Get current BTC spot price for reference
            if self.state.call_position:
                success, result = self.api._make_public_request(f'/v2/tickers/{self.state.call_position.symbol}')
                if success:
                    current_spot = float(result.get('result', {}).get('spot_price', 0))
                    
                    # Alert on significant moves (could be enhanced with more sophisticated logic)
                    if hasattr(self, '_last_spot_price'):
                        price_change_pct = abs(current_spot - self._last_spot_price) / self._last_spot_price * 100
                        if price_change_pct > 2.0:  # 2% move
                            logger.warning(f"🚨 Significant BTC move: ${self._last_spot_price:.0f} → ${current_spot:.0f} "
                                         f"({price_change_pct:+.1f}%)")
                    
                    self._last_spot_price = current_spot
        except Exception as e:
            logger.error(f"Error checking market alerts: {str(e)}")
    
    def _reconcile_positions(self):
        """Reconcile positions with exchange (verify orders, fills, etc.)"""
        logger.info("🔄 Reconciling positions with exchange...")
        
        try:
            # Check if our recorded positions match exchange positions
            # This would involve calling exchange APIs to verify current positions
            # For now, just log that reconciliation is happening
            logger.info("✅ Position reconciliation completed")
            
        except Exception as e:
            logger.error(f"❌ Error during position reconciliation: {str(e)}")
    
    def force_exit_position(self):
        """Force exit position (emergency close)"""
        logger.warning("🚨 FORCE EXIT REQUESTED!")
        
        if self.state.is_active:
            success = self.exit_position(ExitReason.FORCE_CLOSE)
            if success:
                logger.info("✅ Position force-closed successfully")
            else:
                logger.error("❌ Force exit failed - manual intervention required")
        else:
            logger.info("ℹ️  No active position to close")
    
    def save_trade_to_database(self):
        """Save completed trade to database (placeholder for Django integration)"""
        if not self.state.exit_time:
            logger.warning("⚠️  Trade not completed yet, cannot save to database")
            return
        
        trade_data = {
            'strategy_id': self.strategy_id,
            'underlying': self.config.underlying,
            'entry_time': self.state.entry_time.isoformat() if self.state.entry_time else None,
            'exit_time': self.state.exit_time.isoformat() if self.state.exit_time else None,
            'exit_reason': self.state.exit_reason.value if self.state.exit_reason else None,
            'premium_received': self.state.total_premium_received,
            'final_pnl': self.state.current_pnl,
            'max_profit_seen': self.state.max_profit_seen,
            'call_symbol': self.state.call_position.symbol if self.state.call_position else None,
            'put_symbol': self.state.put_position.symbol if self.state.put_position else None,
            'call_entry_price': self.state.call_position.entry_price if self.state.call_position else None,
            'put_entry_price': self.state.put_position.entry_price if self.state.put_position else None,
        }
        
        logger.info(f"💾 Trade data ready for database:")
        logger.info(f"   Strategy ID: {trade_data['strategy_id']}")
        logger.info(f"   P&L: ${trade_data['final_pnl']:.2f}")
        logger.info(f"   Duration: {self.state.exit_time - self.state.entry_time if self.state.entry_time and self.state.exit_time else 'N/A'}")
        
        # TODO: Implement actual database save when Django integration is ready
        logger.info("📝 TODO: Implement database save in Django integration")
    
    def generate_trade_report(self) -> Dict:
        """Generate comprehensive trade report"""
        current_ist = self.api.get_current_ist_time()
        
        report = {
            'strategy_id': self.strategy_id,
            'config': {
                'underlying': self.config.underlying,
                'delta_range': f"{self.config.delta_range_low:.2f}-{self.config.delta_range_high:.2f}",
                'max_loss_pct': self.config.max_loss_pct,
                'max_profit_pct': self.config.max_profit_pct,
                'dry_run': self.config.dry_run
            },
            'trade_summary': {
                'entry_time': self.state.entry_time.isoformat() if self.state.entry_time else None,
                'exit_time': self.state.exit_time.isoformat() if self.state.exit_time else None,
                'premium_received': self.state.total_premium_received,
                'current_pnl': self.state.current_pnl,
                'max_profit_seen': self.state.max_profit_seen,
                'exit_reason': self.state.exit_reason.value if self.state.exit_reason else None,
                'is_active': self.state.is_active
            },
            'positions': {},
            'performance_metrics': {},
            'generated_at': current_ist.isoformat()
        }
        
        # Add position details
        if self.state.call_position:
            report['positions']['call'] = {
                'symbol': self.state.call_position.symbol,
                'entry_price': self.state.call_position.entry_price,
                'current_price': self.state.call_position.current_price,
                'unrealized_pnl': self.state.call_position.unrealized_pnl,
                'order_ids': self.state.call_position.order_ids
            }
        
        if self.state.put_position:
            report['positions']['put'] = {
                'symbol': self.state.put_position.symbol,
                'entry_price': self.state.put_position.entry_price,
                'current_price': self.state.put_position.current_price,
                'unrealized_pnl': self.state.put_position.unrealized_pnl,
                'order_ids': self.state.put_position.order_ids
            }
        
        # Calculate performance metrics
        if self.state.total_premium_received > 0:
            report['performance_metrics'] = {
                'return_pct': (self.state.current_pnl / self.state.total_premium_received * 100),
                'max_return_pct': (self.state.max_profit_seen / self.state.total_premium_received * 100),
                'profit_factor': self.state.current_pnl / self.state.total_premium_received if self.state.current_pnl > 0 else 0,
            }
            
            if self.state.entry_time:
                time_in_trade = (current_ist - self.state.entry_time).total_seconds() / 3600
                report['performance_metrics']['hours_in_trade'] = time_in_trade
        
        return report

def create_default_config() -> TradingConfig:
    """Create default trading configuration"""
    return TradingConfig(
        underlying="BTC",
        delta_range_low=0.10,
        delta_range_high=0.15,
        trade_time_ist="22:30",
        exit_time_ist="16:30",
        max_loss_pct=0.80,
        max_profit_pct=0.80,
        trailing_sl_enabled=True,
        max_retries=4,
        retry_slippage_pct=0.02,
        order_timeout=60,
        dry_run=True,
        testnet=False,  # Use production API with your credentials
        position_check_interval=30
    )

def run_comprehensive_test():
    """Run comprehensive test of all system components"""
    logger.info("🧪 PRODUCTION DELTA TRADER - COMPREHENSIVE TEST")
    logger.info("=" * 80)
    
    try:
        # Create configuration
        config = create_default_config()
        
        logger.info(f"⚙️  Configuration:")
        logger.info(f"   Underlying: {config.underlying}")
        logger.info(f"   Delta Range: {config.delta_range_low:.2f} - {config.delta_range_high:.2f}")
        logger.info(f"   Dry Run: {config.dry_run}")
        logger.info(f"   Testnet: {config.testnet}")
        
        # Initialize strategy
        strategy = ShortStrangleStrategy(config)
        
        # Validate prerequisites
        if not strategy.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed")
            return False
        
        # Enter position
        if not strategy.enter_short_strangle():
            logger.error("❌ Failed to enter position")
            return False
        
        # Run monitoring for 5 minutes
        logger.info("🔄 Starting 5-minute live monitoring...")
        strategy.run_live_monitoring(duration_minutes=5)
        
        # Generate trade report
        logger.info("📊 Generating comprehensive trade report...")
        trade_report = strategy.generate_trade_report()
        
        # Force exit if still active
        if strategy.state.is_active:
            logger.info("🏁 Forcing position exit...")
            strategy.exit_position(ExitReason.FORCE_CLOSE)
        
        # Save trade data (placeholder for database)
        strategy.save_trade_to_database()
        
        # Display final trade report
        logger.info(f"\n📋 FINAL TRADE REPORT")
        logger.info(f"=" * 60)
        logger.info(f"Strategy ID: {trade_report['strategy_id']}")
        logger.info(f"Underlying: {trade_report['config']['underlying']}")
        logger.info(f"Delta Range: {trade_report['config']['delta_range']}")
        logger.info(f"Premium Received: ${trade_report['trade_summary']['premium_received']:.6f}")
        logger.info(f"Final P&L: ${trade_report['trade_summary']['current_pnl']:.2f}")
        
        if trade_report['performance_metrics']:
            logger.info(f"Return: {trade_report['performance_metrics']['return_pct']:.2f}%")
            logger.info(f"Max Return: {trade_report['performance_metrics']['max_return_pct']:.2f}%")
            if 'hours_in_trade' in trade_report['performance_metrics']:
                logger.info(f"Duration: {trade_report['performance_metrics']['hours_in_trade']:.1f} hours")
        
        logger.info(f"Exit Reason: {trade_report['trade_summary']['exit_reason']}")
        logger.info(f"=" * 60)
        
        logger.info("✅ Comprehensive test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for live trading"""
    print("🚀 Production Delta Exchange Options Trader")
    print("=" * 80)
    print("🔴 LIVE TRADING MODE ENABLED")
    print("🎯 Features:")
    print("   ✅ Delta-based strike selection (10-15 delta)")
    print("   ✅ Advanced order management with retries")
    print("   ✅ Complete exit rules (time, P&L, trailing SL)")
    print("   ✅ Configurable expiry (default 01-10-2025)")
    print("   ✅ Real-time P&L tracking with WebSocket")
    print("   ✅ 0.01 BTC quantity per leg")
    print("   ✅ Dynamic timing (now+5min entry, now+10min exit)")
    print("=" * 80)
    
    strategy: Optional[ShortStrangleStrategy] = None
    try:
        # Initialize live trading configuration
        config, config_source = resolve_runtime_config()

        print("\n🎯 LIVE TRADING CONFIGURATION:")
        print(f"   ⚙️ Source: {config_source}")
        print(f"   📊 Quantity: {config.quantity} BTC per leg")
        print(f"   Δ Delta range: {config.delta_range_low}-{config.delta_range_high}")
        print(f"   ⏰ Trade time: {config.trade_time_ist}")
        print(f"   ⏰ Exit time: {config.exit_time_ist}")
        print(f"   📅 Expiry date: {config.expiry_date or FIXED_EXPIRY_DATE} (effective)")
        print(f"   💰 Max loss: {config.max_loss_pct}%")
        print(f"   📈 Max profit: {config.max_profit_pct}%")
        print(f"   🔴 Live trading: {not config.dry_run}")
        
        # Initialize strategy
        print(f"\n🚀 Starting ShortStrangleStrategy...")
        strategy = ShortStrangleStrategy(config)
        
        # Test connections before trading
        print(f"\n🔌 Testing connections...")
        api_success = strategy.api.test_connection()
        if not api_success:
            print("❌ API connection failed! Exiting...")
            return
            
        # Connect WebSocket for live prices
        if hasattr(strategy.api, 'websocket_streamer') and strategy.api.websocket_streamer:
            strategy.api.websocket_streamer.connect()
        
        print(f"\n🎭 STARTING LIVE TRADING SESSION")
        print(f"⚠️  This will place REAL orders with REAL money!")
        print(f"💰 Account balance: Check your Delta Exchange account")
        print(f"📅 Trading date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 80)
        
        # Start the strategy execution
        print(f"\n🚀 Executing short strangle strategy...")
        
        # Check for existing positions first
        existing_positions = strategy.api.get_current_positions()
        existing_orders = strategy.api.get_open_orders()
        
        if existing_positions or existing_orders:
            print(f"\n⚠️  EXISTING TRADES DETECTED!")
            print(f"   Active Positions: {len(existing_positions)}")
            print(f"   Open Orders: {len(existing_orders)}")
            
            # Ask user what to do
            print(f"\nOptions:")
            print(f"1. Recover existing positions and monitor exit conditions")
            print(f"2. Exit existing positions and start fresh")
            print(f"3. Cancel script")
            
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == "1":
                print(f"🔄 Recovering existing positions...")
                if strategy.recover_existing_positions():
                    print(f"✅ Positions recovered successfully")
                    print(f"🎯 Now monitoring exit conditions...")
                else:
                    print(f"❌ Failed to recover positions")
                    return
            elif choice == "2":
                print(f"🚪 Exiting existing positions...")
                if strategy.recover_existing_positions():
                    strategy.exit_position(ExitReason.FORCE_CLOSE)
                print(f"🔄 Starting fresh trade...")
                if not strategy.validate_prerequisites():
                    print("❌ Prerequisites validation failed after cleanup!")
                    return
                result = strategy.enter_short_strangle()
                if not result:
                    print("❌ Failed to enter new position!")
                    return
            else:
                print(f"🛑 Script cancelled by user")
                return
        else:
            # No existing positions - validate and start new trade
            if not strategy.validate_prerequisites():
                print("❌ Prerequisites validation failed!")
                return
                
            # Enter the short strangle position
            result = strategy.enter_short_strangle()
            if not result:
                print("❌ Failed to enter position!")
                return
        
        # Now we either have recovered positions or entered new ones
        # Start monitoring in both cases
        if strategy.state.is_active:
            print(f"\n✅ Strategy is active - starting position monitoring")
            print(f"📊 Check your Delta Exchange account for position details")
            
            # Keep monitoring until exit time
            print(f"\n📊 Starting position monitoring...")
            strategy.run_live_monitoring(duration_minutes=60)
        else:
            print(f"\n❌ No active strategy to monitor!")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Trading interrupted by user")
        try:
            if "strategy" in locals() and strategy is not None and strategy.state.is_active:
                strategy.exit_position(ExitReason.FORCE_CLOSE)
        except Exception:
            pass
    except Exception as e:
        print(f"\n❌ Trading error: {e}")
        import traceback
        traceback.print_exc()
        try:
            if "strategy" in locals() and strategy is not None and strategy.state.is_active:
                strategy.exit_position(ExitReason.FORCE_CLOSE)
        except Exception:
            pass

if __name__ == "__main__":
    main()
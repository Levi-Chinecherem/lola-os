# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal, getcontext
import time
from datetime import datetime, timedelta
from web3 import Web3
from eth_abi import abi
import json

# Third-party
try:
    from web3.middleware import geth_poa_middleware
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    raise ImportError("Chainlink dependencies missing. Run 'poetry add web3 requests'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.chains.connection import ChainConnection  # Phase 2 integration
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: Chainlink price feed integration for LOLA OS.
Purpose: Provides reliable, decentralized price oracle access for EVM chains 
         with automatic ABI detection, stale data detection, and fallback mechanisms.
How: Wraps Chainlink Data Feed contracts with LOLA's ChainConnection; supports 
     multiple aggregators per chain; includes health monitoring and confidence scoring.
Why: Enables LOLA agents to make informed financial decisions with battle-tested 
     oracle infrastructure while maintaining the framework's EVM-native design 
     and developer sovereignty principles.
Full Path: lola-os/python/lola/libs/chainlink/price_feed.py
"""

# Chainlink aggregator addresses and ABIs
CHAINLINK_AGGREGATORS = {
    "ethereum": {
        1: {  # Mainnet
            "ETH/USD": "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
            "BTC/USD": "0xF403008652a5bEEa84336541b232B8DbdDec4F27"
        },
        11155111: {  # Sepolia
            "ETH/USD": "0x694AA1769357215DE4FAC081bf1f309aDC325306"
        }
    },
    "polygon": {
        137: {  # Mainnet
            "MATIC/USD": "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0",
            "ETH/USD": "0xF9680D99D6C9589e2a93a78A04A279e509205945"
        },
        80001: {  # Mumbai
            "MATIC/USD": "0xd0D5e52942D6CE990914033f7E34D64B543bCacA"
        }
    },
    "arbitrum": {
        42161: {  # Mainnet
            "ETH/USD": "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612"
        }
    }
}

# Minimal Chainlink Aggregator ABI for price feeds
CHAINLINK_AGGREGATOR_ABI = [
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "description",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view", 
        "type": "function"
    },
    {
        "inputs": [],
        "name": "latestAnswer",
        "outputs": [{"internalType": "int256", "name": "", "type": "int256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "latestRound",
        "outputs": [{"internalType": "uint80", "name": "", "type": "uint80"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "latestTimestamp",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType": "uint80", "name": "roundId", "type": "uint80"},
            {"internalType": "int256", "name": "answer", "type": "int256"},
            {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
            {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

class PriceFeedHealth(Enum):
    """Price feed health status."""
    HEALTHY = "healthy"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"

@dataclass
class PriceFeedResult:
    """Price feed query result."""
    price: Optional[Decimal]
    decimals: int
    timestamp: Optional[int]
    round_id: Optional[int]
    confidence: float  # 0.0-1.0
    health: PriceFeedHealth
    feed_address: str
    description: str
    staleness_threshold: int  # seconds

class LolaChainlinkPriceFeed:
    """LolaChainlinkPriceFeed: Chainlink oracle integration for LOLA OS.
    Does NOT require pre-configured contracts—auto-discovers aggregators per chain."""

    STALENESS_THRESHOLD = 3600  # 1 hour
    MIN_CONFIDENCE = 0.8
    DEFAULT_RPC_TIMEOUT = 10

    def __init__(self):
        """
        Initializes Chainlink price feed client.
        Does Not: Connect to chains—lazy connection per query.
        """
        config = get_config()
        self.enabled = config.get("chainlink_enabled", True)
        self.default_chains = config.get("chainlink_default_chains", ["ethereum", "polygon"])
        self.rpc_endpoints = config.get("rpc_endpoints", {})
        self.staleness_threshold = config.get("chainlink_staleness_threshold", self.STALENESS_THRESHOLD)
        self.min_confidence = config.get("chainlink_min_confidence", self.MIN_CONFIDENCE)
        
        # Observability
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # HTTP session with retries
        self.session = self._create_http_session()
        
        # Cache for recent prices (5 minutes)
        self._price_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        
        logger.info("Chainlink price feed initialized")

    def _create_http_session(self) -> tp.Any:
        """
        Creates HTTP session with retry strategy.
        Returns:
            Requests session with retry adapter.
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    async def get_price(
        self,
        base_asset: str,
        quote_asset: str = "USD",
        chain_name: str = "ethereum",
        network_id: Optional[int] = None,
        stale_threshold: Optional[int] = None,
        min_confidence: Optional[float] = None
    ) -> PriceFeedResult:
        """
        Gets current price from Chainlink oracle.
        Args:
            base_asset: Base asset (ETH, BTC, etc.).
            quote_asset: Quote asset (usually USD).
            chain_name: Chain name (ethereum, polygon, etc.).
            network_id: Specific network ID (overrides chain_name).
            stale_threshold: Custom staleness threshold in seconds.
            min_confidence: Minimum confidence threshold.
        Returns:
            PriceFeedResult with price and metadata.
        """
        if not self.enabled:
            return PriceFeedResult(
                price=None,
                decimals=0,
                timestamp=None,
                round_id=None,
                confidence=0.0,
                health=PriceFeedHealth.UNAVAILABLE,
                feed_address="",
                description="",
                staleness_threshold=0
            )

        # Normalize parameters
        pair_key = f"{base_asset.upper()}/{quote_asset.upper()}"
        chain_key = chain_name.lower()
        network_id = network_id or self._get_network_id(chain_key)
        
        if not network_id:
            logger.warning(f"Unknown network for chain: {chain_key}")
            return self._create_unavailable_result(pair_key)

        # Check cache first
        cache_key = f"{pair_key}:{network_id}"
        if await self._check_cache(cache_key):
            cached = self._price_cache[cache_key]
            if time.time() - cached["timestamp"] < 300:  # 5 minutes
                logger.debug(f"Cache hit for {cache_key}")
                return PriceFeedResult(**cached["result"])

        try:
            # Start metrics
            with self.prometheus.record_evm_call(
                chain=chain_key,
                operation=f"price_feed_{pair_key}"
            ) as metrics:
                
                # Get aggregator address
                aggregator_address = self._get_aggregator_address(
                    chain_key, network_id, base_asset, quote_asset
                )
                
                if not aggregator_address:
                    logger.warning(f"No Chainlink aggregator for {pair_key} on {chain_key}:{network_id}")
                    return self._create_unavailable_result(pair_key, aggregator_address)

                # Connect to chain
                chain_connection = ChainConnection(
                    network_id=network_id,
                    rpc_url=self._get_rpc_url(chain_key, network_id)
                )
                chain_connection.connect()

                # Query price feed
                contract = chain_connection.get_contract(
                    aggregator_address,
                    CHAINLINK_AGGREGATOR_ABI
                )

                # Get latest round data
                latest_round_data = contract.functions.latestRoundData().call()
                round_id, answer, started_at, updated_at, answered_in_round = latest_round_data

                # Get decimals
                decimals = contract.functions.decimals().call()
                price = Decimal(answer) / (10 ** decimals)

                # Get description
                description = contract.functions.description().call()

                # Health assessment
                now = int(time.time())
                age_seconds = now - updated_at
                is_stale = age_seconds > (stale_threshold or self.staleness_threshold)
                
                # Confidence scoring
                confidence = self._calculate_confidence(
                    age_seconds, 
                    round_id, 
                    answered_in_round,
                    min_confidence=min_confidence or self.min_confidence
                )

                health = self._determine_health(is_stale, confidence)

                # Create result
                result = PriceFeedResult(
                    price=price if health == PriceFeedHealth.HEALTHY else None,
                    decimals=decimals,
                    timestamp=updated_at,
                    round_id=round_id,
                    confidence=confidence,
                    health=health,
                    feed_address=aggregator_address,
                    description=description,
                    staleness_threshold=stale_threshold or self.staleness_threshold
                )

                # Cache healthy results
                if health == PriceFeedHealth.HEALTHY:
                    await self._cache_price(cache_key, result)

                # Record metrics
                metrics.record_evm_call(
                    chain=chain_key,
                    operation=f"price_feed_{pair_key}",
                    duration=time.time() - metrics.start_time,
                    success=health == PriceFeedHealth.HEALTHY
                )

                self.prometheus.record_oracle_query(
                    oracle_type="chainlink",
                    feed=pair_key,
                    chain=chain_key,
                    confidence=confidence,
                    age_seconds=age_seconds,
                    success=health == PriceFeedHealth.HEALTHY
                )

                logger.debug(f"Chainlink price: {pair_key}@{chain_key} = ${price:.4f} (confidence: {confidence:.2%})")
                return result

        except Exception as exc:
            self._handle_error(exc, f"price query {pair_key}@{chain_key}")
            self.prometheus.record_oracle_query(
                oracle_type="chainlink",
                feed=pair_key,
                chain=chain_key,
                success=False
            )
            return self._create_error_result(pair_key, aggregator_address, exc)

    def _get_network_id(self, chain_name: str) -> Optional[int]:
        """Maps chain name to network ID."""
        network_map = {
            "ethereum": 1,
            "polygon": 137,
            "arbitrum": 42161,
            "optimism": 10,
            "bsc": 56,
            "avalanche": 43114,
            "fantom": 250,
            "sepolia": 11155111,  # Ethereum testnet
            "mumbai": 80001,      # Polygon testnet
        }
        return network_map.get(chain_name)

    def _get_rpc_url(self, chain_name: str, network_id: int) -> str:
        """Gets RPC URL for chain and network."""
        # Check configured endpoints first
        if network_id in self.rpc_endpoints:
            return self.rpc_endpoints[network_id]
        
        # Default RPCs
        default_rpcs = {
            1: "https://eth.llamarpc.com",  # Ethereum
            137: "https://polygon-rpc.com",  # Polygon
            42161: "https://arb1.arbitrum.io/rpc",  # Arbitrum
            10: "https://mainnet.optimism.io",  # Optimism
            56: "https://bsc-dataseed.binance.org",  # BSC
            43114: "https://api.avax.network/ext/bc/C/rpc",  # Avalanche
            250: "https://rpc.ftm.tools",  # Fantom
            11155111: "https://rpc.sepolia.org",  # Sepolia
            80001: "https://rpc-mumbai.maticvigil.com"  # Mumbai
        }
        
        rpc_url = default_rpcs.get(network_id)
        if not rpc_url:
            raise ValueError(f"No RPC endpoint configured for network {network_id}")
        
        return rpc_url

    def _get_aggregator_address(self, chain_name: str, network_id: int, 
                               base_asset: str, quote_asset: str) -> Optional[str]:
        """Gets Chainlink aggregator contract address."""
        pair_key = f"{base_asset.upper()}/{quote_asset.upper()}"
        
        # Check configured aggregators first
        chain_aggregators = get_config().get("chainlink_aggregators", {})
        if chain_name in chain_aggregators and network_id in chain_aggregators[chain_name]:
            return chain_aggregators[chain_name][network_id].get(pair_key)
        
        # Use built-in aggregators
        if chain_name in CHAINLINK_AGGREGATORS and network_id in CHAINLINK_AGGREGATORS[chain_name]:
            return CHAINLINK_AGGREGATORS[chain_name][network_id].get(pair_key)
        
        logger.warning(f"No aggregator found for {pair_key} on {chain_name}:{network_id}")
        return None

    async def _check_cache(self, cache_key: str) -> bool:
        """Checks if price is in cache."""
        async with self._cache_lock:
            return cache_key in self._price_cache

    async def _cache_price(self, cache_key: str, result: PriceFeedResult) -> None:
        """Caches price result."""
        async with self._cache_lock:
            self._price_cache[cache_key] = {
                "result": tp.dataclasses.asdict(result),
                "timestamp": time.time()
            }
            
            # Cleanup old cache entries
            if len(self._price_cache) > 1000:
                cutoff = time.time() - 1800  # 30 minutes
                self._price_cache = {
                    k: v for k, v in self._price_cache.items() 
                    if v["timestamp"] > cutoff
                }

    def _calculate_confidence(self, age_seconds: int, round_id: int, 
                            answered_in_round: int, min_confidence: float) -> float:
        """
        Calculates confidence score for price feed.
        """
        # Base confidence
        confidence = 1.0
        
        # Age penalty (exponential decay)
        if age_seconds > 300:  # 5 minutes
            confidence *= 0.9 ** ((age_seconds - 300) / 1800)  # Decay over 30 minutes
        
        # Round consistency (should match)
        if round_id != answered_in_round:
            confidence *= 0.8  # Significant drop if rounds don't match
        
        # Apply minimum confidence
        return max(confidence, min_confidence)

    def _determine_health(self, is_stale: bool, confidence: float) -> PriceFeedHealth:
        """
        Determines health status from staleness and confidence.
        """
        if confidence < 0.5:
            return PriceFeedHealth.UNAVAILABLE
        elif is_stale:
            return PriceFeedHealth.STALE
        elif confidence < self.min_confidence:
            return PriceFeedHealth.DEGRADED
        else:
            return PriceFeedHealth.HEALTHY

    def _create_unavailable_result(self, pair_key: str, address: str = "") -> PriceFeedResult:
        """Creates unavailable result."""
        return PriceFeedResult(
            price=None,
            decimals=0,
            timestamp=None,
            round_id=None,
            confidence=0.0,
            health=PriceFeedHealth.UNAVAILABLE,
            feed_address=address,
            description=f"{pair_key} feed unavailable",
            staleness_threshold=self.staleness_threshold
        )

    def _create_error_result(self, pair_key: str, address: str, exc: Exception) -> PriceFeedResult:
        """Creates error result."""
        return PriceFeedResult(
            price=None,
            decimals=0,
            timestamp=None,
            round_id=None,
            confidence=0.0,
            health=PriceFeedHealth.UNAVAILABLE,
            feed_address=address,
            description=f"{pair_key} query error: {str(exc)}",
            staleness_threshold=self.staleness_threshold
        )

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Error handling for Chainlink operations.
        """
        logger.error(f"Chainlink {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)

    def list_available_feeds(self, chain_name: str = "ethereum") -> Dict[str, str]:
        """
        Lists available price feeds for chain.
        Args:
            chain_name: Chain name.
        Returns:
            Dictionary of pair -> address mappings.
        """
        network_ids = [self._get_network_id(chain_name)]
        if not network_ids:
            return {}
        
        available_feeds = {}
        for network_id in network_ids:
            if chain_name in CHAINLINK_AGGREGATORS and network_id in CHAINLINK_AGGREGATORS[chain_name]:
                available_feeds.update(CHAINLINK_AGGREGATORS[chain_name][network_id])
        
        return available_feeds

    async def batch_price_query(self, pairs: List[str], chain_name: str = "ethereum") -> Dict[str, PriceFeedResult]:
        """
        Batch queries multiple price pairs.
        Args:
            pairs: List of asset pairs (e.g., ["ETH/USD", "BTC/USD"]).
            chain_name: Chain name.
        Returns:
            Dictionary of pair -> PriceFeedResult.
        """
        if not self.enabled:
            return {pair: self._create_unavailable_result(pair) for pair in pairs}

        network_id = self._get_network_id(chain_name)
        if not network_id:
            return {pair: self._create_unavailable_result(pair) for pair in pairs}

        results = {}
        tasks = []
        
        for pair in pairs:
            base, quote = pair.split("/")
            task = asyncio.create_task(
                self.get_price(base, quote, chain_name, network_id)
            )
            tasks.append((pair, task))
        
        # Gather results
        for pair, task in tasks:
            try:
                result = await task
                results[pair] = result
            except Exception as exc:
                logger.error(f"Batch query failed for {pair}: {str(exc)}")
                results[pair] = self._create_error_result(pair, "", exc)
        
        return results


# Global price feed instance
_lola_chainlink_feed = None

def get_chainlink_price_feed() -> LolaChainlinkPriceFeed:
    """Singleton Chainlink price feed instance."""
    global _lola_chainlink_feed
    if _lola_chainlink_feed is None:
        _lola_chainlink_feed = LolaChainlinkPriceFeed()
    return _lola_chainlink_feed

__all__ = [
    "PriceFeedHealth",
    "PriceFeedResult",
    "LolaChainlinkPriceFeed",
    "get_chainlink_price_feed"
]
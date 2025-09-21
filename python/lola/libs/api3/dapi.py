# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional
import time
from decimal import Decimal
import hashlib
from eth_abi import abi

# Third-party
try:
    from web3 import Web3
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    raise ImportError("API3 dependencies missing. Run 'poetry add web3 requests'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.chains.connection import ChainConnection  # Phase 2
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: API3 dAPI integration for LOLA OS.
Purpose: Provides decentralized API access through API3's dAPIs as Chainlink alternative 
         with support for both on-chain verification and off-chain data feeds.
How: Wraps API3 dAPI contracts and endpoints; supports both signed data feeds 
     and push/pull models; includes verification, freshness checks, and confidence scoring.
Why: Offers cost-effective, developer-friendly alternative to Chainlink for 
     LOLA agents requiring real-world data while maintaining EVM-native 
     verification and the framework's oracle abstraction layer.
Full Path: lola-os/python/lola/libs/api3/dapi.py
"""

# API3 dAPI configurations
API3_DAPIS = {
    "ethereum": {
        1: {  # Mainnet
            "ETH/USD": {
                "beacon_id": "0x0000000000000000000000000000000000000000000000000000000000000000",  # Placeholder
                "address": "0x...",  # dAPI contract address
                "signed": True
            },
            "BTC/USD": {
                "beacon_id": "0x0000000000000000000000000000000000000000000000000000000000000000",
                "address": "0x...",
                "signed": True
            }
        }
    },
    "polygon": {
        137: {
            "MATIC/USD": {
                "beacon_id": "0x...",
                "address": "0x...",
                "signed": True
            }
        }
    }
}

# Minimal API3 dAPI ABI
API3_DAPI_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "beaconId", "type": "bytes32"}],
        "name": "read",
        "outputs": [
            {"internalType": "int224", "name": "value", "type": "int224"},
            {"internalType": "uint32", "name": "timestamp", "type": "uint32"},
            {"internalType": "uint256", "name": "beaconIdHash", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "beaconId", "type": "bytes32"}],
        "name": "getDataFeed",
        "outputs": [
            {"internalType": "address", "name": "sender", "type": "address"},
            {"internalType": "bytes32", "name": "beaconId", "type": "bytes32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    }
]

class dAPIHealth(Enum):
    """dAPI health status."""
    HEALTHY = "healthy"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    VERIFICATION_FAILED = "verification_failed"

@dataclass
class dAPIResult:
    """API3 dAPI query result."""
    value: Optional[Decimal]
    decimals: int
    timestamp: Optional[int]
    beacon_id: str
    confidence: float  # 0.0-1.0
    health: dAPIHealth
    dapi_address: str
    verification_status: bool
    staleness_threshold: int  # seconds

class LolaAPI3dAPI:
    """LolaAPI3dAPI: API3 decentralized API integration.
    Does NOT require pre-configured dAPIs—supports discovery and fallback."""

    STALENESS_THRESHOLD = 1800  # 30 minutes
    VERIFICATION_TIMEOUT = 30
    DEFAULT_RPC_TIMEOUT = 10

    def __init__(self):
        """
        Initializes API3 dAPI client.
        """
        config = get_config()
        self.enabled = config.get("api3_enabled", True)
        self.default_chains = config.get("api3_default_chains", ["ethereum", "polygon"])
        self.rpc_endpoints = config.get("rpc_endpoints", {})
        self.api_endpoints = config.get("api3_endpoints", {})
        self.staleness_threshold = config.get("api3_staleness_threshold", self.STALENESS_THRESHOLD)
        
        # Observability
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        # HTTP session
        self.session = self._create_http_session()
        
        # Cache
        self._dapi_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        
        logger.info("API3 dAPI client initialized")

    def _create_http_session(self) -> tp.Any:
        """Creates HTTP session with retries."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    async def get_data(
        self,
        data_feed_id: str,
        chain_name: str = "ethereum",
        network_id: Optional[int] = None,
        stale_threshold: Optional[int] = None
    ) -> dAPIResult:
        """
        Gets data from API3 dAPI.
        Args:
            data_feed_id: dAPI identifier.
            chain_name: Chain name.
            network_id: Network ID.
            stale_threshold: Staleness threshold.
        Returns:
            dAPIResult with data and metadata.
        """
        if not self.enabled:
            return dAPIResult(
                value=None,
                decimals=0,
                timestamp=None,
                beacon_id="",
                confidence=0.0,
                health=dAPIHealth.UNAVAILABLE,
                dapi_address="",
                verification_status=False,
                staleness_threshold=0
            )

        # Normalize parameters
        network_id = network_id or self._get_network_id(chain_name)
        if not network_id:
            return self._create_unavailable_result(data_feed_id)

        # Check cache
        cache_key = f"{data_feed_id}:{network_id}"
        if await self._check_cache(cache_key):
            cached = self._dapi_cache[cache_key]
            if time.time() - cached["timestamp"] < 300:  # 5 minutes
                return dAPIResult(**cached["result"])

        try:
            with self.prometheus.record_evm_call(
                chain=chain_name,
                operation=f"dapi_{data_feed_id}"
            ) as metrics:
                
                # Try on-chain first
                onchain_result = await self._query_onchain_dapi(
                    data_feed_id, chain_name, network_id
                )
                
                if onchain_result and onchain_result.verification_status:
                    # Cache successful on-chain result
                    await self._cache_dapi_result(cache_key, onchain_result)
                    return onchain_result
                
                # Fallback to off-chain
                offchain_result = await self._query_offchain_dapi(
                    data_feed_id, chain_name, network_id
                )
                
                if offchain_result:
                    # Verify off-chain data if possible
                    verification_result = await self._verify_offchain_data(
                        offchain_result, chain_name, network_id
                    )
                    offchain_result.verification_status = verification_result
                    await self._cache_dapi_result(cache_key, offchain_result)
                    return offchain_result
                else:
                    # Both failed
                    return self._create_unavailable_result(data_feed_id)

        except Exception as exc:
            self._handle_error(exc, f"dAPI query {data_feed_id}")
            return self._create_error_result(data_feed_id, exc)

    async def _query_onchain_dapi(self, data_feed_id: str, chain_name: str, 
                                network_id: int) -> Optional[dAPIResult]:
        """Queries dAPI on-chain."""
        try:
            # Get dAPI configuration
            dapi_config = self._get_dapi_config(chain_name, network_id, data_feed_id)
            if not dapi_config:
                logger.debug(f"No on-chain dAPI config for {data_feed_id} on {chain_name}:{network_id}")
                return None

            # Connect to chain
            chain_connection = ChainConnection(network_id=network_id)
            chain_connection.connect()

            # Query contract
            contract = chain_connection.get_contract(
                dapi_config["address"],
                API3_DAPI_ABI
            )

            # Call read function
            read_result = contract.functions.read(
                Web3.toBytes(hexstr=dapi_config["beacon_id"])
            ).call()

            value, timestamp, beacon_id_hash = read_result
            decimals = contract.functions.decimals().call()

            # Health assessment
            now = int(time.time())
            age_seconds = now - timestamp
            is_stale = age_seconds > (self.staleness_threshold)
            confidence = 1.0 if not is_stale else max(0.0, 1.0 - (age_seconds / 3600))

            health = dAPIHealth.HEALTHY if not is_stale else dAPIHealth.STALE

            return dAPIResult(
                value=Decimal(value) / (10 ** decimals) if value != 0 else None,
                decimals=decimals,
                timestamp=timestamp,
                beacon_id=data_feed_id,
                confidence=confidence,
                health=health,
                dapi_address=dapi_config["address"],
                verification_status=True,  # On-chain is verified
                staleness_threshold=self.staleness_threshold
            )

        except Exception as exc:
            logger.debug(f"On-chain dAPI query failed: {str(exc)}")
            return None

    async def _query_offchain_dapi(self, data_feed_id: str, chain_name: str, 
                                 network_id: int) -> Optional[dAPIResult]:
        """Queries dAPI off-chain via API endpoint."""
        try:
            # Get off-chain endpoint
            endpoint_config = self._get_offchain_endpoint(chain_name, network_id, data_feed_id)
            if not endpoint_config:
                return None

            # Query API
            response = self.session.get(
                endpoint_config["url"].format(beacon_id=data_feed_id),
                timeout=10,
                headers={"User-Agent": "LOLA-OS/1.0"}
            )

            if response.status_code != 200:
                logger.warning(f"Off-chain dAPI HTTP {response.status_code}: {data_feed_id}")
                return None

            data = response.json()
            value = Decimal(data.get("value", 0))
            timestamp = data.get("timestamp")
            decimals = data.get("decimals", 18)

            # Basic health check
            now = int(time.time())
            age_seconds = now - timestamp if timestamp else float('inf')
            is_stale = age_seconds > self.staleness_threshold
            confidence = 0.7 if not is_stale else 0.3  # Lower confidence for off-chain

            health = dAPIHealth.HEALTHY if not is_stale else dAPIHealth.STALE

            return dAPIResult(
                value=value / (10 ** decimals) if value != 0 else None,
                decimals=decimals,
                timestamp=timestamp,
                beacon_id=data_feed_id,
                confidence=confidence,
                health=health,
                dapi_address="offchain",
                verification_status=False,  # Pending verification
                staleness_threshold=self.staleness_threshold
            )

        except Exception as exc:
            logger.debug(f"Off-chain dAPI query failed: {str(exc)}")
            return None

    async def _verify_offchain_data(self, result: dAPIResult, chain_name: str, 
                                  network_id: int) -> bool:
        """Verifies off-chain data against on-chain."""
        try:
            # Get on-chain value for comparison
            onchain_result = await self._query_onchain_dapi(
                result.beacon_id, chain_name, network_id
            )
            
            if not onchain_result or not onchain_result.value:
                return False
            
            # Compare values (within 1% tolerance)
            tolerance = Decimal('0.01')
            difference = abs(result.value - onchain_result.value) / onchain_result.value
            
            verification_success = difference <= tolerance
            logger.debug(f"dAPI verification {result.beacon_id}: {'PASS' if verification_success else 'FAIL'} (diff: {difference:.2%})")
            
            return verification_success
            
        except Exception as exc:
            logger.warning(f"dAPI verification failed: {str(exc)}")
            return False

    def _get_dapi_config(self, chain_name: str, network_id: int, 
                        data_feed_id: str) -> Optional[Dict[str, Any]]:
        """Gets dAPI configuration."""
        # Check configured dAPIs first
        dapis_config = get_config().get("api3_dapis", {})
        if chain_name in dapis_config and network_id in dapis_config[chain_name]:
            return dapis_config[chain_name][network_id].get(data_feed_id)
        
        # Use built-in (placeholder for now)
        if chain_name in API3_DAPIS and network_id in API3_DAPIS[chain_name]:
            return API3_DAPIS[chain_name][network_id].get(data_feed_id)
        
        return None

    def _get_offchain_endpoint(self, chain_name: str, network_id: int, 
                              data_feed_id: str) -> Optional[Dict[str, Any]]:
        """Gets off-chain API endpoint."""
        endpoints_config = get_config().get("api3_endpoints", {})
        if network_id in endpoints_config:
            return endpoints_config[network_id].get(data_feed_id)
        
        # Default endpoint pattern
        return {
            "url": f"https://dapi.api3.org/{network_id}/{data_feed_id}/latest"
        }

    def _get_network_id(self, chain_name: str) -> Optional[int]:
        """Maps chain name to network ID."""
        network_map = {
            "ethereum": 1,
            "polygon": 137,
            "arbitrum": 42161,
            "optimism": 10,
            "bsc": 56,
            "avalanche": 43114,
            "fantom": 250
        }
        return network_map.get(chain_name.lower())

    async def _check_cache(self, cache_key: str) -> bool:
        """Checks cache."""
        async with self._cache_lock:
            return cache_key in self._dapi_cache

    async def _cache_dapi_result(self, cache_key: str, result: dAPIResult) -> None:
        """Caches dAPI result."""
        async with self._cache_lock:
            self._dapi_cache[cache_key] = {
                "result": tp.dataclasses.asdict(result),
                "timestamp": time.time()
            }

    def _create_unavailable_result(self, data_feed_id: str) -> dAPIResult:
        """Creates unavailable result."""
        return dAPIResult(
            value=None,
            decimals=0,
            timestamp=None,
            beacon_id=data_feed_id,
            confidence=0.0,
            health=dAPIHealth.UNAVAILABLE,
            dapi_address="",
            verification_status=False,
            staleness_threshold=self.staleness_threshold
        )

    def _create_error_result(self, data_feed_id: str, exc: Exception) -> dAPIResult:
        """Creates error result."""
        return dAPIResult(
            value=None,
            decimals=0,
            timestamp=None,
            beacon_id=data_feed_id,
            confidence=0.0,
            health=dAPIHealth.UNAVAILABLE,
            dapi_address="",
            verification_status=False,
            staleness_threshold=self.staleness_threshold
        )

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Error handling."""
        logger.error(f"API3 dAPI {context}: {str(exc)}")
        if self.sentry_dsn:
            capture_exception(exc)


# Global dAPI instance
_lola_api3_dapi = None

def get_api3_dapi() -> LolaAPI3dAPI:
    """Singleton API3 dAPI instance."""
    global _lola_api3_dapi
    if _lola_api3_dapi is None:
        _lola_api3_dapi = LolaAPI3dAPI()
    return _lola_api3_dapi

__all__ = [
    "dAPIHealth",
    "dAPIResult",
    "LolaAPI3dAPI",
    "get_api3_dapi"
]
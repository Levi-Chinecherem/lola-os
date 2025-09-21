# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional, Union
import asyncio
import subprocess
import json
import time
from pathlib import Path
import psutil
import signal
import atexit
import os
import shutil

# Third-party
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import docker
    from docker.errors import DockerException
except ImportError:
    raise ImportError("Ollama runner dependencies missing. Run 'poetry add requests docker'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: Ollama model runner and manager for LOLA OS.
Purpose: Automates Ollama installation, model management, container orchestration, 
         and health monitoring for local LLM execution.
How: Detects existing Ollama installations; auto-installs via Docker/script if missing; 
     manages model pulls, GPU detection, resource monitoring, and graceful shutdown.
Why: Enables true local-first development with privacy, zero API costs, and 
     offline capabilities while maintaining production-grade reliability.
Full Path: lola-os/python/lola/libs/ollama/runner.py
"""

class LolaOllamaRunner:
    """LolaOllamaRunner: Automated Ollama lifecycle management.
    Does NOT download models automatically—provides tools for management."""

    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_MODELS_DIR = "~/.ollama/models"
    SUPPORTED_ARCHITECTURES = ["linux/amd64", "linux/arm64", "darwin/amd64", "darwin/arm64"]

    def __init__(self):
        """
        Initializes Ollama runner with system detection and configuration.
        Does Not: Start Ollama—lazy startup on first model access.
        """
        config = get_config()
        self.ollama_url = config.get("ollama_base_url", self.DEFAULT_OLLAMA_URL)
        self.models_dir = Path(config.get("ollama_models_dir", self.DEFAULT_MODELS_DIR)).expanduser()
        self.auto_install = config.get("ollama_auto_install", True)
        self.use_docker = config.get("ollama_use_docker", True)
        self.gpu_support = config.get("ollama_gpu_support", True)
        self.model_cache_size_gb = config.get("ollama_cache_size_gb", 10)
        
        # Detect system capabilities
        self._detect_system_capabilities()
        
        # State management
        self._process = None
        self._docker_client = None
        self._health_check_interval = 30
        self._last_health_check = 0
        
        # Observability
        self.prometheus = get_lola_prometheus()
        self.sentry_dsn = config.get("sentry_dsn")
        
        logger.info(f"Ollama runner initialized: {self.ollama_url}, Docker: {self.use_docker}, GPU: {self.gpu_support}")

    def _detect_system_capabilities(self) -> None:
        """
        Detects CPU architecture, GPU availability, and Docker support.
        """
        import platform
        self.architecture = platform.machine()
        self.os = platform.system().lower()
        
        # GPU detection
        self.nvidia_gpu = self._detect_nvidia_gpu()
        self.apple_silicon = self.os == "darwin" and self.architecture == "arm64"
        
        # Docker detection
        self.docker_available = self._check_docker()
        
        logger.debug(f"System: {self.os}/{self.architecture}, GPU: {self.nvidia_gpu}, Docker: {self.docker_available}")

    def _detect_nvidia_gpu(self) -> bool:
        """
        Detects NVIDIA GPU availability.
        Returns:
            True if NVIDIA GPU detected and drivers available.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and len(result.stdout.strip().split('\n')) > 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_docker(self) -> bool:
        """
        Checks if Docker is available and running.
        Returns:
            True if Docker daemon is accessible.
        """
        try:
            if self.use_docker:
                client = docker.from_env()
                client.ping()
                return True
            return False
        except DockerException:
            logger.warning("Docker not available or not running")
            return False

    async def ensure_ollama_running(self, timeout: int = 60) -> bool:
        """
        Ensures Ollama is running (starts if necessary).
        Args:
            timeout: Maximum wait time for startup.
        Returns:
            True if Ollama is running and healthy.
        """
        if await self.is_running():
            logger.debug("Ollama already running")
            return True
        
        if not self.auto_install:
            logger.error("Ollama not running and auto-install disabled")
            return False

        try:
            # Start Ollama
            if self.use_docker:
                success = await self._start_docker_ollama()
            else:
                success = await self._start_native_ollama()
            
            if not success:
                logger.error("Failed to start Ollama")
                return False
            
            # Wait for health
            start_time = time.time()
            while time.time() - start_time < timeout:
                if await self.is_healthy():
                    logger.info("Ollama started successfully")
                    return True
                await asyncio.sleep(2)
            
            logger.error(f"Ollama startup timed out after {timeout}s")
            return False
            
        except Exception as exc:
            self._handle_error(exc, "Ollama startup")
            return False

    async def _start_docker_ollama(self) -> bool:
        """
        Starts Ollama via Docker container.
        Returns:
            True if container started successfully.
        """
        try:
            if not self.docker_available:
                logger.error("Docker not available for Ollama")
                return False

            self._docker_client = docker.from_env()
            
            # Remove existing container if running
            try:
                existing = self._docker_client.containers.get("lola-ollama", ignore_removed=True)
                if existing.status == "running":
                    existing.stop()
                existing.remove()
            except docker.errors.NotFound:
                pass

            # GPU runtime if available
            runtime = "nvidia" if self.nvidia_gpu and self.gpu_support else None
            
            # Start container
            container = self._docker_client.containers.run(
                image="ollama/ollama:latest",
                name="lola-ollama",
                detach=True,
                ports={"11434/tcp": 11434},
                runtime=runtime,
                environment={
                    "OLLAMA_MODELS": str(self.models_dir),
                    "OLLAMA_NUM_PARALLEL": "4",
                    "OLLAMA_MAX_LOADED_MODELS": "2"
                },
                volumes={
                    str(self.models_dir): {"bind": "/root/.ollama/models", "mode": "rw"}
                },
                mem_limit=f"{self.model_cache_size_gb}g",
                restart_policy={"Name": "unless-stopped"}
            )
            
            self._process = container
            logger.info(f"Ollama Docker container started: {container.id[:12]}")
            
            # Register cleanup
            atexit.register(self._cleanup_docker)
            
            return True
            
        except Exception as exc:
            logger.error(f"Docker Ollama startup failed: {str(exc)}")
            if self.sentry_dsn:
                capture_exception(exc)
            return False

    async def _start_native_ollama(self) -> bool:
        """
        Starts Ollama as native process.
        Returns:
            True if process started successfully.
        """
        try:
            # Check if Ollama binary exists
            if not shutil.which("ollama"):
                logger.error("Ollama binary not found in PATH")
                return False

            # Start Ollama server
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # For process group management
            )
            
            logger.info(f"Ollama native process started: PID {self._process.pid}")
            
            # Register cleanup
            atexit.register(self._cleanup_native)
            
            return True
            
        except Exception as exc:
            logger.error(f"Native Ollama startup failed: {str(exc)}")
            return False

    async def is_running(self) -> bool:
        """
        Checks if Ollama server is running.
        Returns:
            True if server responds to health check.
        """
        try:
            # Simple health check
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    async def is_healthy(self) -> bool:
        """
        Performs comprehensive health check.
        Returns:
            True if Ollama is fully operational.
        """
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return self._last_health_check_result or False
        
        self._last_health_check = current_time
        
        try:
            # Check API endpoint
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                self._last_health_check_result = False
                return False
            
            # Check model loading capability
            test_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": "llama2", "prompt": "test", "stream": False},
                timeout=15
            )
            
            self._last_health_check_result = test_response.status_code == 200
            return self._last_health_check_result
            
        except Exception as exc:
            logger.debug(f"Ollama health check failed: {str(exc)}")
            self._last_health_check_result = False
            return False

    async def pull_model(self, model_name: str, timeout: int = 300) -> bool:
        """
        Downloads and prepares Ollama model.
        Args:
            model_name: Model identifier (llama2, mistral, etc.).
            timeout: Maximum wait time.
        Returns:
            True if model successfully pulled.
        """
        if not await self.ensure_ollama_running(timeout=30):
            logger.error("Cannot pull model - Ollama not running")
            return False

        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            existing_models = response.json().get("models", [])
            model_names = [m["name"] for m in existing_models]
            
            if model_name in model_names:
                logger.info(f"Model {model_name} already available")
                return True

            # Pull model
            logger.info(f"Pulling Ollama model: {model_name}")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Model pull failed: HTTP {response.status_code}")
                return False
            
            # Stream progress
            total_size = 0
            last_update = time.time()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "total" in data and "completed" in data:
                            progress = data["completed"] / data["total"] * 100
                            if time.time() - last_update > 5:  # Update every 5s
                                logger.info(f"Model pull progress: {progress:.1f}%")
                                last_update = time.time()
                    except json.JSONDecodeError:
                        continue
            
            duration = time.time() - start_time
            logger.info(f"Model {model_name} pulled successfully in {duration:.1f}s")
            return True
            
        except Exception as exc:
            self._handle_error(exc, f"model pull {model_name}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists available Ollama models.
        Returns:
            List of model information dictionaries.
        """
        if not await self.ensure_ollama_running():
            return []

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                formatted_models = []
                
                for model in models:
                    formatted_models.append({
                        "name": model["name"],
                        "size": model.get("size", 0),
                        "digest": model.get("digest", ""),
                        "details": model.get("details", {})
                    })
                
                return formatted_models
            else:
                logger.warning(f"Failed to list models: HTTP {response.status_code}")
                return []
                
        except Exception as exc:
            self._handle_error(exc, "list models")
            return []

    async def run_completion(self, model: str, prompt: str, 
                           stream: bool = False, **kwargs) -> Union[Dict[str, Any], asyncio.StreamReader]:
        """
        Runs completion through Ollama with error handling.
        Args:
            model: Model name.
            prompt: Prompt text.
            stream: Whether to stream response.
            **kwargs: Additional parameters.
        Returns:
            Response dictionary or stream reader.
        """
        if not await self.ensure_ollama_running():
            raise RuntimeError("Ollama not available")

        try:
            # Start metrics
            provider = "ollama"
            operation = "generate" if not stream else "generate_stream"
            
            with self.prometheus.llm_call(model, provider, operation):
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": stream,
                    **kwargs
                }
                
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    stream=stream,
                    timeout=kwargs.get("timeout", 120)
                )
                
                if response.status_code == 200:
                    if stream:
                        return response.iter_lines()
                    else:
                        result = response.json()
                        # Record tokens and cost (Ollama doesn't provide usage, estimate)
                        estimated_tokens = len(prompt.split()) + len(result["response"].split())
                        self.prometheus.record_llm_tokens(model, len(prompt.split()), estimated_tokens)
                        return result
                else:
                    raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
                    
        except Exception as exc:
            self._handle_error(exc, f"completion {model}")
            raise

    def _cleanup_docker(self) -> None:
        """Cleanup Docker container on exit."""
        if self._docker_client and self._process:
            try:
                container = self._docker_client.containers.get("lola-ollama", ignore_removed=True)
                if container.status == "running":
                    container.stop(timeout=30)
                container.remove()
                logger.debug("Ollama Docker container cleaned up")
            except Exception as exc:
                logger.warning(f"Docker cleanup failed: {str(exc)}")

    def _cleanup_native(self) -> None:
        """Cleanup native process on exit."""
        if self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=30)
                logger.debug("Ollama native process cleaned up")
            except Exception as exc:
                logger.warning(f"Native process cleanup failed: {str(exc)}")

    def _handle_error(self, exc: Exception, context: str) -> None:
        """
        Error handling for Ollama operations.
        """
        logger.error(f"Ollama {context}: {str(exc)}")
        config = get_config()
        if config.get("sentry_dsn"):
            capture_exception(exc)

    @property
    def resource_usage(self) -> Dict[str, Any]:
        """
        Returns current resource usage if Ollama is running.
        Returns:
            Resource usage dictionary.
        """
        if not self._process:
            return {"status": "not_running"}
        
        try:
            if self.use_docker and self._docker_client:
                container = self._docker_client.containers.get("lola-ollama")
                stats = container.stats(stream=False)
                
                return {
                    "status": "running",
                    "cpu_percent": stats["cpu_stats"]["cpu_usage"]["usage_in_kernelmode"],
                    "memory_usage_mb": stats["memory_stats"]["usage"] / 1024 / 1024,
                    "models_loaded": len(self.list_models())
                }
            else:
                # Native process monitoring
                process = psutil.Process(self._process.pid)
                return {
                    "status": "running",
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "models_loaded": len(self.list_models())
                }
        except Exception as exc:
            logger.debug(f"Resource monitoring failed: {str(exc)}")
            return {"status": "error", "error": str(exc)}


# Global runner instance
_lola_ollama_runner = None

def get_ollama_runner() -> LolaOllamaRunner:
    """Singleton Ollama runner instance."""
    global _lola_ollama_runner
    if _lola_ollama_runner is None:
        _lola_ollama_runner = LolaOllamaRunner()
    return _lola_ollama_runner

__all__ = [
    "LolaOllamaRunner",
    "get_ollama_runner"
]
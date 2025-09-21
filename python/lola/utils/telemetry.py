# Standard imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Local imports
from lola.utils.logging import logger

"""
File: Telemetry export for LOLA OS TMVP 1 Phase 3.

Purpose: Exports tracing metrics using OpenTelemetry.
How: Uses opentelemetry-api and otlp exporter for traces.
Why: Enables performance monitoring, per Radical Reliability.
Full Path: lola-os/python/lola/utils/telemetry.py
"""

def setup_telemetry(endpoint: str = "http://localhost:4317") -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        endpoint: OTLP exporter endpoint (default: local gRPC).

    Does not:
        Collect metrics beyond traces.
    """
    try:
        trace.set_tracer_provider(TracerProvider())
        exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        logger.info(f"Telemetry initialized with endpoint {endpoint}")
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        raise
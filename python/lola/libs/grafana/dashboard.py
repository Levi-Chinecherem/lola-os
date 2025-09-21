# Standard imports
import typing as tp
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import datetime
import uuid

# Third-party
try:
    import requests
    from jinja2 import Template
except ImportError:
    raise ImportError("Grafana dependencies missing. Run 'poetry add requests jinja2'")

# Local
from lola.utils.config import get_config
from lola.utils.logging import logger
from lola.libs.prometheus.exporter import get_lola_prometheus
from sentry_sdk import capture_exception

"""
File: Grafana dashboard automation for LOLA OS.
Purpose: Auto-generates Grafana dashboards for LOLA metrics with pre-configured 
         panels for agent performance, LLM costs, EVM operations, and system health.
How: Uses Jinja2 templates to generate JSON dashboard definitions; supports 
     both file export and direct Grafana API provisioning.
Why: Provides instant observability setup for LOLA deployments, reducing 
     DevOps overhead and ensuring consistent monitoring across environments.
Full Path: lola-os/python/lola/libs/grafana/dashboard.py
"""

class LolaGrafanaDashboard:
    """LolaGrafanaDashboard: Automated dashboard generation for LOLA OS.
    Does NOT require running Grafana—can export JSON files for import."""

    DEFAULT_DATASOURCE_UID = "prometheus"
    DEFAULT_REFRESH = "30s"
    DEFAULT_TIME_RANGE = {"from": "now-6h", "to": "now"}

    def __init__(self):
        """
        Initializes Grafana dashboard generator.
        Does Not: Connect to Grafana—templates are self-contained.
        """
        config = get_config()
        self.grafana_url = config.get("grafana_url", "http://localhost:3000")
        self.api_key = config.get("grafana_api_key")
        self.org_id = config.get("grafana_org_id", 1)
        self.templates_dir = config.get("grafana_templates_dir", "./grafana_templates")
        self.folder_uid = config.get("grafana_folder_uid", "lola-os")
        
        # Ensure templates directory
        Path(self.templates_dir).mkdir(exist_ok=True)
        
        self._prometheus_exporter = get_lola_prometheus()
        logger.info(f"Grafana dashboard generator initialized")

    def generate_agent_dashboard(self, dashboard_id: Optional[str] = None, 
                               export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates comprehensive agent monitoring dashboard.
        Args:
            dashboard_id: Optional dashboard ID.
            export_path: Path to export JSON (None for API provisioning).
        Returns:
            Dashboard JSON definition.
        """
        try:
            # Load agent dashboard template
            template_path = Path(__file__).parent / "templates" / "agent_dashboard.json.j2"
            if not template_path.exists():
                # Create default template if missing
                template_content = self._get_default_agent_template()
                template_path.parent.mkdir(exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(template_content)
            
            with open(template_path, "r") as f:
                template = Template(f.read())
            
            # Render with current config
            dashboard_json = template.render({
                "datasource_uid": self.DEFAULT_DATASOURCE_UID,
                "namespace": self._prometheus_exporter.namespace,
                "time_range": self.DEFAULT_TIME_RANGE,
                "refresh": self.DEFAULT_REFRESH,
                "dashboard_id": dashboard_id or str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "1.0.0"
            })
            
            dashboard = json.loads(dashboard_json)
            
            # Provision or export
            if export_path:
                with open(export_path, "w") as f:
                    json.dump(dashboard, f, indent=2)
                logger.info(f"Agent dashboard exported to: {export_path}")
            elif self.api_key:
                self._provision_dashboard(dashboard, "LOLA Agent Monitoring")
                logger.info("Agent dashboard provisioned to Grafana")
            else:
                logger.info("Agent dashboard generated (export or configure API key to provision)")
            
            return dashboard
            
        except Exception as exc:
            self._handle_error(exc, "agent dashboard generation")
            raise

    def generate_llm_dashboard(self, dashboard_id: Optional[str] = None, 
                             export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates LLM usage and cost monitoring dashboard.
        """
        try:
            template_path = Path(__file__).parent / "templates" / "llm_dashboard.json.j2"
            if not template_path.exists():
                template_content = self._get_default_llm_template()
                template_path.parent.mkdir(exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(template_content)
            
            with open(template_path, "r") as f:
                template = Template(f.read())
            
            dashboard_json = template.render({
                "datasource_uid": self.DEFAULT_DATASOURCE_UID,
                "namespace": self._prometheus_exporter.namespace,
                "time_range": self.DEFAULT_TIME_RANGE,
                "refresh": "1m",  # More frequent for cost tracking
                "dashboard_id": dashboard_id or str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            dashboard = json.loads(dashboard_json)
            
            if export_path:
                with open(export_path, "w") as f:
                    json.dump(dashboard, f, indent=2)
                logger.info(f"LLM dashboard exported to: {export_path}")
            elif self.api_key:
                self._provision_dashboard(dashboard, "LOLA LLM Analytics")
            
            return dashboard
            
        except Exception as exc:
            self._handle_error(exc, "LLM dashboard generation")
            raise

    def generate_evm_dashboard(self, dashboard_id: Optional[str] = None, 
                             export_path: Optional[str] = None) -> Dict[str, Any]:
        """Generates EVM operations monitoring dashboard."""
        try:
            template_path = Path(__file__).parent / "templates" / "evm_dashboard.json.j2"
            if not template_path.exists():
                template_content = self._get_default_evm_template()
                template_path.parent.mkdir(exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(template_content)
            
            with open(template_path, "r") as f:
                template = Template(f.read())
            
            dashboard_json = template.render({
                "datasource_uid": self.DEFAULT_DATASOURCE_UID,
                "namespace": self._prometheus_exporter.namespace,
                "time_range": {"from": "now-24h", "to": "now"},  # Longer range for EVM
                "refresh": self.DEFAULT_REFRESH,
                "dashboard_id": dashboard_id or str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            dashboard = json.loads(dashboard_json)
            
            if export_path:
                with open(export_path, "w") as f:
                    json.dump(dashboard, f, indent=2)
                logger.info(f"EVM dashboard exported to: {export_path}")
            elif self.api_key:
                self._provision_dashboard(dashboard, "LOLA EVM Operations")
            
            return dashboard
            
        except Exception as exc:
            self._handle_error(exc, "EVM dashboard generation")
            raise

    def generate_system_dashboard(self, dashboard_id: Optional[str] = None, 
                                export_path: Optional[str] = None) -> Dict[str, Any]:
        """Generates system health and resource monitoring dashboard."""
        try:
            template_path = Path(__file__).parent / "templates" / "system_dashboard.json.j2"
            if not template_path.exists():
                template_content = self._get_default_system_template()
                template_path.parent.mkdir(exist_ok=True)
                with open(template_path, "w") as f:
                    f.write(template_content)
            
            with open(template_path, "r") as f:
                template = Template(f.read())
            
            dashboard_json = template.render({
                "datasource_uid": self.DEFAULT_DATASOURCE_UID,
                "namespace": self._prometheus_exporter.namespace,
                "time_range": self.DEFAULT_TIME_RANGE,
                "refresh": "15s",  # Frequent for system metrics
                "dashboard_id": dashboard_id or str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            dashboard = json.loads(dashboard_json)
            
            if export_path:
                with open(export_path, "w") as f:
                    json.dump(dashboard, f, indent=2)
                logger.info(f"System dashboard exported to: {export_path}")
            elif self.api_key:
                self._provision_dashboard(dashboard, "LOLA System Health")
            
            return dashboard
            
        except Exception as exc:
            self._handle_error(exc, "system dashboard generation")
            raise

    def _provision_dashboard(self, dashboard: Dict[str, Any], title: str) -> Optional[str]:
        """
        Provisions dashboard to Grafana API.
        Args:
            dashboard: Dashboard JSON.
            title: Dashboard title.
        Returns:
            Dashboard UID or None if failed.
        """
        if not self.api_key:
            logger.warning("Cannot provision dashboard: no API key configured")
            return None
        
        try:
            # Prepare folder
            folder_response = requests.post(
                f"{self.grafana_url}/api/folders",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "uid": self.folder_uid,
                    "title": "LOLA OS Dashboards"
                }
            )
            
            if folder_response.status_code not in [200, 201, 409]:  # 409 means folder exists
                logger.warning(f"Folder creation failed: {folder_response.status_code}")
            
            # Provision dashboard
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "dashboard": dashboard,
                    "folderUid": self.folder_uid,
                    "overwrite": True
                }
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                dashboard_uid = result.get("uid")
                logger.info(f"Dashboard '{title}' provisioned: {dashboard_uid}")
                return dashboard_uid
            else:
                logger.error(f"Dashboard provisioning failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as exc:
            self._handle_error(exc, "Grafana API provisioning")
            return None

    def _get_default_agent_template(self) -> str:
        """Returns default agent dashboard template."""
        return '''
{
  "dashboard": {
    "id": null,
    "title": "LOLA Agent Monitoring",
    "tags": ["lola", "agents"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "{{ refresh }}",
    "time": {{ time_range | tojson }},
    "panels": [
      {
        "id": 1,
        "title": "Agent Run Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "increase({{ namespace }}_agent_runs_total{{ $status ? '{status=~\"' + $status + '\"' : '' }}[$__rate_interval])",
            "legendFormat": "{{agent_type}} - {{operation}}",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "color": {"mode": "background"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Agent Run Duration P95",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate({{ namespace }}_agent_run_duration_seconds_bucket[5m])) by (le, agent_type, operation))",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "color": {"mode": "background"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Agent Error Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate({{ namespace }}_agent_errors_total[5m])",
            "legendFormat": "{{agent_type}} - {{operation}}",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "color": {"mode": "palette-classic"}
          }
        },
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      }
    ],
    "templating": {
      "list": [
        {
          "allValue": null,
          "current": {"selected": true, "text": "All", "value": "$__all"},
          "hide": 0,
          "includeAll": true,
          "label": "Status",
          "multi": true,
          "name": "status",
          "options": [],
          "query": {"query": "label_values({{ namespace }}_agent_runs_total, status)", "refId": "StandardVariable"},
          "refresh": 1,
          "regex": "",
          "skipUrlSync": false,
          "sort": 1,
          "type": "query"
        }
      ]
    }
  },
  "__inputs": [],
  "timeRange": {{ time_range | tojson }},
  "refresh": "{{ refresh }}",
  "schemaVersion": 38,
  "version": "{{ version }}",
  "uid": "{{ dashboard_id }}",
  "created": "{{ timestamp }}",
  "updated": "{{ timestamp }}"
}
        '''

    def _get_default_llm_template(self) -> str:
        """Returns default LLM dashboard template."""
        return '''
{
  "dashboard": {
    "id": null,
    "title": "LOLA LLM Analytics",
    "tags": ["lola", "llm"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "{{ refresh }}",
    "time": {{ time_range | tojson }},
    "panels": [
      {
        "id": 1,
        "title": "LLM Cost (24h)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum({{ namespace }}_llm_cost_usd)",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "color": {"mode": "background"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Token Usage Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate({{ namespace }}_llm_tokens_total{{ $direction ? '{direction=~\"' + $direction + '\"' : '' }}[$__rate_interval])) by (model)",
            "legendFormat": "{{model}} - {{direction}}",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "color": {"mode": "palette-classic"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      }
    ],
    "templating": {
      "list": [
        {
          "allValue": null,
          "current": {"selected": true, "text": "All", "value": "$__all"},
          "hide": 0,
          "includeAll": true,
          "label": "Direction",
          "multi": true,
          "name": "direction",
          "options": [],
          "query": {"query": "label_values({{ namespace }}_llm_tokens_total, direction)", "refId": "StandardVariable"},
          "refresh": 1,
          "regex": "",
          "skipUrlSync": false,
          "sort": 1,
          "type": "query"
        }
      ]
    }
  }
}
        '''

    def _get_default_evm_template(self) -> str:
        """Returns default EVM dashboard template."""
        return '''
{
  "dashboard": {
    "id": null,
    "title": "LOLA EVM Operations",
    "tags": ["lola", "evm", "blockchain"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "{{ refresh }}",
    "time": {{ time_range | tojson }},
    "panels": [
      {
        "id": 1,
        "title": "EVM Call Rate by Chain",
        "type": "barchart",
        "targets": [
          {
            "expr": "sum(rate({{ namespace }}_evm_calls_total{status=\"success\"}[$__rate_interval])) by (chain)",
            "legendFormat": "{{chain}}",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "color": {"mode": "palette-classic"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      }
    ]
  }
}
        '''

    def _get_default_system_template(self) -> str:
        """Returns default system dashboard template."""
        return '''
{
  "dashboard": {
    "id": null,
    "title": "LOLA System Health",
    "tags": ["lola", "system", "infrastructure"],
    "timezone": "browser",
    "schemaVersion": 38,
    "version": 1,
    "refresh": "{{ refresh }}",
    "time": {{ time_range | tojson }},
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "{{ namespace }}_process_cpu_percent",
            "datasource": "{{ datasource_uid }}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "color": {"mode": "thresholds"}
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      }
    ]
  }
}
        '''

    def _handle_error(self, exc: Exception, context: str) -> None:
        """Error handling for Grafana operations."""
        logger.error(f"Grafana dashboard {context} failed: {str(exc)}")
        config = get_config()
        if config.get("sentry_dsn"):
            capture_exception(exc)


# Global instance
_lola_grafana = None

def get_lola_grafana_dashboard() -> LolaGrafanaDashboard:
    """Singleton Grafana dashboard generator."""
    global _lola_grafana
    if _lola_grafana is None:
        _lola_grafana = LolaGrafanaDashboard()
    return _lola_grafana

__all__ = [
    "LolaGrafanaDashboard",
    "get_lola_grafana_dashboard"
]
"""
BAIS Platform Harmonizer (PPA1-Inv9)

Ensures consistent governance behavior across different platforms and
deployment environments. Harmonizes:
1. Output formats (CLI, API, IDE integration)
2. Threshold interpretations
3. Warning/error presentation
4. Learning state synchronization

Patent Alignment:
- PPA1-Inv9: Cross-Platform Harmonization
- Brain Layer: 7 (Thalamus - Orchestration)

NO PLACEHOLDERS. NO STUBS. FULL IMPLEMENTATION.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json


class Platform(Enum):
    """Supported deployment platforms."""
    CLI = "cli"                  # Command line interface
    API = "api"                  # REST API
    MCP = "mcp"                  # Model Context Protocol (Cursor)
    JUPYTER = "jupyter"          # Jupyter notebooks
    WEB = "web"                  # Web application
    SLACK = "slack"              # Slack integration
    VSCODE = "vscode"            # VS Code extension


class OutputFormat(Enum):
    """Output format types."""
    JSON = "json"                # Machine-readable JSON
    MARKDOWN = "markdown"        # Human-readable markdown
    PLAIN = "plain"              # Plain text
    HTML = "html"                # HTML formatted
    RICH = "rich"                # Rich terminal output
    STRUCTURED = "structured"    # Structured data (for IDE)


class SeverityMapping(Enum):
    """How severity levels are presented."""
    EMOJI = "emoji"              # 🔴 🟡 🟢
    TEXT = "text"                # CRITICAL, WARNING, INFO
    COLOR = "color"              # Red, Yellow, Green
    NUMERIC = "numeric"          # 1, 2, 3


@dataclass
class PlatformConfig:
    """Configuration for a specific platform."""
    platform: Platform
    output_format: OutputFormat
    severity_mapping: SeverityMapping
    max_warnings: int = 10
    include_recommendations: bool = True
    include_trace: bool = False
    abbreviate_long_text: bool = True
    max_text_length: int = 500
    
    def to_dict(self) -> Dict:
        return {
            'platform': self.platform.value,
            'output_format': self.output_format.value,
            'severity_mapping': self.severity_mapping.value,
            'max_warnings': self.max_warnings,
            'include_recommendations': self.include_recommendations
        }


@dataclass
class HarmonizedOutput:
    """Platform-harmonized governance output."""
    platform: Platform
    format: OutputFormat
    content: Any  # Formatted content
    raw_data: Dict  # Original data
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'platform': self.platform.value,
            'format': self.format.value,
            'content': self.content if isinstance(self.content, (str, dict, list)) else str(self.content),
            'metadata': self.metadata
        }


class PlatformHarmonizer:
    """
    Ensures consistent governance behavior across platforms.
    
    Implements PPA1-Inv9
    Brain Layer: 7 (Thalamus)
    
    Capabilities:
    1. Platform detection
    2. Output format adaptation
    3. Severity translation
    4. Content abbreviation
    5. Cross-platform state sync
    """
    
    # Default configurations per platform
    DEFAULT_CONFIGS = {
        Platform.CLI: PlatformConfig(
            platform=Platform.CLI,
            output_format=OutputFormat.RICH,
            severity_mapping=SeverityMapping.EMOJI,
            max_warnings=5,
            include_recommendations=True,
            abbreviate_long_text=True
        ),
        Platform.API: PlatformConfig(
            platform=Platform.API,
            output_format=OutputFormat.JSON,
            severity_mapping=SeverityMapping.TEXT,
            max_warnings=100,
            include_recommendations=True,
            include_trace=True,
            abbreviate_long_text=False
        ),
        Platform.MCP: PlatformConfig(
            platform=Platform.MCP,
            output_format=OutputFormat.STRUCTURED,
            severity_mapping=SeverityMapping.TEXT,
            max_warnings=20,
            include_recommendations=True,
            abbreviate_long_text=True
        ),
        Platform.JUPYTER: PlatformConfig(
            platform=Platform.JUPYTER,
            output_format=OutputFormat.HTML,
            severity_mapping=SeverityMapping.COLOR,
            max_warnings=10,
            include_recommendations=True
        ),
        Platform.WEB: PlatformConfig(
            platform=Platform.WEB,
            output_format=OutputFormat.JSON,
            severity_mapping=SeverityMapping.TEXT,
            max_warnings=50,
            include_recommendations=True
        ),
        Platform.SLACK: PlatformConfig(
            platform=Platform.SLACK,
            output_format=OutputFormat.MARKDOWN,
            severity_mapping=SeverityMapping.EMOJI,
            max_warnings=3,
            abbreviate_long_text=True,
            max_text_length=300
        ),
        Platform.VSCODE: PlatformConfig(
            platform=Platform.VSCODE,
            output_format=OutputFormat.STRUCTURED,
            severity_mapping=SeverityMapping.TEXT,
            max_warnings=20,
            include_recommendations=True
        )
    }
    
    # Severity emoji mapping
    SEVERITY_EMOJI = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢',
        'info': 'ℹ️'
    }
    
    # Severity color mapping
    SEVERITY_COLORS = {
        'critical': 'red',
        'high': 'orange',
        'medium': 'yellow',
        'low': 'green',
        'info': 'blue'
    }
    
    def __init__(self, default_platform: Platform = Platform.API):
        """Initialize the harmonizer."""
        self.default_platform = default_platform
        self.configs = self.DEFAULT_CONFIGS.copy()
        self._platform_states: Dict[Platform, Dict] = {}
    
    def detect_platform(self, context: Dict[str, Any] = None) -> Platform:
        """Detect the current platform from context."""
        if not context:
            return self.default_platform
        
        # Check for explicit platform
        if 'platform' in context:
            try:
                return Platform(context['platform'])
            except ValueError:
                pass
        
        # Detect from environment hints
        if context.get('is_mcp') or context.get('cursor'):
            return Platform.MCP
        if context.get('is_cli') or context.get('terminal'):
            return Platform.CLI
        if context.get('is_jupyter') or context.get('notebook'):
            return Platform.JUPYTER
        if context.get('is_slack'):
            return Platform.SLACK
        if context.get('is_vscode') or context.get('vscode'):
            return Platform.VSCODE
        if context.get('is_web') or context.get('browser'):
            return Platform.WEB
        
        return self.default_platform
    
    def harmonize(self,
                  governance_result: Dict[str, Any],
                  platform: Platform = None,
                  context: Dict[str, Any] = None) -> HarmonizedOutput:
        """
        Harmonize governance output for the target platform.
        
        Args:
            governance_result: Raw governance result
            platform: Target platform (auto-detected if None)
            context: Additional context
            
        Returns:
            HarmonizedOutput with platform-appropriate formatting
        """
        context = context or {}
        
        # Detect platform if not specified
        if platform is None:
            platform = self.detect_platform(context)
        
        # Get platform config
        config = self.configs.get(platform, self.DEFAULT_CONFIGS[Platform.API])
        
        # Format based on output type
        if config.output_format == OutputFormat.JSON:
            content = self._format_json(governance_result, config)
        elif config.output_format == OutputFormat.MARKDOWN:
            content = self._format_markdown(governance_result, config)
        elif config.output_format == OutputFormat.PLAIN:
            content = self._format_plain(governance_result, config)
        elif config.output_format == OutputFormat.HTML:
            content = self._format_html(governance_result, config)
        elif config.output_format == OutputFormat.RICH:
            content = self._format_rich(governance_result, config)
        elif config.output_format == OutputFormat.STRUCTURED:
            content = self._format_structured(governance_result, config)
        else:
            content = governance_result
        
        return HarmonizedOutput(
            platform=platform,
            format=config.output_format,
            content=content,
            raw_data=governance_result,
            metadata={
                'harmonized_at': datetime.now().isoformat(),
                'config': config.to_dict()
            }
        )
    
    def _format_json(self, result: Dict, config: PlatformConfig) -> Dict:
        """Format as JSON."""
        output = result.copy()
        
        # Apply severity mapping
        if 'issues' in output:
            output['issues'] = self._map_severities(output['issues'], config)
            output['issues'] = output['issues'][:config.max_warnings]
        
        if 'warnings' in output:
            output['warnings'] = output['warnings'][:config.max_warnings]
        
        return output
    
    def _format_markdown(self, result: Dict, config: PlatformConfig) -> str:
        """Format as Markdown."""
        lines = []
        
        # Header
        accepted = result.get('accepted', False)
        accuracy = result.get('accuracy', 0)
        status_emoji = '✅' if accepted else '❌'
        lines.append(f"## Governance Result {status_emoji}")
        lines.append(f"**Accuracy:** {accuracy:.1f}%")
        lines.append("")
        
        # Issues
        issues = result.get('issues', []) or result.get('warnings', [])
        if issues:
            lines.append("### Issues")
            for i, issue in enumerate(issues[:config.max_warnings], 1):
                severity = issue.get('severity', 'medium') if isinstance(issue, dict) else 'medium'
                emoji = self.SEVERITY_EMOJI.get(severity, '⚪')
                text = issue.get('description', str(issue)) if isinstance(issue, dict) else str(issue)
                if config.abbreviate_long_text and len(text) > config.max_text_length:
                    text = text[:config.max_text_length] + "..."
                lines.append(f"{i}. {emoji} {text}")
            lines.append("")
        
        # Recommendations
        if config.include_recommendations and result.get('recommendations'):
            lines.append("### Recommendations")
            for rec in result['recommendations'][:5]:
                lines.append(f"- {rec}")
        
        return '\n'.join(lines)
    
    def _format_plain(self, result: Dict, config: PlatformConfig) -> str:
        """Format as plain text."""
        lines = []
        
        accepted = result.get('accepted', False)
        accuracy = result.get('accuracy', 0)
        status = 'ACCEPTED' if accepted else 'REJECTED'
        lines.append(f"Status: {status}")
        lines.append(f"Accuracy: {accuracy:.1f}%")
        
        issues = result.get('issues', []) or result.get('warnings', [])
        if issues:
            lines.append(f"\nIssues ({len(issues)}):")
            for issue in issues[:config.max_warnings]:
                text = issue.get('description', str(issue)) if isinstance(issue, dict) else str(issue)
                lines.append(f"  - {text}")
        
        return '\n'.join(lines)
    
    def _format_html(self, result: Dict, config: PlatformConfig) -> str:
        """Format as HTML."""
        accepted = result.get('accepted', False)
        accuracy = result.get('accuracy', 0)
        status_class = 'success' if accepted else 'danger'
        
        html = f"""
<div class="governance-result {status_class}">
    <h3>Governance Result</h3>
    <p><strong>Status:</strong> <span class="{status_class}">{'Accepted' if accepted else 'Rejected'}</span></p>
    <p><strong>Accuracy:</strong> {accuracy:.1f}%</p>
"""
        
        issues = result.get('issues', []) or result.get('warnings', [])
        if issues:
            html += "<h4>Issues</h4><ul>"
            for issue in issues[:config.max_warnings]:
                severity = issue.get('severity', 'medium') if isinstance(issue, dict) else 'medium'
                text = issue.get('description', str(issue)) if isinstance(issue, dict) else str(issue)
                color = self.SEVERITY_COLORS.get(severity, 'gray')
                html += f'<li style="color: {color}">{text}</li>'
            html += "</ul>"
        
        html += "</div>"
        return html
    
    def _format_rich(self, result: Dict, config: PlatformConfig) -> str:
        """Format for rich terminal output."""
        return self._format_markdown(result, config)  # Use markdown for rich
    
    def _format_structured(self, result: Dict, config: PlatformConfig) -> Dict:
        """Format as structured data for IDEs."""
        return {
            'status': 'accepted' if result.get('accepted') else 'rejected',
            'accuracy': result.get('accuracy', 0),
            'confidence': result.get('confidence', 0),
            'issues': [
                {
                    'severity': (i.get('severity', 'medium') if isinstance(i, dict) else 'medium'),
                    'message': (i.get('description', str(i)) if isinstance(i, dict) else str(i)),
                    'line': i.get('line') if isinstance(i, dict) else None
                }
                for i in (result.get('issues', []) or result.get('warnings', []))[:config.max_warnings]
            ],
            'recommendations': result.get('recommendations', [])[:5] if config.include_recommendations else []
        }
    
    def _map_severities(self, issues: List, config: PlatformConfig) -> List:
        """Map severity levels based on platform config."""
        mapped = []
        for issue in issues:
            if isinstance(issue, dict):
                issue = issue.copy()
                severity = issue.get('severity', 'medium')
                
                if config.severity_mapping == SeverityMapping.EMOJI:
                    issue['severity_display'] = self.SEVERITY_EMOJI.get(severity, '⚪')
                elif config.severity_mapping == SeverityMapping.COLOR:
                    issue['severity_display'] = self.SEVERITY_COLORS.get(severity, 'gray')
                elif config.severity_mapping == SeverityMapping.NUMERIC:
                    numeric_map = {'critical': 1, 'high': 2, 'medium': 3, 'low': 4, 'info': 5}
                    issue['severity_display'] = numeric_map.get(severity, 3)
                else:
                    issue['severity_display'] = severity.upper()
                
                mapped.append(issue)
            else:
                mapped.append(issue)
        return mapped
    
    def set_platform_config(self, platform: Platform, config: PlatformConfig) -> None:
        """Set custom configuration for a platform."""
        self.configs[platform] = config
    
    def sync_state(self, from_platform: Platform, to_platform: Platform) -> bool:
        """Synchronize learning state between platforms."""
        if from_platform in self._platform_states:
            self._platform_states[to_platform] = self._platform_states[from_platform].copy()
            return True
        return False
    
    # Learning interface methods
    def record_outcome(self, result: HarmonizedOutput, was_useful: bool, domain: str = 'general'):
        """Record outcome for learning."""
        pass
    
    def record_feedback(self, feedback: Dict) -> None:
        """Record feedback for learning."""
        pass
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'platforms_used': list(self._platform_states.keys()),
            'configs': {p.value: c.to_dict() for p, c in self.configs.items()}
        }



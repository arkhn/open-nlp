"""
Custom exceptions for the conflicts module.
Provides standardized error handling across all agents.
"""


class AgentError(Exception):
    """Base exception for all agent-related errors"""

    pass


class DoctorAgentError(AgentError):
    """Raised when Doctor Agent fails to analyze documents or choose conflict type"""

    pass


class EditorAgentError(AgentError):
    """Raised when Editor Agent fails to create modifications"""

    pass


class ModeratorAgentError(AgentError):
    """Raised when Moderator Agent fails to validate modifications"""

    pass


class PropositionAgentError(AgentError):
    """Raised when Proposition Agent fails to decompose documents"""

    pass


class ConfigurationError(Exception):
    """Raised when there are configuration issues"""

    pass


class ValidationError(Exception):
    """Raised when validation fails"""

    pass

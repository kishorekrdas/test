"""
Exception Hierarchy and Error Handling Framework

Production-grade exception system for clinical genomic pipelines.
Implements typed exceptions, error recovery strategies, and clinical safety checks.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import json


# ============================================================================
# ERROR SEVERITY LEVELS
# ============================================================================

class ErrorSeverity(Enum):
    """
    Error severity classification for clinical safety.
    
    CRITICAL: Patient safety impact - halt all processing
    HIGH: Clinical impact - halt sample processing, alert clinical team
    MEDIUM: Quality impact - log, continue with warnings
    LOW: Informational - log only
    """
    CRITICAL = "CRITICAL"  # Patient safety risk
    HIGH = "HIGH"          # Clinical impact
    MEDIUM = "MEDIUM"      # Quality/performance impact
    LOW = "LOW"            # Informational


class ErrorCategory(Enum):
    """Error categories for structured handling."""
    CONFIGURATION = "CONFIGURATION"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    RESOURCE = "RESOURCE"
    PIPELINE = "PIPELINE"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    VALIDATION = "VALIDATION"
    CLINICAL_LOGIC = "CLINICAL_LOGIC"
    AUDIT = "AUDIT"
    NETWORK = "NETWORK"
    PERMISSION = "PERMISSION"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    ABORT = "ABORT"                    # Stop immediately
    RETRY = "RETRY"                    # Retry with backoff
    SKIP = "SKIP"                      # Skip and continue
    FALLBACK = "FALLBACK"              # Use alternative method
    CHECKPOINT_RESUME = "CHECKPOINT_RESUME"  # Resume from checkpoint
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"  # Require human review


# ============================================================================
# BASE EXCEPTION CLASSES
# ============================================================================

@dataclass
class ErrorContext:
    """
    Structured context for error reporting and recovery.
    """
    error_id: str
    timestamp: str
    severity: ErrorSeverity
    category: ErrorCategory
    recovery_strategy: RecoveryStrategy
    sample_id: Optional[str] = None
    stage: Optional[str] = None
    component: Optional[str] = None
    file_path: Optional[Path] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'recovery_strategy': self.recovery_strategy.value,
            'sample_id': self.sample_id,
            'stage': self.stage,
            'component': self.component,
            'file_path': str(self.file_path) if self.file_path else None,
            'command': self.command,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'metadata': self.metadata,
            'stack_trace': self.stack_trace
        }
        return {k: v for k, v in result.items() if v is not None}
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class GenomicPipelineException(Exception):
    """
    Base exception class for all genomic pipeline errors.
    
    All custom exceptions inherit from this to enable:
    - Structured error context
    - Severity classification
    - Recovery strategy hints
    - Clinical safety checks
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        category: ErrorCategory = ErrorCategory.PIPELINE,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT,
        **context_kwargs
    ):
        """
        Initialize exception with structured context.
        
        Args:
            message: Human-readable error message
            severity: Error severity level
            category: Error category
            recovery_strategy: Suggested recovery strategy
            **context_kwargs: Additional context fields
        """
        super().__init__(message)
        
        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        
        # Generate unique error ID
        error_id = self._generate_error_id()
        
        # Build error context
        self.context = ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            category=category,
            recovery_strategy=recovery_strategy,
            stack_trace=self._capture_stack_trace(),
            **context_kwargs
        )
    
    def _generate_error_id(self) -> str:
        """Generate unique error identifier."""
        from uuid import uuid4
        return f"ERR-{uuid4().hex[:8].upper()}"
    
    def _capture_stack_trace(self) -> str:
        """Capture current stack trace."""
        return ''.join(traceback.format_stack()[:-1])
    
    def is_retryable(self) -> bool:
        """Check if error is retryable."""
        return self.recovery_strategy in [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.CHECKPOINT_RESUME
        ]
    
    def is_critical(self) -> bool:
        """Check if error is critical (patient safety)."""
        return self.severity == ErrorSeverity.CRITICAL
    
    def requires_clinical_review(self) -> bool:
        """Check if error requires clinical review."""
        return self.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
    
    def __str__(self) -> str:
        """Human-readable error representation."""
        return (
            f"[{self.severity.value}] {self.category.value}: {self.message}\n"
            f"Error ID: {self.context.error_id}\n"
            f"Recovery: {self.recovery_strategy.value}"
        )


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(GenomicPipelineException):
    """Base class for configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class InvalidConfigurationError(ConfigurationError):
    """Configuration validation failed."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration parameter missing."""
    pass


class ConfigurationVersionMismatchError(ConfigurationError):
    """Configuration version incompatible with pipeline version."""
    pass


# ============================================================================
# DATA INTEGRITY EXCEPTIONS
# ============================================================================

class DataIntegrityError(GenomicPipelineException):
    """Base class for data integrity errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.DATA_INTEGRITY,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class ChecksumMismatchError(DataIntegrityError):
    """File checksum does not match expected value."""
    
    def __init__(self, file_path: Path, expected: str, actual: str, **kwargs):
        message = (
            f"Checksum mismatch for {file_path.name}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )
        super().__init__(
            message=message,
            file_path=file_path,
            metadata={'expected_checksum': expected, 'actual_checksum': actual},
            **kwargs
        )


class CorruptedFileError(DataIntegrityError):
    """File is corrupted or unreadable."""
    
    def __init__(self, file_path: Path, reason: str = "", **kwargs):
        message = f"Corrupted file: {file_path}"
        if reason:
            message += f"\nReason: {reason}"
        super().__init__(message=message, file_path=file_path, **kwargs)


class MissingDataError(DataIntegrityError):
    """Required data file or field is missing."""
    
    def __init__(self, data_type: str, identifier: str, **kwargs):
        message = f"Missing required {data_type}: {identifier}"
        super().__init__(
            message=message,
            metadata={'data_type': data_type, 'identifier': identifier},
            **kwargs
        )


class InvalidFormatError(DataIntegrityError):
    """File format is invalid or unrecognized."""
    
    def __init__(self, file_path: Path, expected_format: str, **kwargs):
        message = f"Invalid format for {file_path.name}, expected {expected_format}"
        super().__init__(
            message=message,
            file_path=file_path,
            metadata={'expected_format': expected_format},
            **kwargs
        )


# ============================================================================
# RESOURCE EXCEPTIONS
# ============================================================================

class ResourceError(GenomicPipelineException):
    """Base class for resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RESOURCE,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class InsufficientResourcesError(ResourceError):
    """Insufficient system resources available."""
    
    def __init__(self, resource_type: str, required: float, available: float, **kwargs):
        message = (
            f"Insufficient {resource_type}: "
            f"required {required}, available {available}"
        )
        super().__init__(
            message=message,
            metadata={
                'resource_type': resource_type,
                'required': required,
                'available': available
            },
            **kwargs
        )


class DiskSpaceError(ResourceError):
    """Insufficient disk space."""
    
    def __init__(self, path: Path, required_gb: float, available_gb: float, **kwargs):
        message = (
            f"Insufficient disk space at {path}\n"
            f"Required: {required_gb:.1f} GB\n"
            f"Available: {available_gb:.1f} GB"
        )
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            file_path=path,
            metadata={'required_gb': required_gb, 'available_gb': available_gb},
            **kwargs
        )


class MemoryExhaustedError(ResourceError):
    """System memory exhausted."""
    
    def __init__(self, process: str, **kwargs):
        message = f"Memory exhausted during {process}"
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            component=process,
            **kwargs
        )


class GPUError(ResourceError):
    """GPU-related error."""
    
    def __init__(self, gpu_id: int, reason: str, **kwargs):
        message = f"GPU {gpu_id} error: {reason}"
        super().__init__(
            message=message,
            metadata={'gpu_id': gpu_id, 'reason': reason},
            **kwargs
        )


# ============================================================================
# PIPELINE EXECUTION EXCEPTIONS
# ============================================================================

class PipelineExecutionError(GenomicPipelineException):
    """Base class for pipeline execution errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PIPELINE,
            recovery_strategy=RecoveryStrategy.CHECKPOINT_RESUME,
            **kwargs
        )


class StageFailureError(PipelineExecutionError):
    """Pipeline stage failed."""
    
    def __init__(self, stage_name: str, reason: str, **kwargs):
        message = f"Stage '{stage_name}' failed: {reason}"
        super().__init__(message=message, stage=stage_name, **kwargs)


class DependencyFailureError(PipelineExecutionError):
    """Task dependency failed, cannot proceed."""
    
    def __init__(self, task_id: str, failed_dependency: str, **kwargs):
        message = f"Task '{task_id}' cannot execute: dependency '{failed_dependency}' failed"
        super().__init__(
            message=message,
            recovery_strategy=RecoveryStrategy.ABORT,
            metadata={'task_id': task_id, 'failed_dependency': failed_dependency},
            **kwargs
        )


class CheckpointCorruptedError(PipelineExecutionError):
    """Checkpoint file is corrupted."""
    
    def __init__(self, checkpoint_path: Path, **kwargs):
        message = f"Checkpoint corrupted: {checkpoint_path}"
        super().__init__(
            message=message,
            file_path=checkpoint_path,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class TimeoutError(PipelineExecutionError):
    """Operation exceeded timeout."""
    
    def __init__(self, operation: str, timeout_hours: float, **kwargs):
        message = f"Operation '{operation}' exceeded timeout of {timeout_hours} hours"
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            component=operation,
            metadata={'timeout_hours': timeout_hours},
            **kwargs
        )


# ============================================================================
# TOOL EXECUTION EXCEPTIONS
# ============================================================================

class ToolExecutionError(GenomicPipelineException):
    """Base class for external tool execution errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TOOL_EXECUTION,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ToolNotFoundError(ToolExecutionError):
    """External tool executable not found."""
    
    def __init__(self, tool_name: str, expected_path: Path, **kwargs):
        message = f"Tool '{tool_name}' not found at {expected_path}"
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.ABORT,
            component=tool_name,
            file_path=expected_path,
            **kwargs
        )


class ToolVersionMismatchError(ToolExecutionError):
    """Tool version incompatible with pipeline."""
    
    def __init__(self, tool_name: str, expected: str, actual: str, **kwargs):
        message = (
            f"Tool '{tool_name}' version mismatch\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )
        super().__init__(
            message=message,
            recovery_strategy=RecoveryStrategy.ABORT,
            component=tool_name,
            metadata={'expected_version': expected, 'actual_version': actual},
            **kwargs
        )


class ToolCrashError(ToolExecutionError):
    """External tool crashed during execution."""
    
    def __init__(
        self,
        tool_name: str,
        command: str,
        exit_code: int,
        stderr: Optional[str] = None,
        **kwargs
    ):
        message = f"Tool '{tool_name}' crashed with exit code {exit_code}"
        if stderr:
            # Include last 500 chars of stderr
            stderr_preview = stderr[-500:] if len(stderr) > 500 else stderr
            message += f"\nStderr: {stderr_preview}"
        
        super().__init__(
            message=message,
            component=tool_name,
            command=command,
            exit_code=exit_code,
            stderr=stderr,
            **kwargs
        )


class ToolOutputError(ToolExecutionError):
    """Tool produced invalid or unexpected output."""
    
    def __init__(self, tool_name: str, output_file: Path, reason: str, **kwargs):
        message = f"Tool '{tool_name}' produced invalid output: {reason}"
        super().__init__(
            message=message,
            component=tool_name,
            file_path=output_file,
            metadata={'reason': reason},
            **kwargs
        )


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(GenomicPipelineException):
    """Base class for validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class QualityControlFailureError(ValidationError):
    """Sample failed quality control checks."""
    
    def __init__(
        self,
        sample_id: str,
        failed_metrics: List[str],
        metric_values: Dict[str, float],
        **kwargs
    ):
        message = (
            f"Sample '{sample_id}' failed QC\n"
            f"Failed metrics: {', '.join(failed_metrics)}"
        )
        super().__init__(
            message=message,
            sample_id=sample_id,
            metadata={
                'failed_metrics': failed_metrics,
                'metric_values': metric_values
            },
            **kwargs
        )


class InvalidVariantError(ValidationError):
    """Variant record is invalid."""
    
    def __init__(self, variant_id: str, reason: str, **kwargs):
        message = f"Invalid variant '{variant_id}': {reason}"
        super().__init__(
            message=message,
            metadata={'variant_id': variant_id, 'reason': reason},
            **kwargs
        )


class ReferenceGenomeMismatchError(ValidationError):
    """Reference genome version mismatch."""
    
    def __init__(self, expected: str, actual: str, file_path: Path, **kwargs):
        message = (
            f"Reference genome mismatch in {file_path.name}\n"
            f"Expected: {expected}\n"
            f"Actual: {actual}"
        )
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            file_path=file_path,
            metadata={'expected_reference': expected, 'actual_reference': actual},
            **kwargs
        )


# ============================================================================
# CLINICAL LOGIC EXCEPTIONS
# ============================================================================

class ClinicalLogicError(GenomicPipelineException):
    """Base class for clinical interpretation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CLINICAL_LOGIC,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            **kwargs
        )


class ClassificationConflictError(ClinicalLogicError):
    """Conflicting variant classifications."""
    
    def __init__(
        self,
        variant_id: str,
        classifications: List[str],
        sources: List[str],
        **kwargs
    ):
        message = (
            f"Classification conflict for variant '{variant_id}'\n"
            f"Classifications: {', '.join(classifications)}\n"
            f"Sources: {', '.join(sources)}"
        )
        super().__init__(
            message=message,
            metadata={
                'variant_id': variant_id,
                'classifications': classifications,
                'sources': sources
            },
            **kwargs
        )


class InsufficientEvidenceError(ClinicalLogicError):
    """Insufficient evidence for variant classification."""
    
    def __init__(self, variant_id: str, available_evidence: List[str], **kwargs):
        message = (
            f"Insufficient evidence to classify variant '{variant_id}'\n"
            f"Available evidence: {', '.join(available_evidence)}"
        )
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.SKIP,
            metadata={
                'variant_id': variant_id,
                'available_evidence': available_evidence
            },
            **kwargs
        )


class IncidentalFindingError(ClinicalLogicError):
    """Incidental finding requires clinical review."""
    
    def __init__(self, variant_id: str, gene: str, condition: str, **kwargs):
        message = (
            f"Incidental finding detected\n"
            f"Variant: {variant_id}\n"
            f"Gene: {gene}\n"
            f"Condition: {condition}"
        )
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            metadata={
                'variant_id': variant_id,
                'gene': gene,
                'condition': condition
            },
            **kwargs
        )


# ============================================================================
# AUDIT EXCEPTIONS
# ============================================================================

class AuditError(GenomicPipelineException):
    """Base class for audit trail errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.AUDIT,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs
        )


class AuditIntegrityError(AuditError):
    """Audit trail integrity compromised."""
    
    def __init__(self, audit_file: Path, invalid_entries: List[str], **kwargs):
        message = (
            f"Audit trail integrity compromised: {audit_file}\n"
            f"Invalid entries: {len(invalid_entries)}"
        )
        super().__init__(
            message=message,
            file_path=audit_file,
            metadata={'invalid_entries': invalid_entries},
            **kwargs
        )


class AuditWriteError(AuditError):
    """Failed to write audit log entry."""
    
    def __init__(self, reason: str, **kwargs):
        message = f"Failed to write audit log: {reason}"
        super().__init__(message=message, **kwargs)


# ============================================================================
# NETWORK & PERMISSION EXCEPTIONS
# ============================================================================

class NetworkError(GenomicPipelineException):
    """Network-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class PermissionError(GenomicPipelineException):
    """File or resource permission denied."""
    
    def __init__(self, resource: Path, operation: str, **kwargs):
        message = f"Permission denied: {operation} on {resource}"
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PERMISSION,
            recovery_strategy=RecoveryStrategy.ABORT,
            file_path=resource,
            metadata={'operation': operation},
            **kwargs
        )


# ============================================================================
# ERROR HANDLER
# ============================================================================

class ErrorHandler:
    """
    Central error handling and recovery coordinator.
    
    Features:
    - Error logging with structured context
    - Recovery strategy execution
    - Clinical safety checks
    - Error aggregation and reporting
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize error handler.
        
        Args:
            log_dir: Directory for error logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.errors_encountered: List[ErrorContext] = []
        
        import logging
        self.logger = logging.getLogger('genomic_pipeline.errors')
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handle exception with appropriate recovery strategy.
        
        Args:
            exception: Exception to handle
            context: Additional context information
        
        Returns:
            True if recovery successful, False otherwise
        """
        # Convert to GenomicPipelineException if needed
        if not isinstance(exception, GenomicPipelineException):
            exception = self._wrap_exception(exception, context or {})
        
        # Log exception
        self._log_exception(exception)
        
        # Store error context
        self.errors_encountered.append(exception.context)
        
        # Check clinical safety
        if exception.requires_clinical_review():
            self._trigger_clinical_alert(exception)
        
        # Execute recovery strategy
        return self._execute_recovery(exception)
    
    def _wrap_exception(
        self,
        exception: Exception,
        context: Dict[str, Any]
    ) -> GenomicPipelineException:
        """Wrap generic exception in GenomicPipelineException."""
        return GenomicPipelineException(
            message=str(exception),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PIPELINE,
            recovery_strategy=RecoveryStrategy.ABORT,
            **context
        )
    
    def _log_exception(self, exception: GenomicPipelineException):
        """Log exception to structured error log."""
        error_record = exception.context.to_dict()
        error_record['exception_type'] = type(exception).__name__
        error_record['exception_message'] = exception.message
        
        # Write to JSONL file
        with open(self.error_log_file, 'a') as f:
            f.write(json.dumps(error_record) + '\n')
        
        # Log to standard logger
        log_level = self._severity_to_log_level(exception.severity)
        self.logger.log(log_level, str(exception))
    
    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        import logging
        mapping = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO
        }
        return mapping.get(severity, logging.ERROR)
    
    def _trigger_clinical_alert(self, exception: GenomicPipelineException):
        """Trigger alert for clinical review."""
        alert_file = self.log_dir / f"clinical_alert_{exception.context.error_id}.json"
        
        alert_data = {
            'alert_type': 'CLINICAL_REVIEW_REQUIRED',
            'error_id': exception.context.error_id,
            'timestamp': exception.context.timestamp,
            'severity': exception.severity.value,
            'sample_id': exception.context.sample_id,
            'message': exception.message,
            'requires_action': True
        }
        
        with open(alert_file, 'w') as f:
            json.dumps(alert_data, f, indent=2)
        
        self.logger.critical(f"CLINICAL ALERT: {exception.context.error_id} - {exception.message}")
    
    def _execute_recovery(self, exception: GenomicPipelineException) -> bool:
        """
        Execute recovery strategy for exception.
        
        Args:
            exception: Exception to recover from
        
        Returns:
            True if recovery successful
        """
        strategy = exception.recovery_strategy
        
        if strategy == RecoveryStrategy.ABORT:
            self.logger.error(f"ABORT: {exception.context.error_id}")
            return False
        
        elif strategy == RecoveryStrategy.RETRY:
            self.logger.warning(f"RETRY: {exception.context.error_id}")
            return True  # Caller should implement retry logic
        
        elif strategy == RecoveryStrategy.SKIP:
            self.logger.warning(f"SKIP: {exception.context.error_id}")
            return True
        
        elif strategy == RecoveryStrategy.FALLBACK:
            self.logger.info(f"FALLBACK: {exception.context.error_id}")
            return True
        
        elif strategy == RecoveryStrategy.CHECKPOINT_RESUME:
            self.logger.info(f"CHECKPOINT_RESUME: {exception.context.error_id}")
            return True
        
        elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
            self.logger.critical(f"MANUAL_INTERVENTION: {exception.context.error_id}")
            return False
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Generate error summary statistics.
        
        Returns:
            Dictionary with error statistics
        """
        summary = {
            'total_errors': len(self.errors_encountered),
            'by_severity': {},
            'by_category': {},
            'critical_errors': [],
            'requires_clinical_review': []
        }
        
        for error in self.errors_encountered:
            # Count by severity
            severity = error.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by category
            category = error.category.value
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
            # Track critical errors
            if error.severity == ErrorSeverity.CRITICAL:
                summary['critical_errors'].append(error.error_id)
            
            # Track errors requiring review
            if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                summary['requires_clinical_review'].append(error.error_id)
        
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_exception_chain(exception: Exception) -> str:
    """
    Format exception chain for logging.
    
    Args:
        exception: Exception to format
    
    Returns:
        Formatted exception chain string
    """
    lines = []
    lines.append("Exception chain:")
    
    current = exception
    depth = 0
    while current is not None:
        indent = "  " * depth
        lines.append(f"{indent}{type(current).__name__}: {str(current)}")
        current = current.__cause__ if hasattr(current, '__cause__') else None
        depth += 1
    
    return '\n'.join(lines)


def safe_execute(
    func: Callable,
    error_handler: ErrorHandler,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    retry_exceptions: Optional[List[type]] = None
) -> Optional[Any]:
    """
    Execute function with automatic error handling and retry logic.
    
    Args:
        func: Function to execute
        error_handler: Error handler instance
        context: Additional context for error reporting
        max_retries: Maximum retry attempts
        retry_exceptions: Exception types to retry (None = retry all retryable)
    
    Returns:
        Function result or None if failed
    
    Example:
        result = safe_execute(
            lambda: run_gatk_command(args),
            error_handler,
            context={'sample_id': 'SAMPLE001', 'stage': 'variant_calling'},
            max_retries=3,
            retry_exceptions=[ToolCrashError, ResourceError]
        )
    """
    attempt = 0
    last_exception = None
    
    while attempt < max_retries:
        try:
            return func()
        
        except Exception as e:
            attempt += 1
            last_exception = e
            
            # Handle exception
            can_recover = error_handler.handle_exception(e, context)
            
            # Check if we should retry
            if isinstance(e, GenomicPipelineException):
                should_retry = e.is_retryable()
                
                # Check retry_exceptions filter
                if retry_exceptions and not isinstance(e, tuple(retry_exceptions)):
                    should_retry = False
            else:
                should_retry = can_recover
            
            if not should_retry or attempt >= max_retries:
                break
            
            # Exponential backoff
            import time
            backoff_seconds = 2 ** attempt
            time.sleep(backoff_seconds)
    
    # All retries exhausted
    error_handler.logger.error(
        f"safe_execute failed after {attempt} attempts: {last_exception}"
    )
    return None


def validate_clinical_safety(
    variant_classification: str,
    evidence_level: str,
    automated_confidence: float
) -> None:
    """
    Validate that automated classification meets clinical safety thresholds.
    
    Args:
        variant_classification: Predicted classification
        evidence_level: ACMG evidence level
        automated_confidence: Confidence score (0-1)
    
    Raises:
        ClinicalLogicError: If classification unsafe for automated reporting
    
    Example:
        validate_clinical_safety(
            variant_classification="Pathogenic",
            evidence_level="PVS1+PS1",
            automated_confidence=0.95
        )
    """
    # Pathogenic/Likely Pathogenic require high confidence
    if variant_classification in ["Pathogenic", "Likely Pathogenic"]:
        if automated_confidence < 0.90:
            raise ClinicalLogicError(
                f"Insufficient confidence ({automated_confidence:.2f}) for "
                f"{variant_classification} classification. Manual review required.",
                metadata={
                    'classification': variant_classification,
                    'confidence': automated_confidence,
                    'threshold': 0.90
                }
            )
        
        # Pathogenic requires strong evidence
        if variant_classification == "Pathogenic":
            strong_evidence = ["PVS1", "PS1", "PS2", "PS3", "PS4"]
            has_strong_evidence = any(ev in evidence_level for ev in strong_evidence)
            
            if not has_strong_evidence:
                raise InsufficientEvidenceError(
                    variant_id="unknown",
                    available_evidence=[evidence_level],
                    metadata={
                        'classification': variant_classification,
                        'required_evidence': 'PVS1 or PS1-PS4',
                        'provided_evidence': evidence_level
                    }
                )


def check_data_integrity(
    file_path: Path,
    expected_checksum: Optional[str] = None,
    expected_format: Optional[str] = None
) -> None:
    """
    Validate file integrity and format.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected SHA256 checksum
        expected_format: Expected file format (VCF, BAM, FASTQ, etc.)
    
    Raises:
        MissingDataError: If file doesn't exist
        ChecksumMismatchError: If checksum doesn't match
        InvalidFormatError: If format is invalid
    
    Example:
        check_data_integrity(
            Path("/data/sample.vcf.gz"),
            expected_checksum="abc123...",
            expected_format="VCF"
        )
    """
    # Check file exists
    if not file_path.exists():
        raise MissingDataError(
            data_type="file",
            identifier=str(file_path)
        )
    
    # Verify checksum if provided
    if expected_checksum:
        import hashlib
        
        actual_checksum = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                actual_checksum.update(chunk)
        
        actual_checksum = actual_checksum.hexdigest()
        
        if actual_checksum != expected_checksum:
            raise ChecksumMismatchError(
                file_path=file_path,
                expected=expected_checksum,
                actual=actual_checksum
            )
    
    # Verify format if provided
    if expected_format:
        format_valid = False
        
        if expected_format == "VCF":
            # Check VCF header
            try:
                import gzip
                open_func = gzip.open if file_path.suffix == '.gz' else open
                with open_func(file_path, 'rt') as f:
                    first_line = f.readline()
                    format_valid = first_line.startswith('##fileformat=VCF')
            except Exception:
                format_valid = False
        
        elif expected_format == "BAM":
            # Check BAM magic bytes
            try:
                with open(file_path, 'rb') as f:
                    magic = f.read(4)
                    format_valid = magic == b'\x1f\x8b\x08\x04' or magic.startswith(b'BAM\x01')
            except Exception:
                format_valid = False
        
        elif expected_format == "FASTQ":
            # Check FASTQ format
            try:
                import gzip
                open_func = gzip.open if file_path.suffix == '.gz' else open
                with open_func(file_path, 'rt') as f:
                    first_line = f.readline()
                    format_valid = first_line.startswith('@')
            except Exception:
                format_valid = False
        
        if not format_valid:
            raise InvalidFormatError(
                file_path=file_path,
                expected_format=expected_format
            )


def assert_reference_genome_match(
    vcf_path: Path,
    expected_reference: str = "GRCh38"
) -> None:
    """
    Assert VCF uses expected reference genome.
    
    Args:
        vcf_path: Path to VCF file
        expected_reference: Expected reference genome version
    
    Raises:
        ReferenceGenomeMismatchError: If reference doesn't match
    
    Example:
        assert_reference_genome_match(
            Path("/data/variants.vcf.gz"),
            expected_reference="GRCh38"
        )
    """
    import gzip
    
    try:
        open_func = gzip.open if vcf_path.suffix == '.gz' else open
        
        with open_func(vcf_path, 'rt') as f:
            for line in f:
                if not line.startswith('##'):
                    break
                
                if line.startswith('##reference='):
                    reference = line.strip().split('=', 1)[1]
                    
                    # Check if reference matches expected
                    if expected_reference not in reference:
                        raise ReferenceGenomeMismatchError(
                            expected=expected_reference,
                            actual=reference,
                            file_path=vcf_path
                        )
                    
                    return  # Match found
        
        # No reference found in header
        raise ReferenceGenomeMismatchError(
            expected=expected_reference,
            actual="Unknown (not specified in VCF header)",
            file_path=vcf_path
        )
    
    except ReferenceGenomeMismatchError:
        raise
    except Exception as e:
        raise InvalidFormatError(
            file_path=vcf_path,
            expected_format="VCF",
            metadata={'error': str(e)}
        )


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================

class ErrorContext:
    """
    Context manager for automatic error handling in pipeline stages.
    
    Example:
        with ErrorContext(error_handler, sample_id="SAMPLE001", stage="alignment"):
            run_alignment_pipeline()
    """
    
    def __init__(
        self,
        error_handler: ErrorHandler,
        **context_kwargs
    ):
        """
        Initialize error context.
        
        Args:
            error_handler: Error handler instance
            **context_kwargs: Context information
        """
        self.error_handler = error_handler
        self.context = context_kwargs
        self.exception_raised = None
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context and handle any exception.
        
        Returns:
            True if exception handled, False to propagate
        """
        if exc_val is not None:
            self.exception_raised = exc_val
            
            # Handle exception
            can_recover = self.error_handler.handle_exception(exc_val, self.context)
            
            # Return True to suppress exception if recoverable
            if isinstance(exc_val, GenomicPipelineException):
                return exc_val.recovery_strategy == RecoveryStrategy.SKIP
        
        return False


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Example usage of exception system."""
    
    # Initialize error handler
    error_handler = ErrorHandler(Path("/data/genomic_pipeline/logs"))
    
    # Example 1: Data integrity check
    try:
        check_data_integrity(
            Path("/data/sample.vcf.gz"),
            expected_checksum="abc123...",
            expected_format="VCF"
        )
    except GenomicPipelineException as e:
        print(f"Caught exception: {e}")
        print(f"Error ID: {e.context.error_id}")
        print(f"Severity: {e.severity.value}")
        print(f"Recovery strategy: {e.recovery_strategy.value}")
    
    # Example 2: Safe execution with retries
    def risky_operation():
        import random
        if random.random() < 0.7:
            raise ToolCrashError(
                tool_name="gatk",
                command="gatk HaplotypeCaller ...",
                exit_code=1,
                stderr="Out of memory"
            )
        return "Success"
    
    result = safe_execute(
        risky_operation,
        error_handler,
        context={'sample_id': 'SAMPLE001'},
        max_retries=3
    )
    
    if result:
        print(f"Operation succeeded: {result}")
    else:
        print("Operation failed after retries")
    
    # Example 3: Clinical safety validation
    try:
        validate_clinical_safety(
            variant_classification="Pathogenic",
            evidence_level="PM1+PM2",  # Insufficient for Pathogenic
            automated_confidence=0.85
        )
    except ClinicalLogicError as e:
        print(f"Clinical safety check failed: {e}")
    
    # Example 4: Error summary
    summary = error_handler.get_error_summary()
    print("\nError Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
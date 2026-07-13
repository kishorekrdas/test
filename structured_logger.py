"""
Structured Logging System for Clinical Genomic Pipelines

Production-grade logging with structured output, metrics collection, and audit integration.
Implements JSON logging, performance tracking, and regulatory compliance.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import contextvars


# ============================================================================
# ENUMERATIONS
# ============================================================================

class LogLevel(Enum):
    """Logging levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log message categories for filtering and analysis."""
    SYSTEM = "SYSTEM"
    PIPELINE = "PIPELINE"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    DATA_PROCESSING = "DATA_PROCESSING"
    QUALITY_CONTROL = "QUALITY_CONTROL"
    CLINICAL_INTERPRETATION = "CLINICAL_INTERPRETATION"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

# Thread-local context for request/pipeline tracking
current_sample_id = contextvars.ContextVar('sample_id', default=None)
current_stage = contextvars.ContextVar('stage', default=None)
current_user = contextvars.ContextVar('user', default=None)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LogRecord:
    """Structured log record with rich metadata."""
    timestamp: str
    level: str
    category: str
    message: str
    logger_name: str
    thread_id: int
    process_id: int
    hostname: str
    
    # Contextual information
    sample_id: Optional[str] = None
    stage: Optional[str] = None
    user: Optional[str] = None
    
    # Technical details
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Exception information
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    exception_traceback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class PerformanceMetric:
    """Performance metric for timing and resource tracking."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_complete(self):
        """Mark metric as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration_seconds,
            'metadata': self.metadata
        }


# ============================================================================
# JSON FORMATTER
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.
    
    Includes contextual information, exception details, and metadata.
    """
    
    def __init__(self, include_context: bool = True):
        """
        Initialize JSON formatter.
        
        Args:
            include_context: Include context variables in output
        """
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
        
        Returns:
            JSON string
        """
        # Build structured log record
        log_record = LogRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=record.levelname,
            category=getattr(record, 'category', LogCategory.SYSTEM.value),
            message=record.getMessage(),
            logger_name=record.name,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            hostname=os.uname().nodename,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            metadata=getattr(record, 'metadata', {})
        )
        
        # Add context variables if enabled
        if self.include_context:
            log_record.sample_id = current_sample_id.get()
            log_record.stage = current_stage.get()
            log_record.user = current_user.get()
        
        # Add exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_record.exception_type = exc_type.__name__
            log_record.exception_message = str(exc_value)
            log_record.exception_traceback = ''.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        
        return log_record.to_json()


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """
    Enhanced logger with structured output and performance tracking.
    
    Features:
    - JSON-formatted logs
    - Contextual information (sample ID, stage, user)
    - Performance metrics
    - Automatic log rotation
    - Multiple output destinations
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Path,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        rotation_size_mb: int = 100,
        backup_count: int = 10,
        enable_json: bool = True
    ):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Console logging level
            file_level: File logging level
            rotation_size_mb: Log rotation size in MB
            backup_count: Number of backup files to keep
            enable_json: Enable JSON formatting
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_json = enable_json
        
        # Create Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        self._add_console_handler(console_level)
        
        # Add file handler
        self._add_file_handler(file_level, rotation_size_mb, backup_count)
        
        # Performance metrics
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metrics_lock = threading.Lock()
        
        # Counters
        self.counters: Dict[str, int] = defaultdict(int)
        self.counters_lock = threading.Lock()
    
    def _add_console_handler(self, level: str):
        """Add console handler with formatting."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        
        if self.enable_json:
            console_handler.setFormatter(JSONFormatter(include_context=True))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self, level: str, rotation_size_mb: int, backup_count: int):
        """Add rotating file handler."""
        log_file = self.log_dir / f"{self.name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=rotation_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, level))
        
        if self.enable_json:
            file_handler.setFormatter(JSONFormatter(include_context=True))
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
        
        self.logger.addHandler(file_handler)
    
    def _log(
        self,
        level: int,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        metadata: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """
        Internal logging method with metadata.
        
        Args:
            level: Log level
            message: Log message
            category: Log category
            metadata: Additional metadata
            exc_info: Include exception information
        """
        # Create log record with extra fields
        extra = {
            'category': category.value,
            'metadata': metadata or {}
        }
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        **metadata
    ):
        """Log debug message."""
        self._log(logging.DEBUG, message, category, metadata)
    
    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        **metadata
    ):
        """Log info message."""
        self._log(logging.INFO, message, category, metadata)
    
    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        **metadata
    ):
        """Log warning message."""
        self._log(logging.WARNING, message, category, metadata)
    
    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        exc_info: bool = False,
        **metadata
    ):
        """Log error message."""
        self._log(logging.ERROR, message, category, metadata, exc_info)
    
    def critical(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        exc_info: bool = False,
        **metadata
    ):
        """Log critical message."""
        self._log(logging.CRITICAL, message, category, metadata, exc_info)
    
    def exception(
        self,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        **metadata
    ):
        """Log exception with traceback."""
        self._log(logging.ERROR, message, category, metadata, exc_info=True)
    
    def start_timer(self, name: str, **metadata) -> str:
        """
        Start performance timer.
        
        Args:
            name: Timer name
            **metadata: Additional metadata
        
        Returns:
            Timer ID
        """
        timer_id = f"{name}_{time.time()}"
        
        metric = PerformanceMetric(
            name=name,
            start_time=time.time(),
            metadata=metadata
        )
        
        with self.metrics_lock:
            self.metrics[timer_id] = metric
        
        self.debug(
            f"Timer started: {name}",
            category=LogCategory.PERFORMANCE,
            timer_id=timer_id,
            **metadata
        )
        
        return timer_id
    
    def stop_timer(self, timer_id: str):
        """
        Stop performance timer and log duration.
        
        Args:
            timer_id: Timer ID from start_timer
        """
        with self.metrics_lock:
            if timer_id not in self.metrics:
                self.warning(f"Timer not found: {timer_id}")
                return
            
            metric = self.metrics[timer_id]
            metric.mark_complete()
        
        self.info(
            f"Timer completed: {metric.name}",
            category=LogCategory.PERFORMANCE,
            timer_id=timer_id,
            duration_seconds=metric.duration_seconds,
            **metric.metadata
        )
    
    def increment_counter(self, name: str, value: int = 1):
        """
        Increment counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        with self.counters_lock:
            self.counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """
        Get counter value.
        
        Args:
            name: Counter name
        
        Returns:
            Counter value
        """
        with self.counters_lock:
            return self.counters[name]
    
    def log_metrics(self):
        """Log all current metrics and counters."""
        with self.counters_lock:
            counters_snapshot = dict(self.counters)
        
        with self.metrics_lock:
            active_timers = len([m for m in self.metrics.values() if m.end_time is None])
            completed_timers = len([m for m in self.metrics.values() if m.end_time is not None])
        
        self.info(
            "Current metrics snapshot",
            category=LogCategory.PERFORMANCE,
            counters=counters_snapshot,
            active_timers=active_timers,
            completed_timers=completed_timers
        )
    
    def set_context(
        self,
        sample_id: Optional[str] = None,
        stage: Optional[str] = None,
        user: Optional[str] = None
    ):
        """
        Set logging context for current thread/task.
        
        Args:
            sample_id: Sample identifier
            stage: Pipeline stage
            user: User identifier
        """
        if sample_id is not None:
            current_sample_id.set(sample_id)
        if stage is not None:
            current_stage.set(stage)
        if user is not None:
            current_user.set(user)
    
    def clear_context(self):
        """Clear logging context."""
        current_sample_id.set(None)
        current_stage.set(None)
        current_user.set(None)


# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class LogContext:
    """
    Context manager for scoped logging context.
    
    Example:
        with LogContext(logger, sample_id="S001", stage="alignment"):
            # All logs in this block include context
            logger.info("Processing sample")
    """
    
    def __init__(
        self,
        logger: StructuredLogger,
        sample_id: Optional[str] = None,
        stage: Optional[str] = None,
        user: Optional[str] = None
    ):
        """
        Initialize log context.
        
        Args:
            logger: StructuredLogger instance
            sample_id: Sample identifier
            stage: Pipeline stage
            user: User identifier
        """
        self.logger = logger
        self.sample_id = sample_id
        self.stage = stage
        self.user = user
        
        # Store previous context
        self.previous_sample = None
        self.previous_stage = None
        self.previous_user = None
    
    def __enter__(self):
        """Enter context and set logging context."""
        self.previous_sample = current_sample_id.get()
        self.previous_stage = current_stage.get()
        self.previous_user = current_user.get()
        
        self.logger.set_context(
            sample_id=self.sample_id,
            stage=self.stage,
            user=self.user
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous context."""
        # Restore previous context
        current_sample_id.set(self.previous_sample)
        current_stage.set(self.previous_stage)
        current_user.set(self.previous_user)


class TimerContext:
    """
    Context manager for automatic timing.
    
    Example:
        with TimerContext(logger, "variant_calling", sample_id="S001"):
            call_variants()
    """
    
    def __init__(
        self,
        logger: StructuredLogger,
        name: str,
        **metadata
    ):
        """
        Initialize timer context.
        
        Args:
            logger: StructuredLogger instance
            name: Timer name
            **metadata: Additional metadata
        """
        self.logger = logger
        self.name = name
        self.metadata = metadata
        self.timer_id = None
    
    def __enter__(self):
        """Start timer."""
        self.timer_id = self.logger.start_timer(self.name, **self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer."""
        if self.timer_id:
            self.logger.stop_timer(self.timer_id)


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """
    Specialized logger for regulatory compliance audit trails.
    
    Features:
    - Immutable logs
    - Cryptographic signatures
    - Structured audit events
    """
    
    def __init__(self, log_dir: Path, secret_key: Optional[str] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            secret_key: Secret key for signatures
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.secret_key = secret_key or os.environ.get('AUDIT_SECRET_KEY', 'default_key')
        self.audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger('genomic_pipeline.audit')
    
    def log_event(
        self,
        event_type: str,
        user: str,
        action: str,
        resource: str,
        result: str,
        **metadata
    ):
        """
        Log audit event.
        
        Args:
            event_type: Type of event (ACCESS, MODIFY, DELETE, etc.)
            user: User performing action
            action: Action description
            resource: Resource affected
            result: Result (SUCCESS, FAILURE)
            **metadata: Additional metadata
        """
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'user': user,
            'action': action,
            'resource': resource,
            'result': result,
            'metadata': metadata,
            'hostname': os.uname().nodename,
            'process_id': os.getpid()
        }
        
        # Compute signature
        import hashlib
        canonical = json.dumps(event, sort_keys=True)
        signature = hashlib.sha256(f"{canonical}{self.secret_key}".encode()).hexdigest()
        event['signature'] = signature
        
        # Write to audit log
        with self.lock:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
                f.flush()
                os.fsync(f.fileno())
        
        self.logger.info(f"Audit event: {event_type} | {user} | {action} | {result}")


# ============================================================================
# LOGGER FACTORY
# ============================================================================

class LoggerFactory:
    """
    Factory for creating configured loggers.
    
    Ensures consistent configuration across the application.
    """
    
    _loggers: Dict[str, StructuredLogger] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Path,
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ) -> StructuredLogger:
        """
        Get or create logger.
        
        Args:
            name: Logger name
            log_dir: Log directory
            console_level: Console level
            file_level: File level
        
        Returns:
            StructuredLogger instance
        """
        with cls._lock:
            if name not in cls._loggers:
                cls._loggers[name] = StructuredLogger(
                    name=name,
                    log_dir=log_dir,
                    console_level=console_level,
                    file_level=file_level
                )
            
            return cls._loggers[name]
    
    @classmethod
    def configure_root_logger(cls, log_dir: Path):
        """
        Configure root logger for the application.
        
        Args:
            log_dir: Log directory
        """
        root_logger = cls.get_logger(
            name='genomic_pipeline',
            log_dir=log_dir,
            console_level='INFO',
            file_level='DEBUG'
        )
        
        return root_logger


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_function_call(logger: StructuredLogger, category: LogCategory = LogCategory.SYSTEM):
    """
    Decorator to automatically log function calls with timing.
    
    Args:
        logger: StructuredLogger instance
        category: Log category
    
    Example:
        @log_function_call(logger, LogCategory.PIPELINE)
        def process_sample(sample_id):
            # Function implementation
            pass
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            logger.info(
                f"Function called: {func_name}",
                category=category,
                args=str(args)[:100],
                kwargs=str(kwargs)[:100]
            )
            
            with TimerContext(logger, func_name):
                try:
                    result = func(*args, **kwargs)
                    
                    logger.info(
                        f"Function completed: {func_name}",
                        category=category
                    )
                    
                    return result
                
                except Exception as e:
                    logger.exception(
                        f"Function failed: {func_name}",
                        category=category,
                        error=str(e)
                    )
                    raise
        
        return wrapper
    return decorator


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Example usage of structured logging system."""
    
    # Initialize logger
    log_dir = Path("/data/genomic_pipeline/logs")
    logger = LoggerFactory.get_logger(
        name="test_pipeline",
        log_dir=log_dir
    )
    
    # Example 1: Basic logging
    logger.info("Pipeline started", category=LogCategory.PIPELINE)
    logger.debug("Debug information", category=LogCategory.SYSTEM, detail="test")
    
    # Example 2: Context logging
    with LogContext(logger, sample_id="SAMPLE001", stage="alignment"):
        logger.info("Processing sample", category=LogCategory.DATA_PROCESSING)
        logger.warning("Low quality detected", category=LogCategory.QUALITY_CONTROL)
    
    # Example 3: Performance timing
    with TimerContext(logger, "variant_calling", sample_id="SAMPLE001"):
        time.sleep(1)  # Simulate work
        logger.info("Variants called", category=LogCategory.PIPELINE)
    
    # Example 4: Counters
    logger.increment_counter("variants_processed", 1000)
    logger.increment_counter("variants_filtered", 50)
    logger.log_metrics()
    
    # Example 5: Function decorator
    @log_function_call(logger, LogCategory.DATA_PROCESSING)
    def process_data(data_size):
        time.sleep(0.5)
        return f"Processed {data_size} records"
    
    result = process_data(10000)
    print(result)
    
    # Example 6: Audit logging
    audit_logger = AuditLogger(log_dir / "audit")
    audit_logger.log_event(
        event_type="ACCESS",
        user="analyst_01",
        action="view_variant",
        resource="SAMPLE001:chr1:12345",
        result="SUCCESS",
        ip_address="192.168.1.100"
    )
    
    logger.info("Pipeline completed", category=LogCategory.PIPELINE)


if __name__ == '__main__':
    main()
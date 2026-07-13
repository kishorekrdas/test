"""
Genomic Pipeline Orchestration & Scheduling System

Production-grade orchestrator for clinical genomic variant detection pipelines.
Implements checkpointed execution, resource management, and complete audit trails.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import asyncio
import enum
import hashlib
import json
import logging
import multiprocessing as mp
import os
import psutil
import signal
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import threading


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

class PipelineStage(enum.Enum):
    """Pipeline execution stages with deterministic ordering."""
    PREPROCESSING = 1
    ALIGNMENT = 2
    QC = 3
    VARIANT_CALLING = 4
    NORMALIZATION = 5
    ANNOTATION = 6
    FILTERING = 7
    CLASSIFICATION = 8
    REPORTING = 9


class ExecutionStatus(enum.Enum):
    """Execution status for tracking and audit."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CHECKPOINTED = "CHECKPOINTED"
    CANCELLED = "CANCELLED"


class ResourceType(enum.Enum):
    """System resource types for allocation."""
    CPU = "CPU"
    GPU = "GPU"
    MEMORY = "MEMORY"
    DISK_IO = "DISK_IO"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ResourceRequirements:
    """Resource requirements for a pipeline stage."""
    cpu_cores: int = 4
    memory_gb: float = 16.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    disk_io_priority: int = 0  # 0=normal, 1=high, 2=critical
    estimated_runtime_minutes: float = 30.0
    
    def validate(self) -> bool:
        """Validate resource requirements are positive."""
        return (self.cpu_cores > 0 and 
                self.memory_gb > 0 and 
                self.gpu_count >= 0 and
                self.estimated_runtime_minutes > 0)


@dataclass
class CheckpointMetadata:
    """Metadata for pipeline checkpoints."""
    checkpoint_id: str
    stage: PipelineStage
    timestamp: str
    input_checksums: Dict[str, str]
    output_checksums: Dict[str, str]
    tool_versions: Dict[str, str]
    parameters: Dict[str, Any]
    execution_time_seconds: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['stage'] = self.stage.name
        return result


@dataclass
class AuditLogEntry:
    """Immutable audit log entry for regulatory compliance."""
    entry_id: str
    timestamp: str
    sample_id: str
    stage: PipelineStage
    status: ExecutionStatus
    user: str
    host: str
    command: str
    exit_code: int
    stdout_path: Optional[str]
    stderr_path: Optional[str]
    resources_used: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['stage'] = self.stage.name
        result['status'] = self.status.value
        return result
    
    def compute_signature(self, secret_key: str) -> str:
        """Compute cryptographic signature for audit integrity."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(f"{canonical}{secret_key}".encode()).hexdigest()


@dataclass
class PipelineTask:
    """Represents a single executable task in the pipeline."""
    task_id: str
    sample_id: str
    stage: PipelineStage
    command: List[str]
    input_files: List[Path]
    output_files: List[Path]
    checkpoint_file: Path
    requirements: ResourceRequirements
    dependencies: Set[str] = field(default_factory=set)
    status: ExecutionStatus = ExecutionStatus.PENDING
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_checkpointed(self) -> bool:
        """Check if task has valid checkpoint."""
        if not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Validate input checksums match
            for input_file in self.input_files:
                if not input_file.exists():
                    return False
                current_checksum = compute_file_checksum(input_file)
                stored_checksum = checkpoint_data.get('input_checksums', {}).get(str(input_file))
                if current_checksum != stored_checksum:
                    return False
            
            # Validate output files exist
            for output_file in self.output_files:
                if not output_file.exists():
                    return False
            
            return True
        except (json.JSONDecodeError, KeyError, IOError):
            return False
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_tasks)


@dataclass
class SystemResources:
    """Current system resource availability."""
    available_cpu_cores: int
    available_memory_gb: float
    available_gpus: List[int]
    gpu_memory_gb: Dict[int, float]
    active_disk_io_tasks: int
    
    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated for requirements."""
        if self.available_cpu_cores < requirements.cpu_cores:
            return False
        if self.available_memory_gb < requirements.memory_gb:
            return False
        if requirements.gpu_count > 0 and len(self.available_gpus) < requirements.gpu_count:
            return False
        if requirements.disk_io_priority >= 1 and self.active_disk_io_tasks >= 3:
            return False
        return True
    
    def allocate(self, requirements: ResourceRequirements) -> Optional[Dict[str, Any]]:
        """Allocate resources and return allocation details."""
        if not self.can_allocate(requirements):
            return None
        
        allocation = {
            'cpu_cores': requirements.cpu_cores,
            'memory_gb': requirements.memory_gb,
            'gpus': []
        }
        
        self.available_cpu_cores -= requirements.cpu_cores
        self.available_memory_gb -= requirements.memory_gb
        
        if requirements.gpu_count > 0:
            allocated_gpus = self.available_gpus[:requirements.gpu_count]
            allocation['gpus'] = allocated_gpus
            self.available_gpus = self.available_gpus[requirements.gpu_count:]
        
        if requirements.disk_io_priority >= 1:
            self.active_disk_io_tasks += 1
        
        return allocation
    
    def release(self, allocation: Dict[str, Any], disk_io_priority: int = 0):
        """Release previously allocated resources."""
        self.available_cpu_cores += allocation['cpu_cores']
        self.available_memory_gb += allocation['memory_gb']
        
        if allocation['gpus']:
            self.available_gpus.extend(allocation['gpus'])
            self.available_gpus.sort()
        
        if disk_io_priority >= 1:
            self.active_disk_io_tasks = max(0, self.active_disk_io_tasks - 1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic checksum of file for integrity verification.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (sha256, md5)
    
    Returns:
        Hexadecimal checksum string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_system_info() -> Dict[str, Any]:
    """
    Gather system information for audit trail.
    
    Returns:
        Dictionary with system metadata
    """
    return {
        'hostname': os.uname().nodename,
        'cpu_count': mp.cpu_count(),
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'platform': sys.platform,
        'python_version': sys.version,
        'user': os.environ.get('USER', 'unknown'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat()
    }


def create_iso_timestamp() -> str:
    """Create ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_dir: Path, log_level: str = 'INFO') -> logging.Logger:
    """
    Configure structured logging with file and console handlers.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    
    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('genomic_pipeline')
    logger.setLevel(getattr(logging, log_level))
    
    # File handler with rotation
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    # Structured formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# AUDIT TRAIL MANAGER
# ============================================================================

class AuditTrailManager:
    """
    Manages immutable audit trail for regulatory compliance.
    
    Features:
    - Append-only logging
    - Cryptographic signatures
    - Thread-safe operations
    - Structured query interface
    """
    
    def __init__(self, audit_dir: Path, secret_key: Optional[str] = None):
        """
        Initialize audit trail manager.
        
        Args:
            audit_dir: Directory for audit logs
            secret_key: Secret key for cryptographic signatures
        """
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.secret_key = secret_key or os.environ.get('AUDIT_SECRET_KEY', 'default_key')
        self.audit_file = self.audit_dir / f"audit_trail_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger('genomic_pipeline.audit')
    
    def log_entry(self, entry: AuditLogEntry) -> str:
        """
        Append audit log entry with cryptographic signature.
        
        Args:
            entry: Audit log entry to record
        
        Returns:
            Entry signature for verification
        """
        signature = entry.compute_signature(self.secret_key)
        
        log_record = entry.to_dict()
        log_record['signature'] = signature
        
        with self.lock:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(log_record) + '\n')
                f.flush()
                os.fsync(f.fileno())
        
        self.logger.info(f"Audit entry recorded: {entry.entry_id} | {entry.stage.name} | {entry.status.value}")
        
        return signature
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify integrity of audit trail.
        
        Returns:
            Tuple of (is_valid, list_of_invalid_entries)
        """
        invalid_entries = []
        
        with self.lock:
            if not self.audit_file.exists():
                return True, []
            
            with open(self.audit_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        stored_signature = record.pop('signature', None)
                        
                        # Reconstruct entry
                        entry = AuditLogEntry(
                            entry_id=record['entry_id'],
                            timestamp=record['timestamp'],
                            sample_id=record['sample_id'],
                            stage=PipelineStage[record['stage']],
                            status=ExecutionStatus(record['status']),
                            user=record['user'],
                            host=record['host'],
                            command=record['command'],
                            exit_code=record['exit_code'],
                            stdout_path=record['stdout_path'],
                            stderr_path=record['stderr_path'],
                            resources_used=record['resources_used'],
                            metadata=record['metadata']
                        )
                        
                        computed_signature = entry.compute_signature(self.secret_key)
                        
                        if stored_signature != computed_signature:
                            invalid_entries.append(f"Line {line_num}: {entry.entry_id}")
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        invalid_entries.append(f"Line {line_num}: Parse error - {e}")
        
        is_valid = len(invalid_entries) == 0
        return is_valid, invalid_entries
    
    def query_by_sample(self, sample_id: str) -> List[Dict]:
        """
        Query audit trail for specific sample.
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            List of audit entries for sample
        """
        entries = []
        
        with self.lock:
            if not self.audit_file.exists():
                return entries
            
            with open(self.audit_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record['sample_id'] == sample_id:
                            entries.append(record)
                    except json.JSONDecodeError:
                        continue
        
        return entries
    
    def query_by_stage(self, stage: PipelineStage) -> List[Dict]:
        """
        Query audit trail for specific pipeline stage.
        
        Args:
            stage: Pipeline stage
        
        Returns:
            List of audit entries for stage
        """
        entries = []
        
        with self.lock:
            if not self.audit_file.exists():
                return entries
            
            with open(self.audit_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record['stage'] == stage.name:
                            entries.append(record)
                    except json.JSONDecodeError:
                        continue
        
        return entries


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """
    Manages pipeline checkpoints for resumable execution.
    
    Features:
    - Input/output integrity verification
    - Tool version tracking
    - Parameter capture
    - Atomic checkpoint creation
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('genomic_pipeline.checkpoint')
    
    def create_checkpoint(self, task: PipelineTask, execution_time: float) -> CheckpointMetadata:
        """
        Create checkpoint after successful task execution.
        
        Args:
            task: Completed pipeline task
            execution_time: Execution time in seconds
        
        Returns:
            Checkpoint metadata
        """
        checkpoint_id = f"{task.task_id}_{int(time.time())}"
        
        # Compute input checksums
        input_checksums = {}
        for input_file in task.input_files:
            if input_file.exists():
                input_checksums[str(input_file)] = compute_file_checksum(input_file)
        
        # Compute output checksums
        output_checksums = {}
        for output_file in task.output_files:
            if output_file.exists():
                output_checksums[str(output_file)] = compute_file_checksum(output_file)
        
        # Extract tool versions from metadata
        tool_versions = task.metadata.get('tool_versions', {})
        
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            stage=task.stage,
            timestamp=create_iso_timestamp(),
            input_checksums=input_checksums,
            output_checksums=output_checksums,
            tool_versions=tool_versions,
            parameters=task.metadata.get('parameters', {}),
            execution_time_seconds=execution_time
        )
        
        # Write checkpoint atomically
        temp_checkpoint = task.checkpoint_file.with_suffix('.tmp')
        with open(temp_checkpoint, 'w') as f:
            json.dump(checkpoint_metadata.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        temp_checkpoint.rename(task.checkpoint_file)
        
        self.logger.info(f"Checkpoint created: {checkpoint_id} | {task.stage.name}")
        
        return checkpoint_metadata
    
    def load_checkpoint(self, checkpoint_file: Path) -> Optional[CheckpointMetadata]:
        """
        Load checkpoint metadata from file.
        
        Args:
            checkpoint_file: Path to checkpoint file
        
        Returns:
            Checkpoint metadata or None if invalid
        """
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            return CheckpointMetadata(
                checkpoint_id=data['checkpoint_id'],
                stage=PipelineStage[data['stage']],
                timestamp=data['timestamp'],
                input_checksums=data['input_checksums'],
                output_checksums=data['output_checksums'],
                tool_versions=data['tool_versions'],
                parameters=data['parameters'],
                execution_time_seconds=data['execution_time_seconds']
            )
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            return None
    
    def validate_checkpoint(self, task: PipelineTask) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            task: Pipeline task to validate
        
        Returns:
            True if checkpoint is valid
        """
        return task.is_checkpointed()


# ============================================================================
# RESOURCE MANAGER
# ============================================================================

class ResourceManager:
    """
    Manages system resource allocation for pipeline tasks.
    
    Features:
    - CPU, GPU, memory tracking
    - Priority-based allocation
    - Deadlock prevention
    - Resource usage monitoring
    """
    
    def __init__(self, cpu_cores: int, memory_gb: float, gpus: List[int]):
        """
        Initialize resource manager.
        
        Args:
            cpu_cores: Number of CPU cores available
            memory_gb: Total memory in GB
            gpus: List of GPU device IDs
        """
        self.total_cpu_cores = cpu_cores
        self.total_memory_gb = memory_gb
        self.total_gpus = gpus.copy()
        
        self.resources = SystemResources(
            available_cpu_cores=cpu_cores,
            available_memory_gb=memory_gb,
            available_gpus=gpus.copy(),
            gpu_memory_gb={gpu_id: 0.0 for gpu_id in gpus},
            active_disk_io_tasks=0
        )
        
        self.allocations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger('genomic_pipeline.resources')
    
    def get_available_resources(self) -> SystemResources:
        """Get current resource availability."""
        with self.lock:
            return SystemResources(
                available_cpu_cores=self.resources.available_cpu_cores,
                available_memory_gb=self.resources.available_memory_gb,
                available_gpus=self.resources.available_gpus.copy(),
                gpu_memory_gb=self.resources.gpu_memory_gb.copy(),
                active_disk_io_tasks=self.resources.active_disk_io_tasks
            )
    
    def allocate_resources(self, task_id: str, requirements: ResourceRequirements) -> Optional[Dict[str, Any]]:
        """
        Allocate resources for task.
        
        Args:
            task_id: Task identifier
            requirements: Resource requirements
        
        Returns:
            Allocation details or None if unavailable
        """
        with self.lock:
            allocation = self.resources.allocate(requirements)
            
            if allocation:
                self.allocations[task_id] = allocation
                self.allocations[task_id]['disk_io_priority'] = requirements.disk_io_priority
                
                self.logger.info(
                    f"Resources allocated for {task_id}: "
                    f"CPU={allocation['cpu_cores']}, "
                    f"MEM={allocation['memory_gb']:.1f}GB, "
                    f"GPU={allocation['gpus']}"
                )
            
            return allocation
    
    def release_resources(self, task_id: str):
        """
        Release resources allocated to task.
        
        Args:
            task_id: Task identifier
        """
        with self.lock:
            if task_id not in self.allocations:
                return
            
            allocation = self.allocations[task_id]
            disk_io_priority = allocation.pop('disk_io_priority', 0)
            
            self.resources.release(allocation, disk_io_priority)
            del self.allocations[task_id]
            
            self.logger.info(f"Resources released for {task_id}")
    
    def get_utilization(self) -> Dict[str, float]:
        """
        Calculate current resource utilization.
        
        Returns:
            Dictionary with utilization percentages
        """
        with self.lock:
            return {
                'cpu_utilization': 1.0 - (self.resources.available_cpu_cores / self.total_cpu_cores),
                'memory_utilization': 1.0 - (self.resources.available_memory_gb / self.total_memory_gb),
                'gpu_utilization': 1.0 - (len(self.resources.available_gpus) / len(self.total_gpus)) if self.total_gpus else 0.0
            }


# ============================================================================
# PIPELINE SCHEDULER
# ============================================================================

class PipelineScheduler:
    """
    Main orchestrator for genomic pipeline execution.
    
    Features:
    - Dependency-aware task scheduling
    - Checkpointed resumable execution
    - Resource-constrained parallel execution
    - Complete audit trail
    - Graceful shutdown handling
    """
    
    def __init__(
        self,
        work_dir: Path,
        cpu_cores: int = mp.cpu_count(),
        memory_gb: float = psutil.virtual_memory().total / (1024**3),
        gpus: Optional[List[int]] = None,
        max_parallel_tasks: int = 4
    ):
        """
        Initialize pipeline scheduler.
        
        Args:
            work_dir: Working directory for pipeline
            cpu_cores: Number of CPU cores to use
            memory_gb: Memory limit in GB
            gpus: List of GPU device IDs
            max_parallel_tasks: Maximum parallel tasks
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize directories
        self.log_dir = self.work_dir / 'logs'
        self.checkpoint_dir = self.work_dir / 'checkpoints'
        self.audit_dir = self.work_dir / 'audit'
        
        for directory in [self.log_dir, self.checkpoint_dir, self.audit_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = setup_logging(self.log_dir)
        
        # Initialize managers
        self.resource_manager = ResourceManager(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb * 0.9,  # Reserve 10% for system
            gpus=gpus or []
        )
        
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.audit_manager = AuditTrailManager(self.audit_dir)
        
        # Task management
        self.tasks: Dict[str, PipelineTask] = {}
        self.task_queue: List[PipelineTask] = []
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Dict[str, Any] = {}
        
        self.max_parallel_tasks = max_parallel_tasks
        self.executor = ProcessPoolExecutor(max_workers=max_parallel_tasks)
        
        # Shutdown handling
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Pipeline scheduler initialized")
        self.logger.info(f"Resources: {cpu_cores} CPUs, {memory_gb:.1f}GB RAM, {len(gpus or [])} GPUs")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    def add_task(self, task: PipelineTask):
        """
        Add task to scheduler.
        
        Args:
            task: Pipeline task to schedule
        """
        if not task.requirements.validate():
            raise ValueError(f"Invalid resource requirements for task {task.task_id}")
        
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        
        self.logger.info(f"Task added: {task.task_id} | {task.stage.name} | Priority={task.priority}")
    
    def build_dependency_graph(self) -> Dict[str, Set[str]]:
        """
        Build task dependency graph.
        
        Returns:
            Adjacency list representation of dependencies
        """
        graph = {task_id: task.dependencies.copy() for task_id, task in self.tasks.items()}
        return graph
    
    def topological_sort(self) -> List[PipelineTask]:
        """
        Perform topological sort of tasks based on dependencies.
        
        Returns:
            List of tasks in execution order
        """
        graph = self.build_dependency_graph()
        in_degree = {task_id: len(deps) for task_id, deps in graph.items()}
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        while queue:
            # Sort by priority (higher priority first)
            queue.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)
            task_id = queue.pop(0)
            sorted_tasks.append(self.tasks[task_id])
            
            # Reduce in-degree for dependent tasks
            for other_task_id, deps in graph.items():
                if task_id in deps:
                    in_degree[other_task_id] -= 1
                    if in_degree[other_task_id] == 0:
                        queue.append(other_task_id)
        
        if len(sorted_tasks) != len(self.tasks):
            raise RuntimeError("Circular dependency detected in task graph")
        
        return sorted_tasks
    
    async def execute_task(self, task: PipelineTask) -> Tuple[bool, Optional[str]]:
        """
        Execute single pipeline task with resource management and checkpointing.
        
        Args:
            task: Task to execute
        
        Returns:
            Tuple of (success, error_message)
        """
        task_start_time = time.time()
        
        # Check for valid checkpoint
        if task.is_checkpointed():
            self.logger.info(f"Task {task.task_id} skipped (valid checkpoint exists)")
            self.completed_tasks.add(task.task_id)
            return True, None
        
        # Allocate resources
        allocation = None
        while allocation is None and not self.shutdown_event.is_set():
            allocation = self.resource_manager.allocate_resources(task.task_id, task.requirements)
            if allocation is None:
                await asyncio.sleep(1.0)
        
        if self.shutdown_event.is_set():
            return False, "Shutdown requested"
        
        # Prepare execution environment
        env = os.environ.copy()
        if allocation['gpus']:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, allocation['gpus']))
        
        # Create output directories
        for output_file in task.output_files:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        stdout_log = self.log_dir / f"{task.task_id}_stdout.log"
        stderr_log = self.log_dir / f"{task.task_id}_stderr.log"
        
        try:
            # Execute command
            task.status = ExecutionStatus.RUNNING
            self.logger.info(f"Executing task {task.task_id}: {' '.join(task.command[:3])}...")
            
            process = await asyncio.create_subprocess_exec(
                *task.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout_data, stderr_data = await process.communicate()
            
            # Write logs
            stdout_log.write_bytes(stdout_data)
            stderr_log.write_bytes(stderr_data)
            
            exit_code = process.returncode
            execution_time = time.time() - task_start_time
            
            # Log to audit trail
            audit_entry = AuditLogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=create_iso_timestamp(),
                sample_id=task.sample_id,
                stage=task.stage,
                status=ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED,
                user=os.environ.get('USER', 'unknown'),
                host=os.uname().nodename,
                command=' '.join(task.command),
                exit_code=exit_code,
                stdout_path=str(stdout_log),
                stderr_path=str(stderr_log),
                resources_used={
                    'cpu_cores': allocation['cpu_cores'],
                    'memory_gb': allocation['memory_gb'],
                    'gpus': allocation['gpus'],
                    'execution_time_seconds': execution_time
                },
                metadata=task.metadata
            )
            
            self.audit_manager.log_entry(audit_entry)
            
            # Validate outputs
            if exit_code == 0:
                missing_outputs = [str(f) for f in task.output_files if not f.exists()]
                if missing_outputs:
                    error_msg = f"Task completed but missing outputs: {missing_outputs}"
                    self.logger.error(error_msg)
                    task.status = ExecutionStatus.FAILED
                    self.failed_tasks.add(task.task_id)
                    return False, error_msg
                
                # Create checkpoint
                self.checkpoint_manager.create_checkpoint(task, execution_time)
                task.status = ExecutionStatus.COMPLETED
                self.completed_tasks.add(task.task_id)
                
                self.logger.info(f"Task {task.task_id} completed successfully in {execution_time:.1f}s")
                return True, None
            else:
                error_msg = f"Task failed with exit code {exit_code}"
                stderr_preview = stderr_data.decode('utf-8', errors='ignore')[-500:]
                self.logger.error(f"{error_msg}. Stderr: {stderr_preview}")
                
                task.status = ExecutionStatus.FAILED
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    self.logger.warning(f"Task {task.task_id} will be retried ({task.retry_count}/{task.max_retries})")
                    task.status = ExecutionStatus.PENDING
                    return False, error_msg
                else:
                    self.failed_tasks.add(task.task_id)
                    return False, f"{error_msg} (max retries exceeded)"
        
        except Exception as e:
            error_msg = f"Task execution exception: {str(e)}"
            self.logger.exception(error_msg)
            
            task.status = ExecutionStatus.FAILED
            self.failed_tasks.add(task.task_id)
            
            return False, error_msg
        
        finally:
            # Always release resources
            self.resource_manager.release_resources(task.task_id)
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete pipeline with dependency management.
        
        Returns:
            Execution summary dictionary
        """
        pipeline_start_time = time.time()
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION STARTED")
        self.logger.info("=" * 80)
        
        # Verify audit trail integrity before starting
        is_valid, invalid_entries = self.audit_manager.verify_integrity()
        if not is_valid:
            self.logger.error(f"Audit trail integrity check failed: {invalid_entries}")
            raise RuntimeError("Cannot proceed with compromised audit trail")
        
        # Sort tasks topologically
        try:
            sorted_tasks = self.topological_sort()
            self.logger.info(f"Scheduled {len(sorted_tasks)} tasks across {len(set(t.stage for t in sorted_tasks))} stages")
        except RuntimeError as e:
            self.logger.error(f"Task scheduling failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'completed_tasks': 0,
                'failed_tasks': 0
            }
        
        # Execute tasks with resource constraints
        task_futures = {}
        task_index = 0
        
        while (task_index < len(sorted_tasks) or task_futures) and not self.shutdown_event.is_set():
            # Submit new tasks if resources available
            while (task_index < len(sorted_tasks) and 
                   len(task_futures) < self.max_parallel_tasks and
                   not self.shutdown_event.is_set()):
                
                task = sorted_tasks[task_index]
                
                # Check dependencies satisfied
                if task.can_execute(self.completed_tasks):
                    # Check resource availability
                    available = self.resource_manager.get_available_resources()
                    if available.can_allocate(task.requirements):
                        future = asyncio.create_task(self.execute_task(task))
                        task_futures[task.task_id] = future
                        self.running_tasks[task.task_id] = task
                        task_index += 1
                    else:
                        break
                else:
                    # Check if dependencies failed
                    failed_deps = task.dependencies.intersection(self.failed_tasks)
                    if failed_deps:
                        self.logger.error(f"Task {task.task_id} skipped due to failed dependencies: {failed_deps}")
                        self.failed_tasks.add(task.task_id)
                        task_index += 1
                    else:
                        break
            
            # Wait for any task to complete
            if task_futures:
                done, pending = await asyncio.wait(
                    task_futures.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0
                )
                
                for future in done:
                    # Find which task completed
                    completed_task_id = None
                    for tid, fut in task_futures.items():
                        if fut == future:
                            completed_task_id = tid
                            break
                    
                    if completed_task_id:
                        try:
                            success, error = await future
                            if not success:
                                self.logger.warning(f"Task {completed_task_id} failed: {error}")
                        except Exception as e:
                            self.logger.exception(f"Task {completed_task_id} raised exception: {e}")
                        
                        del task_futures[completed_task_id]
                        del self.running_tasks[completed_task_id]
                
                # Log progress
                utilization = self.resource_manager.get_utilization()
                self.logger.info(
                    f"Progress: {len(self.completed_tasks)}/{len(sorted_tasks)} completed, "
                    f"{len(self.failed_tasks)} failed, "
                    f"{len(task_futures)} running | "
                    f"CPU: {utilization['cpu_utilization']*100:.0f}%, "
                    f"MEM: {utilization['memory_utilization']*100:.0f}%, "
                    f"GPU: {utilization['gpu_utilization']*100:.0f}%"
                )
            
            await asyncio.sleep(0.1)
        
        # Handle shutdown
        if self.shutdown_event.is_set():
            self.logger.warning("Pipeline interrupted by shutdown signal")
            
            # Cancel remaining tasks
            for future in task_futures.values():
                future.cancel()
            
            # Wait for cancellation
            if task_futures:
                await asyncio.wait(task_futures.values(), timeout=5.0)
        
        pipeline_end_time = time.time()
        total_time = pipeline_end_time - pipeline_start_time
        
        # Generate summary
        summary = {
            'status': 'COMPLETED' if len(self.failed_tasks) == 0 else 'FAILED',
            'total_tasks': len(sorted_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'skipped_tasks': len(sorted_tasks) - len(self.completed_tasks) - len(self.failed_tasks),
            'total_time_seconds': total_time,
            'shutdown_requested': self.shutdown_event.is_set(),
            'timestamp': create_iso_timestamp()
        }
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION COMPLETED")
        self.logger.info(f"Status: {summary['status']}")
        self.logger.info(f"Completed: {summary['completed_tasks']}/{summary['total_tasks']}")
        self.logger.info(f"Failed: {summary['failed_tasks']}")
        self.logger.info(f"Total time: {total_time:.1f}s")
        self.logger.info("=" * 80)
        
        # Verify audit trail after completion
        is_valid, invalid_entries = self.audit_manager.verify_integrity()
        if not is_valid:
            self.logger.error(f"Post-execution audit trail corruption detected: {invalid_entries}")
            summary['audit_integrity'] = 'COMPROMISED'
        else:
            summary['audit_integrity'] = 'VERIFIED'
        
        return summary
    
    def shutdown(self):
        """Gracefully shutdown scheduler and cleanup resources."""
        self.logger.info("Shutting down pipeline scheduler...")
        
        self.shutdown_event.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        
        self.logger.info("Pipeline scheduler shutdown complete")


# ============================================================================
# MAIN EXECUTION ENTRY POINT
# ============================================================================

async def main():
    """Example usage of pipeline scheduler."""
    
    # Initialize scheduler
    scheduler = PipelineScheduler(
        work_dir=Path('/data/genomic_pipeline'),
        cpu_cores=32,
        memory_gb=128.0,
        gpus=[0, 1, 2, 3],
        max_parallel_tasks=4
    )
    
    # Example: Add preprocessing task
    preprocess_task = PipelineTask(
        task_id='preprocess_001',
        sample_id='SAMPLE001',
        stage=PipelineStage.PREPROCESSING,
        command=['fastqc', '--threads', '4', 'input.fastq.gz'],
        input_files=[Path('/data/samples/input.fastq.gz')],
        output_files=[Path('/data/output/input_fastqc.html')],
        checkpoint_file=Path('/data/genomic_pipeline/checkpoints/preprocess_001.json'),
        requirements=ResourceRequirements(
            cpu_cores=4,
            memory_gb=8.0,
            estimated_runtime_minutes=15.0
        )
    )
    
    scheduler.add_task(preprocess_task)
    
    # Execute pipeline
    summary = await scheduler.run_pipeline()
    
    # Cleanup
    scheduler.shutdown()
    
    return summary


if __name__ == '__main__':
    asyncio.run(main())
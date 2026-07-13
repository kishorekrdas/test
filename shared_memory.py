"""
Shared Memory Management for Parallel Genomic Processing

Production-grade shared memory system for efficient inter-process communication.
Optimized for large genomic data structures shared across worker processes.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import mmap
import multiprocessing as mp
import os
import pickle
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SharedDataType(Enum):
    """Types of shared data structures."""
    NUMPY_ARRAY = "NUMPY_ARRAY"
    DICT = "DICT"
    LIST = "LIST"
    COUNTER = "COUNTER"
    QUEUE = "QUEUE"
    LOCK = "LOCK"
    EVENT = "EVENT"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SharedMemoryInfo:
    """Metadata for shared memory segments."""
    name: str
    size_bytes: int
    data_type: SharedDataType
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    created_by: int = 0  # Process ID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'size_bytes': self.size_bytes,
            'data_type': self.data_type.value,
            'shape': self.shape,
            'dtype': self.dtype,
            'created_by': self.created_by
        }


# ============================================================================
# SHARED MEMORY MANAGER
# ============================================================================

class SharedMemoryManager:
    """
    Central manager for shared memory segments.
    
    Features:
    - Process-safe memory allocation
    - Automatic cleanup
    - Memory usage tracking
    - Support for NumPy arrays, dicts, lists
    """
    
    def __init__(self):
        """Initialize shared memory manager."""
        self.segments: Dict[str, SharedMemoryInfo] = {}
        self.lock = mp.Lock()
        self.logger = logging.getLogger(f'{__name__}.SharedMemoryManager')
    
    def allocate(
        self,
        name: str,
        size_bytes: int,
        data_type: SharedDataType
    ) -> mp.shared_memory.SharedMemory:
        """
        Allocate shared memory segment.
        
        Args:
            name: Unique name for segment
            size_bytes: Size in bytes
            data_type: Type of data to store
        
        Returns:
            SharedMemory object
        """
        with self.lock:
            if name in self.segments:
                raise ValueError(f"Shared memory segment already exists: {name}")
            
            # Create shared memory
            shm = mp.shared_memory.SharedMemory(
                name=name,
                create=True,
                size=size_bytes
            )
            
            # Store metadata
            self.segments[name] = SharedMemoryInfo(
                name=name,
                size_bytes=size_bytes,
                data_type=data_type,
                created_by=os.getpid()
            )
            
            self.logger.debug(
                f"Allocated shared memory: {name}, {size_bytes} bytes, type={data_type.value}"
            )
            
            return shm
    
    def attach(self, name: str) -> mp.shared_memory.SharedMemory:
        """
        Attach to existing shared memory segment.
        
        Args:
            name: Name of segment
        
        Returns:
            SharedMemory object
        """
        if name not in self.segments:
            # Try to attach anyway (might exist from another manager)
            try:
                shm = mp.shared_memory.SharedMemory(name=name)
                self.logger.debug(f"Attached to existing shared memory: {name}")
                return shm
            except FileNotFoundError:
                raise ValueError(f"Shared memory segment not found: {name}")
        
        shm = mp.shared_memory.SharedMemory(name=name)
        self.logger.debug(f"Attached to shared memory: {name}")
        
        return shm
    
    def deallocate(self, name: str):
        """
        Deallocate shared memory segment.
        
        Args:
            name: Name of segment
        """
        with self.lock:
            if name not in self.segments:
                self.logger.warning(f"Shared memory segment not found: {name}")
                return
            
            try:
                shm = mp.shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
                
                del self.segments[name]
                self.logger.debug(f"Deallocated shared memory: {name}")
            
            except FileNotFoundError:
                self.logger.warning(f"Shared memory already deallocated: {name}")
    
    def get_info(self, name: str) -> Optional[SharedMemoryInfo]:
        """
        Get metadata for shared memory segment.
        
        Args:
            name: Name of segment
        
        Returns:
            SharedMemoryInfo or None
        """
        return self.segments.get(name)
    
    def get_total_usage(self) -> int:
        """
        Get total shared memory usage in bytes.
        
        Returns:
            Total bytes allocated
        """
        with self.lock:
            return sum(info.size_bytes for info in self.segments.values())
    
    def cleanup_all(self):
        """Cleanup all shared memory segments."""
        with self.lock:
            segment_names = list(self.segments.keys())
        
        for name in segment_names:
            try:
                self.deallocate(name)
            except Exception as e:
                self.logger.error(f"Failed to cleanup {name}: {e}")


# ============================================================================
# SHARED NUMPY ARRAY
# ============================================================================

class SharedNumpyArray:
    """
    Shared memory NumPy array for inter-process communication.
    
    Optimized for large genomic arrays (quality scores, coverage, etc.).
    """
    
    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        create: bool = True
    ):
        """
        Initialize shared NumPy array.
        
        Args:
            name: Unique name for array
            shape: Array shape
            dtype: NumPy data type
            create: Create new (True) or attach to existing (False)
        """
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        # Calculate size
        self.size_bytes = int(np.prod(shape) * self.dtype.itemsize)
        
        if create:
            self.shm = mp.shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self.size_bytes
            )
        else:
            self.shm = mp.shared_memory.SharedMemory(name=name)
        
        # Create NumPy array view
        self.array = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self.shm.buf
        )
        
        self.logger = logging.getLogger(f'{__name__}.SharedNumpyArray')
    
    def __getitem__(self, key):
        """Get item from array."""
        return self.array[key]
    
    def __setitem__(self, key, value):
        """Set item in array."""
        self.array[key] = value
    
    def close(self):
        """Close shared memory (does not unlink)."""
        if hasattr(self, 'shm'):
            self.shm.close()
    
    def unlink(self):
        """Unlink shared memory (permanent deletion)."""
        if hasattr(self, 'shm'):
            self.shm.unlink()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    @classmethod
    def from_array(cls, name: str, array: np.ndarray) -> 'SharedNumpyArray':
        """
        Create shared array from existing NumPy array.
        
        Args:
            name: Name for shared array
            array: NumPy array to copy
        
        Returns:
            SharedNumpyArray instance
        """
        shared = cls(name, array.shape, array.dtype, create=True)
        np.copyto(shared.array, array)
        return shared


# ============================================================================
# SHARED DICTIONARY
# ============================================================================

class SharedDict:
    """
    Shared dictionary using Manager for process-safe access.
    
    Suitable for configuration, metadata, and small data structures.
    """
    
    def __init__(self, manager: Optional[mp.Manager] = None):
        """
        Initialize shared dictionary.
        
        Args:
            manager: Multiprocessing Manager (creates one if None)
        """
        if manager is None:
            self.manager = mp.Manager()
            self.owns_manager = True
        else:
            self.manager = manager
            self.owns_manager = False
        
        self.dict = self.manager.dict()
        self.lock = self.manager.Lock()
    
    def __getitem__(self, key):
        """Get item from dictionary."""
        with self.lock:
            return self.dict[key]
    
    def __setitem__(self, key, value):
        """Set item in dictionary."""
        with self.lock:
            self.dict[key] = value
    
    def __delitem__(self, key):
        """Delete item from dictionary."""
        with self.lock:
            del self.dict[key]
    
    def __contains__(self, key):
        """Check if key exists."""
        with self.lock:
            return key in self.dict
    
    def get(self, key, default=None):
        """Get item with default."""
        with self.lock:
            return self.dict.get(key, default)
    
    def keys(self):
        """Get dictionary keys."""
        with self.lock:
            return list(self.dict.keys())
    
    def values(self):
        """Get dictionary values."""
        with self.lock:
            return list(self.dict.values())
    
    def items(self):
        """Get dictionary items."""
        with self.lock:
            return list(self.dict.items())
    
    def update(self, other: dict):
        """Update dictionary with another dict."""
        with self.lock:
            self.dict.update(other)
    
    def clear(self):
        """Clear dictionary."""
        with self.lock:
            self.dict.clear()
    
    def __len__(self):
        """Get dictionary length."""
        with self.lock:
            return len(self.dict)


# ============================================================================
# SHARED COUNTER
# ============================================================================

class SharedCounter:
    """
    Thread-safe and process-safe counter.
    
    Optimized for tracking progress across multiple workers.
    """
    
    def __init__(self, initial_value: int = 0):
        """
        Initialize shared counter.
        
        Args:
            initial_value: Initial counter value
        """
        self.value = mp.Value('i', initial_value)
        self.lock = mp.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """
        Increment counter.
        
        Args:
            amount: Amount to increment
        
        Returns:
            New value
        """
        with self.lock:
            self.value.value += amount
            return self.value.value
    
    def decrement(self, amount: int = 1) -> int:
        """
        Decrement counter.
        
        Args:
            amount: Amount to decrement
        
        Returns:
            New value
        """
        with self.lock:
            self.value.value -= amount
            return self.value.value
    
    def get(self) -> int:
        """
        Get current value.
        
        Returns:
            Current counter value
        """
        with self.lock:
            return self.value.value
    
    def set(self, value: int):
        """
        Set counter value.
        
        Args:
            value: New value
        """
        with self.lock:
            self.value.value = value
    
    def reset(self):
        """Reset counter to zero."""
        self.set(0)


# ============================================================================
# SHARED QUEUE
# ============================================================================

class SharedQueue:
    """
    Process-safe queue for producer-consumer patterns.
    
    Optimized for genomic data chunks between pipeline stages.
    """
    
    def __init__(self, maxsize: int = 0):
        """
        Initialize shared queue.
        
        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self.queue = mp.Queue(maxsize=maxsize)
        self.logger = logging.getLogger(f'{__name__}.SharedQueue')
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None):
        """
        Put item in queue.
        
        Args:
            item: Item to add
            block: Block if queue is full
            timeout: Timeout in seconds
        """
        self.queue.put(item, block=block, timeout=timeout)
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """
        Get item from queue.
        
        Args:
            block: Block if queue is empty
            timeout: Timeout in seconds
        
        Returns:
            Item from queue
        """
        return self.queue.get(block=block, timeout=timeout)
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get approximate queue size."""
        return self.queue.qsize()
    
    def put_nowait(self, item: Any):
        """Put item without blocking."""
        self.queue.put_nowait(item)
    
    def get_nowait(self) -> Any:
        """Get item without blocking."""
        return self.queue.get_nowait()


# ============================================================================
# MEMORY-MAPPED FILE
# ============================================================================

class MemoryMappedFile:
    """
    Memory-mapped file for shared data persistence.
    
    Combines shared memory performance with disk persistence.
    """
    
    def __init__(
        self,
        filepath: Path,
        size_bytes: int,
        create: bool = True
    ):
        """
        Initialize memory-mapped file.
        
        Args:
            filepath: Path to file
            size_bytes: File size in bytes
            create: Create new file (True) or open existing (False)
        """
        self.filepath = Path(filepath)
        self.size_bytes = size_bytes
        
        if create:
            # Create file
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, 'wb') as f:
                f.write(b'\x00' * size_bytes)
        
        # Open file
        self.file_handle = open(self.filepath, 'r+b')
        
        # Create memory map
        self.mmap = mmap.mmap(
            self.file_handle.fileno(),
            size_bytes,
            access=mmap.ACCESS_WRITE
        )
        
        self.logger = logging.getLogger(f'{__name__}.MemoryMappedFile')
    
    def write(self, offset: int, data: bytes):
        """
        Write data at offset.
        
        Args:
            offset: Byte offset
            data: Data to write
        """
        self.mmap.seek(offset)
        self.mmap.write(data)
    
    def read(self, offset: int, size: int) -> bytes:
        """
        Read data from offset.
        
        Args:
            offset: Byte offset
            size: Number of bytes to read
        
        Returns:
            Data bytes
        """
        self.mmap.seek(offset)
        return self.mmap.read(size)
    
    def flush(self):
        """Flush changes to disk."""
        self.mmap.flush()
    
    def close(self):
        """Close memory map and file."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file_handle'):
            self.file_handle.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ============================================================================
# SHARED BUFFER POOL
# ============================================================================

class SharedBufferPool:
    """
    Pool of reusable shared memory buffers.
    
    Reduces allocation overhead for temporary buffers.
    """
    
    def __init__(
        self,
        buffer_size: int,
        pool_size: int = 10
    ):
        """
        Initialize buffer pool.
        
        Args:
            buffer_size: Size of each buffer in bytes
            pool_size: Number of buffers in pool
        """
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        
        self.manager = mp.Manager()
        self.available_buffers = self.manager.Queue()
        self.lock = self.manager.Lock()
        
        # Create buffers
        for i in range(pool_size):
            buffer_name = f"buffer_pool_{id(self)}_{i}"
            shm = mp.shared_memory.SharedMemory(
                name=buffer_name,
                create=True,
                size=buffer_size
            )
            self.available_buffers.put((buffer_name, shm))
        
        self.logger = logging.getLogger(f'{__name__}.SharedBufferPool')
    
    def acquire(self, timeout: Optional[float] = None) -> Tuple[str, mp.shared_memory.SharedMemory]:
        """
        Acquire buffer from pool.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Tuple of (buffer_name, SharedMemory)
        """
        try:
            return self.available_buffers.get(timeout=timeout)
        except:
            raise RuntimeError("No buffers available in pool")
    
    def release(self, buffer_name: str, shm: mp.shared_memory.SharedMemory):
        """
        Release buffer back to pool.
        
        Args:
            buffer_name: Buffer name
            shm: SharedMemory object
        """
        self.available_buffers.put((buffer_name, shm))
    
    def cleanup(self):
        """Cleanup all buffers in pool."""
        while not self.available_buffers.empty():
            try:
                buffer_name, shm = self.available_buffers.get_nowait()
                shm.close()
                shm.unlink()
            except:
                pass


# ============================================================================
# PARALLEL PROCESSING UTILITIES
# ============================================================================

def parallel_map_with_shared_memory(
    func: callable,
    items: List[Any],
    shared_data: Dict[str, Any],
    num_workers: int = 4
) -> List[Any]:
    """
    Parallel map with shared memory data.
    
    Args:
        func: Function to apply (must accept item and shared_data)
        items: List of items to process
        shared_data: Dictionary of shared data structures
        num_workers: Number of worker processes
    
    Returns:
        List of results
    """
    def worker_func(item):
        return func(item, shared_data)
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, items)
    
    return results


def chunked_parallel_processing(
    data_array: np.ndarray,
    process_func: callable,
    chunk_size: int,
    num_workers: int = 4
) -> np.ndarray:
    """
    Process NumPy array in parallel chunks using shared memory.
    
    Args:
        data_array: Input array
        process_func: Function to process each chunk
        chunk_size: Size of each chunk
        num_workers: Number of worker processes
    
    Returns:
        Processed array
    """
    # Create shared input array
    shared_input = SharedNumpyArray.from_array(
        name=f"input_{id(data_array)}",
        array=data_array
    )
    
    # Create shared output array
    shared_output = SharedNumpyArray(
        name=f"output_{id(data_array)}",
        shape=data_array.shape,
        dtype=data_array.dtype,
        create=True
    )
    
    # Split into chunks
    num_chunks = (len(data_array) + chunk_size - 1) // chunk_size
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, len(data_array))) 
              for i in range(num_chunks)]
    
    def process_chunk(chunk_range):
        start, end = chunk_range
        
        # Attach to shared memory
        input_array = SharedNumpyArray(
            name=shared_input.name,
            shape=shared_input.shape,
            dtype=shared_input.dtype,
            create=False
        )
        
        output_array = SharedNumpyArray(
            name=shared_output.name,
            shape=shared_output.shape,
            dtype=shared_output.dtype,
            create=False
        )
        
        # Process chunk
        result = process_func(input_array.array[start:end])
        output_array.array[start:end] = result
        
        # Cleanup
        input_array.close()
        output_array.close()
    
    # Process in parallel
    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_chunk, chunks)
    
    # Copy result
    result = shared_output.array.copy()
    
    # Cleanup
    shared_input.unlink()
    shared_output.unlink()
    
    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_shared_memory_size(
    data_structure: Union[np.ndarray, list, dict]
) -> int:
    """
    Estimate memory size for data structure.
    
    Args:
        data_structure: Data to estimate
    
    Returns:
        Estimated size in bytes
    """
    if isinstance(data_structure, np.ndarray):
        return data_structure.nbytes
    elif isinstance(data_structure, (list, tuple)):
        return len(pickle.dumps(data_structure))
    elif isinstance(data_structure, dict):
        return len(pickle.dumps(data_structure))
    else:
        return len(pickle.dumps(data_structure))


def get_system_shared_memory_limit() -> int:
    """
    Get system shared memory limit.
    
    Returns:
        Maximum shared memory in bytes
    """
    try:
        # Linux
        with open('/proc/sys/kernel/shmmax', 'r') as f:
            return int(f.read().strip())
    except:
        # Fallback to conservative estimate
        import psutil
        return int(psutil.virtual_memory().total * 0.5)


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Example usage of shared memory system."""
    
    print("=== Shared Memory Examples ===\n")
    
    # Example 1: Shared NumPy array
    print("1. Shared NumPy Array")
    with SharedNumpyArray(
        name="test_array",
        shape=(1000, 100),
        dtype=np.float32,
        create=True
    ) as shared_arr:
        # Write data
        shared_arr.array[:] = np.random.randn(1000, 100)
        print(f"   Created shared array: shape={shared_arr.shape}, size={shared_arr.size_bytes} bytes")
        
        # Access from another process would use:
        # shared_arr2 = SharedNumpyArray("test_array", shape=(1000, 100), dtype=np.float32, create=False)
    
    # Example 2: Shared dictionary
    print("\n2. Shared Dictionary")
    shared_dict = SharedDict()
    shared_dict['sample_id'] = 'SAMPLE001'
    shared_dict['stage'] = 'variant_calling'
    shared_dict['progress'] = 0.75
    print(f"   Shared dict: {dict(shared_dict.items())}")
    
    # Example 3: Shared counter
    print("\n3. Shared Counter")
    counter = SharedCounter(initial_value=0)
    for i in range(10):
        new_value = counter.increment()
    print(f"   Counter value: {counter.get()}")
    
    # Example 4: Shared queue
    print("\n4. Shared Queue")
    queue = SharedQueue(maxsize=100)
    for i in range(5):
        queue.put(f"item_{i}")
    print(f"   Queue size: {queue.qsize()}")
    print(f"   First item: {queue.get()}")
    
    # Example 5: Parallel processing with shared memory
    print("\n5. Parallel Processing")
    data = np.random.randn(10000)
    
    def square_chunk(chunk):
        return chunk ** 2
    
    result = chunked_parallel_processing(
        data_array=data,
        process_func=square_chunk,
        chunk_size=1000,
        num_workers=4
    )
    
    print(f"   Processed {len(result)} elements in parallel")
    print(f"   Result sample: {result[:5]}")
    
    # Example 6: Memory estimation
    print("\n6. Memory Estimation")
    test_array = np.zeros((1000, 1000), dtype=np.float64)
    estimated_size = estimate_shared_memory_size(test_array)
    print(f"   Estimated size for 1000x1000 array: {estimated_size / 1024 / 1024:.2f} MB")
    
    system_limit = get_system_shared_memory_limit()
    print(f"   System shared memory limit: {system_limit / 1024 / 1024 / 1024:.2f} GB")


if __name__ == '__main__':
    main()
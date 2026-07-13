"""
I/O Management System for Genomic Data

Production-grade I/O system optimized for large genomic files.
Implements memory-mapped reading, async streaming, and compression handling.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import asyncio
import gzip
import hashlib
import mmap
import os
import struct
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Tuple, Union, BinaryIO, TextIO
import logging


# ============================================================================
# ENUMERATIONS
# ============================================================================

class FileFormat(Enum):
    """Supported genomic file formats."""
    FASTQ = "FASTQ"
    FASTQ_GZ = "FASTQ_GZ"
    BAM = "BAM"
    CRAM = "CRAM"
    SAM = "SAM"
    VCF = "VCF"
    VCF_GZ = "VCF_GZ"
    BCF = "BCF"
    BED = "BED"
    GTF = "GTF"
    GFF = "GFF"
    FASTA = "FASTA"
    FASTA_GZ = "FASTA_GZ"
    UNKNOWN = "UNKNOWN"


class CompressionType(Enum):
    """Compression types."""
    NONE = "NONE"
    GZIP = "GZIP"
    BGZIP = "BGZIP"
    BZIP2 = "BZIP2"
    XZ = "XZ"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FileMetadata:
    """Metadata for genomic files."""
    path: Path
    format: FileFormat
    compression: CompressionType
    size_bytes: int
    line_count: Optional[int] = None
    record_count: Optional[int] = None
    checksum_sha256: Optional[str] = None
    is_indexed: bool = False
    index_path: Optional[Path] = None


# ============================================================================
# FORMAT DETECTION
# ============================================================================

class FormatDetector:
    """Detect genomic file formats from magic bytes and extensions."""
    
    # Magic byte signatures
    MAGIC_BYTES = {
        b'\x1f\x8b': CompressionType.GZIP,  # gzip/bgzip
        b'BZ': CompressionType.BZIP2,
        b'\xfd7zXZ\x00': CompressionType.XZ,
        b'BAM\x01': FileFormat.BAM,
        b'CRAM': FileFormat.CRAM,
    }
    
    @classmethod
    def detect_format(cls, file_path: Path) -> Tuple[FileFormat, CompressionType]:
        """
        Detect file format and compression type.
        
        Args:
            file_path: Path to file
        
        Returns:
            Tuple of (FileFormat, CompressionType)
        """
        if not file_path.exists():
            return FileFormat.UNKNOWN, CompressionType.NONE
        
        # Check extension first (fast path)
        extension = file_path.suffix.lower()
        
        if extension == '.gz':
            # Need to check inner format
            stem_ext = file_path.stem.split('.')[-1].lower()
            if stem_ext == 'fastq' or stem_ext == 'fq':
                return FileFormat.FASTQ_GZ, CompressionType.BGZIP
            elif stem_ext == 'vcf':
                return FileFormat.VCF_GZ, CompressionType.BGZIP
            elif stem_ext in ('fasta', 'fa', 'fna'):
                return FileFormat.FASTA_GZ, CompressionType.GZIP
        
        elif extension in ('.fastq', '.fq'):
            return FileFormat.FASTQ, CompressionType.NONE
        elif extension == '.bam':
            return FileFormat.BAM, CompressionType.NONE
        elif extension == '.cram':
            return FileFormat.CRAM, CompressionType.NONE
        elif extension == '.sam':
            return FileFormat.SAM, CompressionType.NONE
        elif extension == '.vcf':
            return FileFormat.VCF, CompressionType.NONE
        elif extension == '.bcf':
            return FileFormat.BCF, CompressionType.NONE
        elif extension == '.bed':
            return FileFormat.BED, CompressionType.NONE
        elif extension == '.gtf':
            return FileFormat.GTF, CompressionType.NONE
        elif extension == '.gff':
            return FileFormat.GFF, CompressionType.NONE
        elif extension in ('.fasta', '.fa', '.fna'):
            return FileFormat.FASTA, CompressionType.NONE
        
        # Check magic bytes
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(8)
                
                # Check compression
                if magic[:2] == b'\x1f\x8b':
                    # Decompress first chunk to check format
                    f.seek(0)
                    with gzip.open(f, 'rt') as gz:
                        first_line = gz.readline()
                        if first_line.startswith('@'):
                            return FileFormat.FASTQ_GZ, CompressionType.BGZIP
                        elif first_line.startswith('##fileformat=VCF'):
                            return FileFormat.VCF_GZ, CompressionType.BGZIP
                        elif first_line.startswith('>'):
                            return FileFormat.FASTA_GZ, CompressionType.GZIP
                
                elif magic[:4] == b'BAM\x01':
                    return FileFormat.BAM, CompressionType.NONE
                elif magic[:4] == b'CRAM':
                    return FileFormat.CRAM, CompressionType.NONE
                
                # Check text formats
                f.seek(0)
                first_line = f.readline(1024).decode('utf-8', errors='ignore')
                
                if first_line.startswith('@'):
                    return FileFormat.FASTQ, CompressionType.NONE
                elif first_line.startswith('##fileformat=VCF'):
                    return FileFormat.VCF, CompressionType.NONE
                elif first_line.startswith('>'):
                    return FileFormat.FASTA, CompressionType.NONE
                elif '\t' in first_line and not first_line.startswith('#'):
                    # Could be BED or SAM
                    fields = first_line.split('\t')
                    if len(fields) >= 11:  # SAM has 11+ fields
                        return FileFormat.SAM, CompressionType.NONE
                    elif len(fields) >= 3:  # BED has 3+ fields
                        return FileFormat.BED, CompressionType.NONE
        
        except Exception:
            pass
        
        return FileFormat.UNKNOWN, CompressionType.NONE
    
    @classmethod
    def get_metadata(cls, file_path: Path) -> FileMetadata:
        """
        Extract file metadata.
        
        Args:
            file_path: Path to file
        
        Returns:
            FileMetadata instance
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        format_type, compression = cls.detect_format(file_path)
        size_bytes = file_path.stat().st_size
        
        # Check for index files
        is_indexed = False
        index_path = None
        
        if format_type == FileFormat.BAM:
            bai_path = file_path.with_suffix('.bam.bai')
            if bai_path.exists():
                is_indexed = True
                index_path = bai_path
        
        elif format_type in (FileFormat.VCF_GZ, FileFormat.BCF):
            tbi_path = Path(str(file_path) + '.tbi')
            csi_path = Path(str(file_path) + '.csi')
            if tbi_path.exists():
                is_indexed = True
                index_path = tbi_path
            elif csi_path.exists():
                is_indexed = True
                index_path = csi_path
        
        return FileMetadata(
            path=file_path,
            format=format_type,
            compression=compression,
            size_bytes=size_bytes,
            is_indexed=is_indexed,
            index_path=index_path
        )


# ============================================================================
# BASE READER INTERFACE
# ============================================================================

class GenomicFileReader(ABC):
    """Abstract base class for genomic file readers."""
    
    def __init__(self, file_path: Path):
        """
        Initialize reader.
        
        Args:
            file_path: Path to genomic file
        """
        self.file_path = Path(file_path)
        self.metadata = FormatDetector.get_metadata(self.file_path)
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    @abstractmethod
    def read_records(self, chunk_size: int = 10000) -> Iterator:
        """
        Read records from file.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            Records from file
        """
        pass
    
    @abstractmethod
    async def read_records_async(self, chunk_size: int = 10000) -> AsyncIterator:
        """
        Asynchronously read records from file.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            Records from file
        """
        pass
    
    def compute_checksum(self, algorithm: str = 'sha256') -> str:
        """
        Compute file checksum.
        
        Args:
            algorithm: Hash algorithm (sha256, md5)
        
        Returns:
            Hexadecimal checksum string
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(self.file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()


# ============================================================================
# MEMORY-MAPPED FILE READER
# ============================================================================

class MemoryMappedReader:
    """
    Memory-mapped file reader for efficient random access.
    
    Optimized for large genomic files (>1GB) with random access patterns.
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize memory-mapped reader.
        
        Args:
            file_path: Path to file
        """
        self.file_path = Path(file_path)
        self.file_handle = None
        self.mmap_obj = None
        self.logger = logging.getLogger(f'{__name__}.MemoryMappedReader')
    
    def __enter__(self):
        """Open file for memory mapping."""
        self.file_handle = open(self.file_path, 'rb')
        self.mmap_obj = mmap.mmap(
            self.file_handle.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close memory map and file."""
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.file_handle:
            self.file_handle.close()
    
    def read_chunk(self, offset: int, size: int) -> bytes:
        """
        Read chunk at specific offset.
        
        Args:
            offset: Byte offset
            size: Number of bytes to read
        
        Returns:
            Bytes from file
        """
        if not self.mmap_obj:
            raise RuntimeError("Memory map not initialized. Use context manager.")
        
        self.mmap_obj.seek(offset)
        return self.mmap_obj.read(size)
    
    def read_line(self, offset: int) -> Tuple[bytes, int]:
        """
        Read line starting at offset.
        
        Args:
            offset: Starting byte offset
        
        Returns:
            Tuple of (line_bytes, next_offset)
        """
        if not self.mmap_obj:
            raise RuntimeError("Memory map not initialized. Use context manager.")
        
        self.mmap_obj.seek(offset)
        line = self.mmap_obj.readline()
        next_offset = self.mmap_obj.tell()
        
        return line, next_offset
    
    def find_line_offsets(self, max_lines: Optional[int] = None) -> list:
        """
        Build index of line offsets for random access.
        
        Args:
            max_lines: Maximum number of lines to index (None = all)
        
        Returns:
            List of byte offsets for each line
        """
        if not self.mmap_obj:
            raise RuntimeError("Memory map not initialized. Use context manager.")
        
        offsets = [0]
        self.mmap_obj.seek(0)
        
        line_count = 0
        while True:
            line = self.mmap_obj.readline()
            if not line:
                break
            
            offsets.append(self.mmap_obj.tell())
            line_count += 1
            
            if max_lines and line_count >= max_lines:
                break
        
        return offsets


# ============================================================================
# FASTQ READER
# ============================================================================

class FastqReader(GenomicFileReader):
    """
    FASTQ file reader with support for gzipped files.
    
    Efficiently reads FASTQ records (4 lines per record).
    """
    
    def read_records(self, chunk_size: int = 10000) -> Iterator[dict]:
        """
        Read FASTQ records.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            Dictionary with keys: id, sequence, plus, quality
        """
        open_func = gzip.open if self.metadata.compression == CompressionType.BGZIP else open
        
        with open_func(self.file_path, 'rt') as f:
            chunk = []
            
            while True:
                # Read 4 lines (one FASTQ record)
                header = f.readline().strip()
                if not header:
                    if chunk:
                        yield chunk
                    break
                
                sequence = f.readline().strip()
                plus = f.readline().strip()
                quality = f.readline().strip()
                
                if not (sequence and plus and quality):
                    self.logger.warning(f"Incomplete FASTQ record at {header}")
                    continue
                
                record = {
                    'id': header[1:] if header.startswith('@') else header,
                    'sequence': sequence,
                    'plus': plus,
                    'quality': quality
                }
                
                chunk.append(record)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    
    async def read_records_async(self, chunk_size: int = 10000) -> AsyncIterator[dict]:
        """
        Asynchronously read FASTQ records.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            List of FASTQ record dictionaries
        """
        loop = asyncio.get_event_loop()
        
        for chunk in await loop.run_in_executor(None, self._read_all_chunks, chunk_size):
            yield chunk
            await asyncio.sleep(0)  # Yield control
    
    def _read_all_chunks(self, chunk_size: int) -> list:
        """Helper to read all chunks synchronously."""
        return list(self.read_records(chunk_size))


# ============================================================================
# VCF READER
# ============================================================================

class VcfReader(GenomicFileReader):
    """
    VCF file reader with header parsing and record streaming.
    
    Supports both plain and bgzipped VCF files.
    """
    
    def __init__(self, file_path: Path):
        """Initialize VCF reader."""
        super().__init__(file_path)
        self.header_lines = []
        self.column_names = []
        self._parse_header()
    
    def _parse_header(self):
        """Parse VCF header lines."""
        open_func = gzip.open if self.metadata.compression == CompressionType.BGZIP else open
        
        with open_func(self.file_path, 'rt') as f:
            for line in f:
                if line.startswith('##'):
                    self.header_lines.append(line.strip())
                elif line.startswith('#CHROM'):
                    self.column_names = line.strip().split('\t')
                    break
    
    def read_records(self, chunk_size: int = 10000) -> Iterator[dict]:
        """
        Read VCF records.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            List of VCF record dictionaries
        """
        open_func = gzip.open if self.metadata.compression == CompressionType.BGZIP else open
        
        with open_func(self.file_path, 'rt') as f:
            # Skip header
            for line in f:
                if line.startswith('#CHROM'):
                    break
            
            chunk = []
            
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                
                if len(fields) < 8:
                    self.logger.warning(f"Invalid VCF line: {line[:100]}")
                    continue
                
                record = {
                    'CHROM': fields[0],
                    'POS': int(fields[1]),
                    'ID': fields[2],
                    'REF': fields[3],
                    'ALT': fields[4],
                    'QUAL': fields[5],
                    'FILTER': fields[6],
                    'INFO': fields[7]
                }
                
                # Add FORMAT and sample columns if present
                if len(fields) > 8:
                    record['FORMAT'] = fields[8]
                    record['SAMPLES'] = fields[9:]
                
                chunk.append(record)
                
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
            
            if chunk:
                yield chunk
    
    async def read_records_async(self, chunk_size: int = 10000) -> AsyncIterator[dict]:
        """
        Asynchronously read VCF records.
        
        Args:
            chunk_size: Number of records per chunk
        
        Yields:
            List of VCF record dictionaries
        """
        loop = asyncio.get_event_loop()
        
        for chunk in await loop.run_in_executor(None, self._read_all_chunks, chunk_size):
            yield chunk
            await asyncio.sleep(0)
    
    def _read_all_chunks(self, chunk_size: int) -> list:
        """Helper to read all chunks synchronously."""
        return list(self.read_records(chunk_size))
    
    def get_sample_names(self) -> list:
        """
        Get sample names from VCF header.
        
        Returns:
            List of sample names
        """
        if len(self.column_names) > 9:
            return self.column_names[9:]
        return []


# ============================================================================
# COMPRESSED STREAM READER
# ============================================================================

class CompressedStreamReader:
    """
    Streaming reader for compressed files with parallel decompression.
    
    Uses external tools (pigz, bgzip) for faster decompression.
    """
    
    def __init__(
        self,
        file_path: Path,
        compression_type: CompressionType = CompressionType.GZIP,
        threads: int = 4
    ):
        """
        Initialize compressed stream reader.
        
        Args:
            file_path: Path to compressed file
            compression_type: Compression type
            threads: Number of decompression threads
        """
        self.file_path = Path(file_path)
        self.compression_type = compression_type
        self.threads = threads
        self.logger = logging.getLogger(f'{__name__}.CompressedStreamReader')
    
    def stream_lines(self, buffer_size: int = 65536) -> Iterator[str]:
        """
        Stream lines from compressed file with parallel decompression.
        
        Args:
            buffer_size: Buffer size in bytes
        
        Yields:
            Lines from decompressed file
        """
        if self.compression_type == CompressionType.GZIP:
            # Use pigz for parallel decompression if available
            try:
                process = subprocess.Popen(
                    ['pigz', '-dc', '-p', str(self.threads), str(self.file_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=buffer_size
                )
            except FileNotFoundError:
                # Fallback to gzip
                process = subprocess.Popen(
                    ['gzip', '-dc', str(self.file_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=buffer_size
                )
            
            for line in process.stdout:
                yield line.decode('utf-8')
            
            process.wait()
            
            if process.returncode != 0:
                stderr = process.stderr.read().decode('utf-8')
                raise RuntimeError(f"Decompression failed: {stderr}")
        
        else:
            # Use Python gzip for other compression types
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    yield line


# ============================================================================
# PRODUCER-CONSUMER BUFFER
# ============================================================================

class AsyncBuffer:
    """
    Async producer-consumer buffer for streaming genomic data.
    
    Enables pipeline stages to run in parallel with buffered I/O.
    """
    
    def __init__(self, maxsize: int = 1000):
        """
        Initialize async buffer.
        
        Args:
            maxsize: Maximum buffer size
        """
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.is_complete = False
        self.logger = logging.getLogger(f'{__name__}.AsyncBuffer')
    
    async def put(self, item):
        """
        Put item in buffer.
        
        Args:
            item: Item to buffer
        """
        await self.queue.put(item)
    
    async def get(self):
        """
        Get item from buffer.
        
        Returns:
            Item from buffer or None if complete
        """
        if self.queue.empty() and self.is_complete:
            return None
        
        return await self.queue.get()
    
    def mark_complete(self):
        """Mark buffer as complete (no more items will be added)."""
        self.is_complete = True
    
    async def producer(self, reader: GenomicFileReader, chunk_size: int = 10000):
        """
        Producer coroutine that reads from file and fills buffer.
        
        Args:
            reader: GenomicFileReader instance
            chunk_size: Records per chunk
        """
        try:
            async for chunk in reader.read_records_async(chunk_size):
                await self.put(chunk)
        finally:
            self.mark_complete()
    
    async def consumer(self, processor):
        """
        Consumer coroutine that processes items from buffer.
        
        Args:
            processor: Callable that processes each chunk
        """
        while True:
            item = await self.get()
            if item is None:
                break
            
            await processor(item)


# ============================================================================
# FILE WRITER
# ============================================================================

class GenomicFileWriter:
    """
    Generic writer for genomic files with compression support.
    """
    
    def __init__(
        self,
        file_path: Path,
        compression: CompressionType = CompressionType.NONE,
        compression_level: int = 6
    ):
        """
        Initialize file writer.
        
        Args:
            file_path: Output file path
            compression: Compression type
            compression_level: Compression level (1-9)
        """
        self.file_path = Path(file_path)
        self.compression = compression
        self.compression_level = compression_level
        self.file_handle = None
        self.logger = logging.getLogger(f'{__name__}.GenomicFileWriter')
    
    def __enter__(self):
        """Open file for writing."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.compression == CompressionType.GZIP:
            self.file_handle = gzip.open(
                self.file_path,
                'wt',
                compresslevel=self.compression_level
            )
        else:
            self.file_handle = open(self.file_path, 'w')
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close file handle."""
        if self.file_handle:
            self.file_handle.close()
    
    def write(self, data: str):
        """
        Write data to file.
        
        Args:
            data: String data to write
        """
        if not self.file_handle:
            raise RuntimeError("File not opened. Use context manager.")
        
        self.file_handle.write(data)
    
    def writelines(self, lines: list):
        """
        Write multiple lines to file.
        
        Args:
            lines: List of strings to write
        """
        if not self.file_handle:
            raise RuntimeError("File not opened. Use context manager.")
        
        self.file_handle.writelines(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_lines(file_path: Path, approximate: bool = False) -> int:
    """
    Count lines in file (optimized for large files).
    
    Args:
        file_path: Path to file
        approximate: Use sampling for very large files
    
    Returns:
        Number of lines
    """
    if approximate and file_path.stat().st_size > 1_000_000_000:  # >1GB
        # Sample first 10MB
        sample_size = 10 * 1024 * 1024
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
            lines_in_sample = sample.count(b'\n')
            
            total_size = file_path.stat().st_size
            estimated_lines = int((lines_in_sample / sample_size) * total_size)
            
            return estimated_lines
    
    # Exact count
    line_count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            line_count += 1
    
    return line_count


def split_file_by_size(
    file_path: Path,
    output_dir: Path,
    chunk_size_mb: int = 100
) -> list:
    """
    Split large file into smaller chunks.
    
    Args:
        file_path: Input file
        output_dir: Output directory for chunks
        chunk_size_mb: Size of each chunk in MB
    
    Returns:
        List of chunk file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_index = 0
    
    with open(file_path, 'rb') as infile:
        while True:
            chunk_data = infile.read(chunk_size_bytes)
            if not chunk_data:
                break
            
            chunk_path = output_dir / f"{file_path.stem}_chunk{chunk_index:04d}{file_path.suffix}"
            
            with open(chunk_path, 'wb') as outfile:
                outfile.write(chunk_data)
            
            chunk_paths.append(chunk_path)
            chunk_index += 1
    
    return chunk_paths


async def parallel_file_processing(
    input_files: list,
    processor: callable,
    max_concurrent: int = 4
) -> list:
    """
    Process multiple files in parallel.
    
    Args:
        input_files: List of input file paths
        processor: Async function to process each file
        max_concurrent: Maximum concurrent file processors
    
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(file_path):
        async with semaphore:
            return await processor(file_path)
    
    tasks = [process_with_limit(f) for f in input_files]
    results = await asyncio.gather(*tasks)
    
    return results


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

async def main():
    """Example usage of I/O management system."""
    
    # Example 1: Memory-mapped reading
    test_file = Path("/data/test.txt")
    
    if test_file.exists():
        with MemoryMappedReader(test_file) as reader:
            # Build line index
            offsets = reader.find_line_offsets(max_lines=1000)
            print(f"Indexed {len(offsets)} lines")
            
            # Random access
            line, next_offset = reader.read_line(offsets[100])
            print(f"Line 100: {line.decode()}")
    
    # Example 2: FASTQ streaming
    fastq_file = Path("/data/sample.fastq.gz")
    
    if fastq_file.exists():
        reader = FastqReader(fastq_file)
        
        for chunk in reader.read_records(chunk_size=1000):
            print(f"Processing {len(chunk)} FASTQ records")
            break  # Process first chunk only
    
    # Example 3: VCF reading
    vcf_file = Path("/data/variants.vcf.gz")
    
    if vcf_file.exists():
        reader = VcfReader(vcf_file)
        print(f"VCF samples: {reader.get_sample_names()}")
        
        for chunk in reader.read_records(chunk_size=1000):
            print(f"Processing {len(chunk)} variants")
            break
    
    # Example 4: Async producer-consumer
    async def process_chunk(chunk):
        """Example processor."""
        await asyncio.sleep(0.1)  # Simulate processing
        print(f"Processed {len(chunk)} records")
    
    if vcf_file.exists():
        reader = VcfReader(vcf_file)
        buffer = AsyncBuffer(maxsize=10)
        
        # Start producer and consumer
        producer_task = asyncio.create_task(buffer.producer(reader, chunk_size=1000))
        consumer_task = asyncio.create_task(buffer.consumer(process_chunk))
        
        await producer_task
        await consumer_task
        
        print("Async processing complete")


if __name__ == '__main__':
    asyncio.run(main())
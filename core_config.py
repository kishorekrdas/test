"""
Centralized Configuration Management System

Production-grade configuration with validation, versioning, and audit trails.
Implements hierarchical configuration with base, site, and run-specific overrides.

Compliance: ICMR/NABL aligned, ACMG/AMP ready
Author: Clinical Bioinformatics Engineering Team
License: Proprietary - Clinical Diagnostic Use
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EnvironmentType(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class RegulatoryFramework(Enum):
    """Regulatory compliance frameworks."""
    ICMR_NABL = "ICMR_NABL"
    CLIA = "CLIA"
    CAP = "CAP"
    ISO_15189 = "ISO_15189"


class PipelineMode(Enum):
    """Pipeline execution modes."""
    GERMLINE_RARE_DISEASE = "germline_rare_disease"
    GERMLINE_CARRIER_SCREENING = "germline_carrier_screening"
    SOMATIC_CANCER = "somatic_cancer"
    PHARMACOGENOMICS = "pharmacogenomics"
    POPULATION_STUDY = "population_study"


class SequencingPlatform(Enum):
    """Supported sequencing platforms."""
    ILLUMINA_NOVASEQ = "illumina_novaseq"
    ILLUMINA_HISEQ = "illumina_hiseq"
    ILLUMINA_MISEQ = "illumina_miseq"
    MGI_DNBSEQ = "mgi_dnbseq"
    PACBIO_SEQUEL = "pacbio_sequel"
    OXFORD_NANOPORE = "oxford_nanopore"


class VariantCallerType(Enum):
    """Supported variant calling tools."""
    GATK_HAPLOTYPECALLER = "gatk_haplotypecaller"
    DEEPVARIANT = "deepvariant"
    FREEBAYES = "freebayes"
    OCTOPUS = "octopus"
    STRELKA2 = "strelka2"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DatasetManifest:
    """Metadata for genomic reference datasets."""
    name: str
    version: str
    source_url: str
    license: str
    date_downloaded: str
    file_path: Path
    checksum_sha256: str
    checksum_md5: Optional[str] = None
    file_size_bytes: int = 0
    description: str = ""
    citation: str = ""
    usage_restrictions: Optional[str] = None
    update_frequency: str = "on_release"
    last_verified: Optional[str] = None
    
    def validate_integrity(self) -> bool:
        """Verify file integrity using checksum."""
        if not self.file_path.exists():
            return False
        
        computed_checksum = compute_file_checksum(self.file_path)
        return computed_checksum == self.checksum_sha256
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['file_path'] = str(self.file_path)
        return result


@dataclass
class HardwareConfig:
    """Hardware resource configuration."""
    max_cpu_cores: int = 64
    max_memory_gb: float = 256.0
    available_gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    gpu_memory_gb_per_device: Dict[int, float] = field(default_factory=lambda: {i: 40.0 for i in range(7)})
    max_parallel_tasks: int = 8
    max_disk_io_concurrent: int = 3
    
    # Storage Configuration
    nvme_storage_path: Path = Path("/data/nvme")
    nvme_quota_tb: float = 10.0
    hdd_storage_path: Path = Path("/data/hdd")
    hdd_quota_tb: float = 100.0
    object_storage_path: Path = Path("/data/object")
    object_storage_quota_tb: float = 100.0
    temp_storage_path: Path = Path("/tmp/genomics")
    temp_retention_days: int = 7
    
    # Resource Margins
    cpu_system_reserve_percent: float = 10.0
    memory_system_reserve_percent: float = 10.0
    disk_safety_margin_gb: float = 100.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate hardware configuration."""
        errors = []
        
        if self.max_cpu_cores < 1:
            errors.append("max_cpu_cores must be >= 1")
        
        if self.max_memory_gb < 1.0:
            errors.append("max_memory_gb must be >= 1.0")
        
        if self.max_parallel_tasks > self.max_cpu_cores:
            errors.append("max_parallel_tasks cannot exceed max_cpu_cores")
        
        if not 0 <= self.cpu_system_reserve_percent <= 50:
            errors.append("cpu_system_reserve_percent must be between 0 and 50")
        
        if not 0 <= self.memory_system_reserve_percent <= 50:
            errors.append("memory_system_reserve_percent must be between 0 and 50")
        
        return len(errors) == 0, errors


@dataclass
class ReferenceGenomeConfig:
    """Reference genome configuration."""
    version: str = "GRCh38"
    fasta_path: Path = Path("/data/references/GRCh38.fasta")
    fai_path: Path = Path("/data/references/GRCh38.fasta.fai")
    dict_path: Path = Path("/data/references/GRCh38.dict")
    checksum: str = ""
    source: str = "Genome Reference Consortium"
    
    # Ensembl Gene Models
    ensembl_version: str = "110"
    ensembl_gtf_path: Path = Path("/data/references/ensembl_110.gtf")
    ensembl_gtf_checksum: str = ""


@dataclass
class PopulationDatabaseConfig:
    """Population allele frequency database configuration."""
    # Indian Population (PRIMARY)
    indigenomes_version: str = "v1.0"
    indigenomes_vcf_path: Path = Path("/data/population/IndiGenomes_v1.0.vcf.gz")
    indigenomes_checksum: str = ""
    indigenomes_common_af: float = 0.01  # 1% threshold
    
    # Asian Population (SECONDARY)
    genomeasia_version: str = "2023"
    genomeasia_vcf_path: Path = Path("/data/population/GenomeAsia100K.vcf.gz")
    genomeasia_checksum: str = ""
    genomeasia_common_af: float = 0.005  # 0.5% threshold
    
    # Global Population (TERTIARY)
    gnomad_version: str = "v4.0"
    gnomad_exomes_vcf_path: Path = Path("/data/population/gnomad.exomes.v4.0.vcf.gz")
    gnomad_genomes_vcf_path: Path = Path("/data/population/gnomad.genomes.v4.0.vcf.gz")
    gnomad_checksum: str = ""
    gnomad_common_af: float = 0.001  # 0.1% threshold
    gnomad_popmax_af_threshold: float = 0.001
    
    # Filtering Hierarchy
    population_hierarchy: List[str] = field(default_factory=lambda: ["IndiGenomes", "GenomeAsia", "gnomAD"])
    enable_population_stratification: bool = True
    default_population: str = "Indian"


@dataclass
class ClinicalDatabaseConfig:
    """Clinical variant database configuration."""
    # ClinVar
    clinvar_version: str = "2024-01"
    clinvar_vcf_path: Path = Path("/data/clinical/clinvar_20240101.vcf.gz")
    clinvar_date_downloaded: str = ""
    clinvar_checksum: str = ""
    
    # ClinGen
    clingen_version: str = "2024-01"
    clingen_dosage_path: Path = Path("/data/clinical/clingen_dosage.tsv")
    clingen_gene_validity_path: Path = Path("/data/clinical/clingen_gene_validity.tsv")
    
    # OMIM
    omim_version: str = "2024-01"
    omim_genemap_path: Path = Path("/data/clinical/genemap2.txt")
    
    # HGMD (if licensed)
    hgmd_licensed: bool = False
    hgmd_version: Optional[str] = None
    hgmd_path: Optional[Path] = None


@dataclass
class ValidationDatasetConfig:
    """Validation and benchmarking dataset configuration."""
    # GIAB Truth Sets
    giab_version: str = "v4.2.1"
    giab_samples: Dict[str, Path] = field(default_factory=lambda: {
        "HG001": Path("/data/validation/giab/HG001_GRCh38.vcf.gz"),
        "HG002": Path("/data/validation/giab/HG002_GRCh38.vcf.gz"),
        "HG003": Path("/data/validation/giab/HG003_GRCh38.vcf.gz"),
        "HG004": Path("/data/validation/giab/HG004_GRCh38.vcf.gz"),
        "HG005": Path("/data/validation/giab/HG005_GRCh38.vcf.gz"),
        "HG006": Path("/data/validation/giab/HG006_GRCh38.vcf.gz"),
        "HG007": Path("/data/validation/giab/HG007_GRCh38.vcf.gz")
    })
    giab_confidence_beds: Dict[str, Path] = field(default_factory=lambda: {
        "HG001": Path("/data/validation/giab/HG001_GRCh38_confident.bed"),
        "HG002": Path("/data/validation/giab/HG002_GRCh38_confident.bed"),
    })
    
    # Platinum Genomes
    platinum_genomes_path: Path = Path("/data/validation/platinum_genomes")


@dataclass
class ToolVersionConfig:
    """Tool versions and executable paths."""
    # Preprocessing
    fastqc_version: str = "0.12.1"
    fastqc_path: Path = Path("/usr/local/bin/fastqc")
    fastp_version: str = "0.23.4"
    fastp_path: Path = Path("/usr/local/bin/fastp")
    
    # Alignment
    bwa_mem2_version: str = "2.2.1"
    bwa_mem2_path: Path = Path("/usr/local/bin/bwa-mem2")
    
    # Sorting & Processing
    samtools_version: str = "1.19"
    samtools_path: Path = Path("/usr/local/bin/samtools")
    sambamba_version: str = "1.0.1"
    sambamba_path: Path = Path("/usr/local/bin/sambamba")
    
    # Variant Calling
    gatk_version: str = "4.5.0.0"
    gatk_path: Path = Path("/usr/local/bin/gatk")
    deepvariant_version: str = "1.6.0"
    deepvariant_path: Path = Path("/opt/deepvariant/bin/run_deepvariant")
    deepvariant_model_path: Path = Path("/opt/deepvariant/models")
    freebayes_version: str = "1.3.6"
    freebayes_path: Path = Path("/usr/local/bin/freebayes")
    
    # Structural Variants
    manta_version: str = "1.6.0"
    manta_path: Path = Path("/usr/local/bin/configManta.py")
    delly_version: str = "1.1.8"
    delly_path: Path = Path("/usr/local/bin/delly")
    lumpy_version: str = "0.3.1"
    lumpy_path: Path = Path("/usr/local/bin/lumpy")
    smoove_version: str = "0.2.8"
    smoove_path: Path = Path("/usr/local/bin/smoove")
    
    # CNV Calling
    cnvkit_version: str = "0.9.10"
    cnvkit_path: Path = Path("/usr/local/bin/cnvkit.py")
    gatk_gcnv_model_path: Path = Path("/data/models/gatk_gcnv")
    
    # Annotation
    vep_version: str = "110"
    vep_path: Path = Path("/usr/local/bin/vep")
    vep_cache_dir: Path = Path("/data/vep_cache")
    vep_plugins_dir: Path = Path("/data/vep_plugins")
    snpeff_version: str = "5.2"
    snpeff_path: Path = Path("/usr/local/bin/snpEff")
    snpeff_data_dir: Path = Path("/data/snpeff_data")
    
    # Normalization
    bcftools_version: str = "1.19"
    bcftools_path: Path = Path("/usr/local/bin/bcftools")
    vt_version: str = "0.57721"
    vt_path: Path = Path("/usr/local/bin/vt")
    
    # Compression
    bgzip_version: str = "1.19"
    bgzip_path: Path = Path("/usr/local/bin/bgzip")
    tabix_version: str = "1.19"
    tabix_path: Path = Path("/usr/local/bin/tabix")
    pigz_version: str = "2.8"
    pigz_path: Path = Path("/usr/local/bin/pigz")


@dataclass
class PipelineParametersConfig:
    """Pipeline execution parameters."""
    # Preprocessing
    fastp_min_read_length: int = 50
    fastp_min_quality: int = 20
    fastp_complexity_threshold: float = 30.0
    
    # Alignment
    bwa_threads: int = 16
    bwa_min_seed_length: int = 19
    bwa_match_score: int = 1
    bwa_mismatch_penalty: int = 4
    bwa_gap_open_penalty: int = 6
    bwa_gap_extension_penalty: int = 1
    bwa_read_group_template: str = "@RG\\tID:{sample}\\tSM:{sample}\\tPL:ILLUMINA\\tLB:{library}\\tPU:{flowcell}"
    
    # Base Quality Score Recalibration
    bqsr_known_sites: List[Path] = field(default_factory=list)
    bqsr_apply: bool = True
    
    # GATK HaplotypeCaller
    gatk_hc_min_base_quality: int = 20
    gatk_hc_min_mapping_quality: int = 20
    gatk_hc_ploidy: int = 2
    gatk_hc_stand_call_conf: float = 30.0
    gatk_hc_max_alternate_alleles: int = 6
    
    # DeepVariant
    deepvariant_model_type: str = "WGS"  # WGS, WES, PACBIO, HYBRID
    deepvariant_min_mapping_quality: int = 20
    deepvariant_min_base_quality: int = 10
    
    # Structural Variants
    manta_min_sv_size: int = 50
    manta_max_sv_size: int = 1000000
    delly_min_sv_size: int = 300
    delly_min_mapping_quality: int = 20
    
    # CNV Detection
    cnvkit_bin_size: int = 1000
    cnvkit_min_cnv_size: int = 5000
    gatk_gcnv_min_segment_length: int = 10000


@dataclass
class QualityControlConfig:
    """Quality control thresholds."""
    # Sample-Level QC
    min_total_reads: int = 30_000_000
    min_mean_coverage: float = 30.0
    max_contamination_rate: float = 0.02
    min_pct_bases_20x: float = 0.90
    max_duplication_rate: float = 0.30
    min_insert_size_median: int = 300
    max_insert_size_sd: int = 150
    
    # Alignment QC
    min_mapping_rate: float = 0.95
    min_properly_paired_rate: float = 0.90
    max_unmapped_rate: float = 0.05
    max_secondary_alignment_rate: float = 0.05
    
    # Variant-Level QC (SNV)
    min_variant_quality: float = 30.0
    min_genotype_quality: float = 20.0
    min_read_depth: int = 10
    min_allele_balance: float = 0.20
    max_allele_balance: float = 0.80
    min_strand_bias_pvalue: float = 0.001
    max_fisher_strand: float = 60.0
    min_quality_by_depth: float = 2.0
    
    # Variant-Level QC (Indel)
    min_indel_quality: float = 30.0
    min_indel_depth: int = 20
    max_indel_length: int = 50
    min_indel_allele_balance: float = 0.25
    
    # Structural Variant QC
    min_sv_quality: float = 20.0
    min_sv_support_reads: int = 4
    min_sv_support_read_pairs: int = 2
    max_sv_breakpoint_uncertainty: int = 500
    
    # CNV QC
    min_cnv_log2_ratio_deletion: float = -0.3
    max_cnv_log2_ratio_duplication: float = 0.3
    min_cnv_bins: int = 3


@dataclass
class FilteringConfig:
    """Variant filtering configuration."""
    # Rare Disease Thresholds
    rare_disease_max_af: float = 0.0001  # 0.01%
    ultra_rare_max_af: float = 0.00001  # 0.001%
    
    # ACMG/AMP Classification Weights
    acmg_pvs1_weight: float = 8.0
    acmg_ps1_weight: float = 4.0
    acmg_ps2_weight: float = 4.0
    acmg_ps3_weight: float = 4.0
    acmg_ps4_weight: float = 4.0
    acmg_pm1_weight: float = 2.0
    acmg_pm2_weight: float = 2.0
    acmg_pm3_weight: float = 2.0
    acmg_pm4_weight: float = 2.0
    acmg_pm5_weight: float = 2.0
    acmg_pm6_weight: float = 2.0
    acmg_pp1_weight: float = 1.0
    acmg_pp2_weight: float = 1.0
    acmg_pp3_weight: float = 1.0
    acmg_pp4_weight: float = 1.0
    acmg_pp5_weight: float = 1.0
    
    # Classification Thresholds
    pathogenic_score_threshold: float = 10.0
    likely_pathogenic_score_threshold: float = 6.0
    benign_score_threshold: float = -10.0
    likely_benign_score_threshold: float = -6.0
    
    # In Silico Prediction Thresholds
    revel_pathogenic_threshold: float = 0.5
    cadd_phred_threshold: float = 20.0
    spliceai_threshold: float = 0.5
    polyphen2_damaging_threshold: float = 0.85
    sift_deleterious_threshold: float = 0.05


@dataclass
class ClinicalReportingConfig:
    """Clinical reporting configuration."""
    # Reportable Variant Classes
    report_pathogenic: bool = True
    report_likely_pathogenic: bool = True
    report_vus_strong_evidence: bool = True
    report_vus_all: bool = False
    report_likely_benign: bool = False
    report_benign: bool = False
    
    # Gene Lists
    actionable_genes_list: Path = Path("/data/gene_lists/acmg73.txt")
    disease_specific_panels: Dict[str, Path] = field(default_factory=dict)
    pharmacogenes_list: Path = Path("/data/gene_lists/pharmgkb.txt")
    
    # Secondary Findings
    report_secondary_findings: bool = True
    acmg_sf_version: str = "v3.2"
    acmg_sf_gene_list: Path = Path("/data/gene_lists/acmg_sf_v3.2.txt")
    
    # Report Format
    report_format: str = "PDF"
    report_include_negative_results: bool = True
    report_include_qc_metrics: bool = True
    report_include_methodology: bool = True
    report_include_limitations: bool = True
    
    # Clinical Context
    min_phenotype_match_score: float = 0.6
    inheritance_models: List[str] = field(default_factory=lambda: [
        "autosomal_recessive",
        "autosomal_dominant",
        "x_linked",
        "de_novo"
    ])


@dataclass
class AuditComplianceConfig:
    """Audit trail and regulatory compliance configuration."""
    # Regulatory Framework
    regulatory_framework: RegulatoryFramework = RegulatoryFramework.ICMR_NABL
    lab_accreditation_id: str = ""
    lab_clia_number: Optional[str] = None
    lab_cap_number: Optional[str] = None
    
    # Data Retention
    raw_data_retention_days: int = 365 * 7  # 7 years
    processed_data_retention_days: int = 365 * 10  # 10 years
    audit_log_retention_days: int = 365 * 15  # 15 years
    temp_file_retention_hours: int = 24
    
    # Audit Trail
    audit_secret_key: Optional[str] = None  # From environment
    audit_signature_algorithm: str = "SHA256"
    audit_log_rotation_size_mb: int = 100
    enable_real_time_audit_verification: bool = True
    
    # Access Control
    require_two_factor_auth: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 3
    password_min_length: int = 12
    password_require_complexity: bool = True
    
    # Encryption
    encrypt_data_at_rest: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    encrypt_backups: bool = True


@dataclass
class ExecutionConfig:
    """Execution and performance configuration."""
    # Checkpoint & Resume
    enable_checkpointing: bool = True
    checkpoint_interval_minutes: int = 30
    auto_resume_on_failure: bool = True
    max_auto_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    retry_max_backoff_minutes: int = 60
    
    # Parallelization
    enable_gpu_acceleration: bool = True
    gpu_task_priority: List[str] = field(default_factory=lambda: ["deepvariant", "parabricks"])
    enable_multiprocessing: bool = True
    chunk_size_variants: int = 10000
    chunk_size_reads: int = 5_000_000
    
    # I/O Optimization
    use_memory_mapped_io: bool = True
    async_io_enabled: bool = True
    compression_level: int = 6
    compression_threads: int = 4
    buffer_size_mb: int = 64
    
    # Timeout Configuration
    alignment_timeout_hours: int = 12
    variant_calling_timeout_hours: int = 24
    annotation_timeout_hours: int = 6
    total_pipeline_timeout_hours: int = 48


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    # Log Levels
    log_level_console: str = "INFO"
    log_level_file: str = "DEBUG"
    log_level_audit: str = "INFO"
    
    # Log Destinations
    log_dir: Path = Path("/data/logs")
    log_rotation_size_mb: int = 100
    log_retention_count: int = 10
    log_format: str = "structured_json"
    
    # Monitoring & Alerts
    enable_prometheus_metrics: bool = True
    prometheus_port: int = 9090
    enable_email_alerts: bool = True
    alert_email_recipients: List[str] = field(default_factory=list)
    alert_on_failure: bool = True
    alert_on_qc_failure: bool = True
    alert_on_low_disk_space_gb: float = 100.0
    
    # Performance Metrics
    track_resource_usage: bool = True
    resource_sampling_interval_seconds: int = 60
    track_stage_execution_time: bool = True


@dataclass
class EnvironmentConfig:
    """Environment and deployment configuration."""
    # Environment Type
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    debug_mode: bool = False
    enable_profiling: bool = False
    synthetic_data_mode: bool = False
    
    # Container Configuration
    container_runtime: str = "docker"
    container_base_image: str = "ubuntu:22.04"
    cuda_version: str = "11.8"
    enable_container_isolation: bool = True
    
    # Network
    enable_internet_access: bool = False  # Air-gapped for PHI
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    database_mirror_url: Optional[str] = None
    
    # Locale
    locale: str = "en_IN"
    timezone: str = "Asia/Kolkata"
    date_format: str = "%Y-%m-%d"
    currency: str = "INR"


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration with validation."""
    
    # Metadata
    config_version: str = "1.0.0"
    config_name: str = "genomic_pipeline_config"
    config_last_modified: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_checksum: str = ""
    
    # Configuration Components
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    reference_genome: ReferenceGenomeConfig = field(default_factory=ReferenceGenomeConfig)
    population_databases: PopulationDatabaseConfig = field(default_factory=PopulationDatabaseConfig)
    clinical_databases: ClinicalDatabaseConfig = field(default_factory=ClinicalDatabaseConfig)
    validation_datasets: ValidationDatasetConfig = field(default_factory=ValidationDatasetConfig)
    tools: ToolVersionConfig = field(default_factory=ToolVersionConfig)
    parameters: PipelineParametersConfig = field(default_factory=PipelineParametersConfig)
    qc_thresholds: QualityControlConfig = field(default_factory=QualityControlConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    clinical_reporting: ClinicalReportingConfig = field(default_factory=ClinicalReportingConfig)
    audit_compliance: AuditComplianceConfig = field(default_factory=AuditComplianceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Dataset Manifests
    datasets: Dict[str, DatasetManifest] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and checksum computation."""
        # Load audit secret key from environment if not set
        if self.audit_compliance.audit_secret_key is None:
            self.audit_compliance.audit_secret_key = os.environ.get(
                'GENOMIC_AUDIT_SECRET_KEY',
                'default_key_change_in_production'
            )
        
        # Compute configuration checksum
        self.config_checksum = self.compute_checksum()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of all configuration parameters.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate hardware
        hw_valid, hw_errors = self.hardware.validate()
        if not hw_valid:
            errors.extend([f"Hardware: {e}" for e in hw_errors])
        
        # Validate cross-parameter constraints
        errors.extend(self._validate_cross_parameter_constraints())
        
        # Validate file paths exist (if not in synthetic mode)
        if not self.environment.synthetic_data_mode:
            errors.extend(self._validate_file_paths())
        
        # Validate regulatory compliance
        errors.extend(self._validate_regulatory_compliance())
        
        # Validate tool versions compatibility
        errors.extend(self._validate_tool_compatibility())
        
        return len(errors) == 0, errors
    
    def _validate_cross_parameter_constraints(self) -> List[str]:
        """Validate dependencies between configuration parameters."""
        errors = []
        
        # Parallelism constraint
        min_cpu_per_task = 4
        if self.hardware.max_parallel_tasks > self.hardware.max_cpu_cores / min_cpu_per_task:
            errors.append(
                f"max_parallel_tasks ({self.hardware.max_parallel_tasks}) too high for "
                f"available CPUs ({self.hardware.max_cpu_cores})"
            )
        
        # Memory constraint for variant processing
        estimated_memory_per_task = 16.0
        total_memory_needed = estimated_memory_per_task * self.hardware.max_parallel_tasks
        if total_memory_needed > self.hardware.max_memory_gb * 0.9:
            errors.append(
                f"Insufficient memory for {self.hardware.max_parallel_tasks} parallel tasks. "
                f"Need {total_memory_needed:.1f}GB, have {self.hardware.max_memory_gb:.1f}GB"
            )
        
        # Timeout hierarchy
        total_stage_timeout = (
            self.execution.alignment_timeout_hours +
            self.execution.variant_calling_timeout_hours +
            self.execution.annotation_timeout_hours
        )
        if self.execution.total_pipeline_timeout_hours < total_stage_timeout:
            errors.append(
                f"total_pipeline_timeout ({self.execution.total_pipeline_timeout_hours}h) must be >= "
                f"sum of stage timeouts ({total_stage_timeout}h)"
            )
        
        # Reference genome version consistency
        if self.tools.vep_version != self.reference_genome.ensembl_version:
            errors.append(
                f"VEP version ({self.tools.vep_version}) should match Ensembl version "
                f"({self.reference_genome.ensembl_version})"
            )
        
        return errors
    
    def _validate_file_paths(self) -> List[str]:
        """Validate that required files exist."""
        errors = []
        
        critical_files = [
            ("Reference genome FASTA", self.reference_genome.fasta_path),
            ("Reference genome index", self.reference_genome.fai_path),
            ("Reference genome dict", self.reference_genome.dict_path),
            ("Ensembl GTF", self.reference_genome.ensembl_gtf_path),
            ("IndiGenomes VCF", self.population_databases.indigenomes_vcf_path),
            ("ClinVar VCF", self.clinical_databases.clinvar_vcf_path),
        ]
        
        for name, path in critical_files:
            if not path.exists():
                errors.append(f"Missing required file: {name} at {path}")
        
        # Validate tool executables
        tool_paths = [
            ("GATK", self.tools.gatk_path),
            ("DeepVariant", self.tools.deepvariant_path),
            ("BWA-MEM2", self.tools.bwa_mem2_path),
            ("SAMtools", self.tools.samtools_path),
            ("BCFtools", self.tools.bcftools_path),
            ("VEP", self.tools.vep_path),
        ]
        
        for name, path in tool_paths:
            if not path.exists():
                errors.append(f"Missing required tool: {name} at {path}")
        
        return errors
    
    def _validate_regulatory_compliance(self) -> List[str]:
        """Validate regulatory compliance requirements."""
        errors = []
        
        if self.audit_compliance.regulatory_framework == RegulatoryFramework.ICMR_NABL:
            # ICMR/NABL requires 10+ year retention
            if self.audit_compliance.audit_log_retention_days < 365 * 10:
                errors.append(
                    f"ICMR/NABL requires audit log retention >= 10 years, "
                    f"configured: {self.audit_compliance.audit_log_retention_days} days"
                )
            
            # Must have accreditation ID
            if not self.audit_compliance.lab_accreditation_id:
                errors.append("ICMR/NABL requires lab_accreditation_id")
        
        # Secondary findings require gene list
        if self.clinical_reporting.report_secondary_findings:
            if not self.clinical_reporting.acmg_sf_gene_list.exists():
                errors.append(
                    "report_secondary_findings enabled but acmg_sf_gene_list not found"
                )
        
        # Encryption requirements for production
        if self.environment.environment == EnvironmentType.PRODUCTION:
            if not self.audit_compliance.encrypt_data_at_rest:
                errors.append("Production environment requires encrypt_data_at_rest=True")
        
        return errors
    
    def _validate_tool_compatibility(self) -> List[str]:
        """Validate tool version compatibility."""
        errors = []
        
        # GATK version compatibility
        gatk_version = self.tools.gatk_version
        if not gatk_version.startswith("4."):
            errors.append(f"GATK version {gatk_version} not supported, requires 4.x")
        
        # DeepVariant model compatibility
        model_type = self.parameters.deepvariant_model_type
        valid_models = ["WGS", "WES", "PACBIO", "HYBRID"]
        if model_type not in valid_models:
            errors.append(
                f"Invalid DeepVariant model type: {model_type}, "
                f"must be one of {valid_models}"
            )
        
        return errors
    
    def compute_checksum(self) -> str:
        """
        Compute SHA256 checksum of configuration for versioning.
        
        Returns:
            Hex string of configuration checksum
        """
        # Serialize config to canonical JSON
        config_dict = self.to_dict()
        # Remove checksum field before computing
        config_dict.pop('config_checksum', None)
        
        canonical_json = json.dumps(config_dict, sort_keys=True, indent=2)
        checksum = hashlib.sha256(canonical_json.encode()).hexdigest()
        
        return checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        def convert_value(v):
            """Recursively convert values to serializable types."""
            if isinstance(v, Path):
                return str(v)
            elif isinstance(v, Enum):
                return v.value
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(val) for val in v]
            elif hasattr(v, 'to_dict'):
                return v.to_dict()
            elif hasattr(v, '__dict__'):
                return convert_value(asdict(v))
            else:
                return v
        
        return convert_value(asdict(self))
    
    def to_json(self, filepath: Path):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'PipelineConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            PipelineConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
        
        Returns:
            PipelineConfig instance
        """
        # Helper to convert paths
        def convert_paths(obj, cls_type):
            if hasattr(cls_type, '__annotations__'):
                for field_name, field_type in cls_type.__annotations__.items():
                    if field_name in obj:
                        if field_type == Path or (hasattr(field_type, '__origin__') and 
                                                  field_type.__origin__ == Union and 
                                                  Path in field_type.__args__):
                            if obj[field_name] is not None:
                                obj[field_name] = Path(obj[field_name])
                        elif isinstance(obj[field_name], dict):
                            # Recursively convert nested dicts
                            for key, value in obj[field_name].items():
                                if isinstance(value, str) and '/' in value:
                                    obj[field_name][key] = Path(value)
            return obj
        
        # Convert enum strings back to enums
        if 'environment' in data and 'environment' in data['environment']:
            data['environment']['environment'] = EnvironmentType(data['environment']['environment'])
        
        if 'audit_compliance' in data and 'regulatory_framework' in data['audit_compliance']:
            data['audit_compliance']['regulatory_framework'] = RegulatoryFramework(
                data['audit_compliance']['regulatory_framework']
            )
        
        # Create nested configurations
        config = cls(
            config_version=data.get('config_version', '1.0.0'),
            config_name=data.get('config_name', 'genomic_pipeline_config'),
            config_last_modified=data.get('config_last_modified', ''),
            hardware=HardwareConfig(**convert_paths(data.get('hardware', {}), HardwareConfig)),
            reference_genome=ReferenceGenomeConfig(**convert_paths(data.get('reference_genome', {}), ReferenceGenomeConfig)),
            population_databases=PopulationDatabaseConfig(**convert_paths(data.get('population_databases', {}), PopulationDatabaseConfig)),
            clinical_databases=ClinicalDatabaseConfig(**convert_paths(data.get('clinical_databases', {}), ClinicalDatabaseConfig)),
            validation_datasets=ValidationDatasetConfig(**convert_paths(data.get('validation_datasets', {}), ValidationDatasetConfig)),
            tools=ToolVersionConfig(**convert_paths(data.get('tools', {}), ToolVersionConfig)),
            parameters=PipelineParametersConfig(**data.get('parameters', {})),
            qc_thresholds=QualityControlConfig(**data.get('qc_thresholds', {})),
            filtering=FilteringConfig(**data.get('filtering', {})),
            clinical_reporting=ClinicalReportingConfig(**convert_paths(data.get('clinical_reporting', {}), ClinicalReportingConfig)),
            audit_compliance=AuditComplianceConfig(**data.get('audit_compliance', {})),
            execution=ExecutionConfig(**data.get('execution', {})),
            logging_config=LoggingConfig(**convert_paths(data.get('logging_config', {}), LoggingConfig)),
            environment=EnvironmentConfig(**data.get('environment', {})),
        )
        
        # Load dataset manifests
        if 'datasets' in data:
            for name, manifest_data in data['datasets'].items():
                manifest_data['file_path'] = Path(manifest_data['file_path'])
                config.datasets[name] = DatasetManifest(**manifest_data)
        
        return config
    
    def merge_override(self, override_config: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create new configuration by merging overrides.
        
        Args:
            override_config: Dictionary with override parameters
        
        Returns:
            New PipelineConfig with overrides applied
        """
        base_dict = self.to_dict()
        
        # Deep merge override into base
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(base_dict, override_config)
        
        return self.from_dict(base_dict)


# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """
    Manages hierarchical configuration with base, site, and run overrides.
    
    Configuration precedence (highest to lowest):
    1. Run-specific config
    2. Site config
    3. Base config
    """
    
    def __init__(self, config_dir: Path):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_config_path = self.config_dir / "base_config.json"
        self.site_config_path = self.config_dir / "site_config.json"
        
        self.logger = logging.getLogger('genomic_pipeline.config')
    
    def load_base_config(self) -> PipelineConfig:
        """
        Load base configuration.
        
        Returns:
            Base PipelineConfig
        """
        if self.base_config_path.exists():
            config = PipelineConfig.from_json(self.base_config_path)
            self.logger.info(f"Loaded base config: {config.config_version}")
        else:
            config = PipelineConfig()
            self.logger.warning("Base config not found, using defaults")
        
        return config
    
    def load_site_config(self) -> Optional[Dict[str, Any]]:
        """
        Load site-specific configuration overrides.
        
        Returns:
            Dictionary with site overrides or None
        """
        if self.site_config_path.exists():
            with open(self.site_config_path, 'r') as f:
                site_config = json.load(f)
            self.logger.info("Loaded site config overrides")
            return site_config
        
        return None
    
    def load_run_config(self, run_config_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Load run-specific configuration overrides.
        
        Args:
            run_config_path: Path to run config file
        
        Returns:
            Dictionary with run overrides or None
        """
        if run_config_path and run_config_path.exists():
            with open(run_config_path, 'r') as f:
                run_config = json.load(f)
            self.logger.info(f"Loaded run config: {run_config_path}")
            return run_config
        
        return None
    
    def build_config(self, run_config_path: Optional[Path] = None) -> PipelineConfig:
        """
        Build final configuration with all overrides applied.
        
        Args:
            run_config_path: Optional path to run-specific config
        
        Returns:
            Final merged PipelineConfig
        """
        # Start with base
        config = self.load_base_config()
        
        # Apply site overrides
        site_overrides = self.load_site_config()
        if site_overrides:
            config = config.merge_override(site_overrides)
        
        # Apply run overrides
        run_overrides = self.load_run_config(run_config_path)
        if run_overrides:
            config = config.merge_override(run_overrides)
        
        # Validate final configuration
        is_valid, errors = config.validate()
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Configuration built successfully. Checksum: {config.config_checksum}")
        
        return config
    
    def save_base_config(self, config: PipelineConfig):
        """
        Save base configuration.
        
        Args:
            config: Configuration to save
        """
        config.to_json(self.base_config_path)
        self.logger.info(f"Saved base config: {self.base_config_path}")
    
    def save_site_config(self, overrides: Dict[str, Any]):
        """
        Save site-specific overrides.
        
        Args:
            overrides: Dictionary with site overrides
        """
        with open(self.site_config_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        self.logger.info(f"Saved site config: {self.site_config_path}")
    
    def create_run_config(self, output_path: Path, overrides: Dict[str, Any]):
        """
        Create run-specific configuration file.
        
        Args:
            output_path: Path to save run config
            overrides: Dictionary with run-specific overrides
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        
        self.logger.info(f"Created run config: {output_path}")
    
    def verify_dataset_integrity(self, config: PipelineConfig) -> Tuple[bool, List[str]]:
        """
        Verify integrity of all datasets in configuration.
        
        Args:
            config: Configuration to verify
        
        Returns:
            Tuple of (all_valid, list_of_failed_datasets)
        """
        failed = []
        
        for name, manifest in config.datasets.items():
            if not manifest.validate_integrity():
                failed.append(f"{name}: Checksum mismatch or file missing")
                self.logger.error(f"Dataset integrity check failed: {name}")
        
        if not failed:
            self.logger.info("All datasets passed integrity verification")
        
        return len(failed) == 0, failed


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_file_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute cryptographic checksum of file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm
    
    Returns:
        Hexadecimal checksum string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def create_default_config(output_path: Path):
    """
    Create default configuration file with sensible defaults.
    
    Args:
        output_path: Path to save default config
    """
    config = PipelineConfig()
    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")


def validate_config_file(config_path: Path) -> bool:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        True if valid
    """
    try:
        config = PipelineConfig.from_json(config_path)
        is_valid, errors = config.validate()
        
        if is_valid:
            print(f"✓ Configuration valid: {config_path}")
            print(f"  Version: {config.config_version}")
            print(f"  Checksum: {config.config_checksum}")
            return True
        else:
            print(f"✗ Configuration invalid: {config_path}")
            for error in errors:
                print(f"  - {error}")
            return False
    
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def main():
    """Example usage of configuration system."""
    
    # Initialize configuration manager
    config_manager = ConfigurationManager(Path("/data/genomic_pipeline/config"))
    
    # Create default base configuration
    base_config = PipelineConfig()
    
    # Customize for Indian clinical use
    base_config.population_databases.default_population = "Indian"
    base_config.audit_compliance.regulatory_framework = RegulatoryFramework.ICMR_NABL
    base_config.audit_compliance.lab_accreditation_id = "NABL-2024-12345"
    base_config.environment.locale = "en_IN"
    base_config.environment.timezone = "Asia/Kolkata"
    
    # Save base config
    config_manager.save_base_config(base_config)
    
    # Create site-specific overrides
    site_overrides = {
        "hardware": {
            "max_cpu_cores": 32,
            "max_memory_gb": 128.0,
            "available_gpus": [0, 1, 2, 3]
        },
        "logging_config": {
            "alert_email_recipients": ["lab@example.com", "admin@example.com"]
        }
    }
    config_manager.save_site_config(site_overrides)
    
    # Build final configuration
    final_config = config_manager.build_config()
    
    # Validate
    is_valid, errors = final_config.validate()
    if is_valid:
        print("Configuration validated successfully")
        print(f"Checksum: {final_config.config_checksum}")
    else:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    return final_config


if __name__ == '__main__':
    main()
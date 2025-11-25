"""
Statistical Analysis Framework for Performance Validation
Provides rigorous statistical methods for validating performance claims
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
from dataclasses import dataclass

@dataclass
class StatisticalTest:
    """Results of statistical hypothesis testing"""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    degrees_of_freedom: int
    effect_size: float
    power: float
    confidence_interval: Tuple[float, float]
    interpretation: str

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for performance validation"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
        self.results = []
    
    def validate_speedup_claims(self, baseline_times: List[float], 
                              optimized_times: List[float],
                              claimed_speedup: float) -> StatisticalTest:
        """Validate speedup claims with rigorous statistical testing"""
        
        # Calculate actual speedup
        baseline_mean = np.mean(baseline_times)
        optimized_mean = np.mean(optimized_times)
        actual_speedup = baseline_mean / optimized_mean
        
        # One-sample t-test against claimed speedup
        speedup_ratios = [b/o for b, o in zip(baseline_times, optimized_times)]
        t_stat, p_value = stats.ttest_1samp(speedup_ratios, claimed_speedup)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(speedup_ratios) - claimed_speedup) / np.std(speedup_ratios)
        
        # Confidence interval for speedup
        ci_lower, ci_upper = stats.t.interval(
            1 - self.alpha, 
            len(speedup_ratios) - 1,
            loc=np.mean(speedup_ratios),
            scale=stats.sem(speedup_ratios)
        )
        
        # Power analysis
        power = self._calculate_power(speedup_ratios, claimed_speedup, self.alpha)
        
        # Interpretation
        if p_value < self.alpha:
            if actual_speedup >= claimed_speedup * 0.9:  # Within 10%
                interpretation = f"VALIDATED: Actual speedup ({actual_speedup:.2f}x) meets claimed performance"
            else:
                interpretation = f"OVERSTATED: Actual speedup ({actual_speedup:.2f}x) significantly less than claimed"
        else:
            interpretation = f"INCONCLUSIVE: Insufficient evidence to validate {claimed_speedup:.2f}x speedup claim"
        
        return StatisticalTest(
            test_name="Speedup Validation",
            statistic=t_stat,
            p_value=p_value,
            critical_value=stats.t.ppf(1 - self.alpha/2, len(speedup_ratios) - 1),
            degrees_of_freedom=len(speedup_ratios) - 1,
            effect_size=effect_size,
            power=power,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def validate_energy_efficiency(self, baseline_energy: List[float],
                                 optimized_energy: List[float],
                                 claimed_reduction: float) -> StatisticalTest:
        """Validate energy efficiency claims"""
        
        # Calculate actual energy reduction
        baseline_mean = np.mean(baseline_energy)
        optimized_mean = np.mean(optimized_energy)
        actual_reduction = (baseline_mean - optimized_mean) / baseline_mean * 100
        
        # Paired t-test for energy reduction
        energy_reductions = [(b - o) / b * 100 for b, o in zip(baseline_energy, optimized_energy)]
        t_stat, p_value = stats.ttest_1samp(energy_reductions, claimed_reduction)
        
        # Effect size
        effect_size = (np.mean(energy_reductions) - claimed_reduction) / np.std(energy_reductions)
        
        # Confidence interval
        ci_lower, ci_upper = stats.t.interval(
            1 - self.alpha,
            len(energy_reductions) - 1,
            loc=np.mean(energy_reductions),
            scale=stats.sem(energy_reductions)
        )
        
        # Power analysis
        power = self._calculate_power(energy_reductions, claimed_reduction, self.alpha)
        
        # Interpretation
        if p_value < self.alpha:
            if actual_reduction >= claimed_reduction * 0.8:  # Within 20%
                interpretation = f"VALIDATED: Actual energy reduction ({actual_reduction:.1f}%) meets claimed efficiency"
            else:
                interpretation = f"OVERSTATED: Actual reduction ({actual_reduction:.1f}%) less than claimed"
        else:
            interpretation = f"INCONCLUSIVE: Insufficient evidence for {claimed_reduction:.1f}% energy reduction"
        
        return StatisticalTest(
            test_name="Energy Efficiency Validation",
            statistic=t_stat,
            p_value=p_value,
            critical_value=stats.t.ppf(1 - self.alpha/2, len(energy_reductions) - 1),
            degrees_of_freedom=len(energy_reductions) - 1,
            effect_size=effect_size,
            power=power,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def validate_accuracy_retention(self, baseline_accuracy: List[float],
                                  optimized_accuracy: List[float],
                                  minimum_retention: float = 99.0) -> StatisticalTest:
        """Validate that accuracy is retained after optimization"""
        
        # Calculate retention percentages
        retention_percentages = [o / b * 100 for b, o in zip(baseline_accuracy, optimized_accuracy)]
        
        # One-sample t-test against minimum retention
        t_stat, p_value = stats.ttest_1samp(retention_percentages, minimum_retention)
        
        # Effect size
        effect_size = (np.mean(retention_percentages) - minimum_retention) / np.std(retention_percentages)
        
        # Confidence interval
        ci_lower, ci_upper = stats.t.interval(
            1 - self.alpha,
            len(retention_percentages) - 1,
            loc=np.mean(retention_percentages),
            scale=stats.sem(retention_percentages)
        )
        
        # Power analysis
        power = self._calculate_power(retention_percentages, minimum_retention, self.alpha)
        
        # Interpretation
        actual_retention = np.mean(retention_percentages)
        if actual_retention >= minimum_retention and ci_lower >= minimum_retention * 0.98:
            interpretation = f"VALIDATED: Accuracy retention ({actual_retention:.2f}%) exceeds minimum requirement"
        elif actual_retention >= minimum_retention:
            interpretation = f"MARGINAL: Average retention acceptable but confidence interval includes risk"
        else:
            interpretation = f"FAILED: Accuracy retention ({actual_retention:.2f}%) below minimum requirement"
        
        return StatisticalTest(
            test_name="Accuracy Retention Validation",
            statistic=t_stat,
            p_value=p_value,
            critical_value=stats.t.ppf(1 - self.alpha, len(retention_percentages) - 1),
            degrees_of_freedom=len(retention_percentages) - 1,
            effect_size=effect_size,
            power=power,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def perform_equivalence_testing(self, group1: List[float], group2: List[float],
                                  equivalence_margin: float) -> StatisticalTest:
        """Perform equivalence testing (TOST - Two One-Sided Tests)"""
        
        # Calculate difference
        diff = np.mean(group1) - np.mean(group2)
        se_diff = np.sqrt(np.var(group1)/len(group1) + np.var(group2)/len(group2))
        
        # Two one-sided tests
        t1 = (diff - equivalence_margin) / se_diff
        t2 = (diff + equivalence_margin) / se_diff
        
        df = len(group1) + len(group2) - 2
        p1 = stats.t.cdf(t1, df)
        p2 = 1 - stats.t.cdf(t2, df)
        
        # TOST p-value is the maximum of the two one-sided p-values
        tost_p = max(p1, p2)
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1) + (len(group2)-1)*np.var(group2)) / df)
        effect_size = diff / pooled_std
        
        # Confidence interval for difference
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        # Interpretation
        if tost_p < self.alpha:
            interpretation = "EQUIVALENT: Groups are statistically equivalent within specified margin"
        else:
            interpretation = "NOT EQUIVALENT: Cannot conclude statistical equivalence"
        
        return StatisticalTest(
            test_name="Equivalence Testing (TOST)",
            statistic=min(abs(t1), abs(t2)),
            p_value=tost_p,
            critical_value=stats.t.ppf(1 - self.alpha, df),
            degrees_of_freedom=df,
            effect_size=effect_size,
            power=self._calculate_equivalence_power(group1, group2, equivalence_margin),
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def perform_multiple_comparison_correction(self, p_values: List[float], 
                                             method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparison correction"""
        
        if method == 'bonferroni':
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif method == 'holm':
            return self._holm_correction(p_values)
        elif method == 'fdr':
            return self._benjamini_hochberg_correction(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def calculate_required_sample_size(self, effect_size: float, power: float = 0.8,
                                     alpha: float = 0.05, test_type: str = 'two_sample') -> int:
        """Calculate required sample size for desired power"""
        
        if test_type == 'two_sample':
            # Two-sample t-test
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))
        
        elif test_type == 'one_sample':
            # One-sample t-test
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n = ((z_alpha + z_beta) / effect_size) ** 2
            return int(np.ceil(n))
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def generate_power_analysis_report(self, effect_sizes: List[float],
                                     sample_sizes: List[int]) -> str:
        """Generate power analysis report"""
        
        report = """
# Power Analysis Report

## Overview
Power analysis determines the probability of detecting a true effect given the sample size and effect size.

## Power Analysis Results

| Effect Size | Sample Size | Power | Interpretation |
|-------------|-------------|-------|----------------|
"""
        
        for effect_size in effect_sizes:
            for sample_size in sample_sizes:
                power = self._calculate_power_for_size(effect_size, sample_size)
                interpretation = self._interpret_power(power)
                report += f"| {effect_size:.2f} | {sample_size} | {power:.3f} | {interpretation} |\n"
        
        report += """

## Power Interpretation Guidelines
- **Power ≥ 0.80**: Adequate power to detect effects
- **Power 0.60-0.79**: Moderate power, consider larger sample
- **Power < 0.60**: Insufficient power, increase sample size

## Recommendations
1. Use sample sizes that provide ≥80% power for expected effect sizes
2. Consider effect size when interpreting non-significant results
3. Report power analysis in academic publications
"""
        
        return report
    
    def _calculate_power(self, data: List[float], null_value: float, alpha: float) -> float:
        """Calculate statistical power for one-sample t-test"""
        
        n = len(data)
        mean_diff = np.mean(data) - null_value
        std_dev = np.std(data, ddof=1)
        
        # Effect size
        effect_size = mean_diff / std_dev
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Critical value
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Power calculation using non-central t-distribution
        power = 1 - stats.nct.cdf(t_critical, n - 1, ncp) + stats.nct.cdf(-t_critical, n - 1, ncp)
        
        return power
    
    def _calculate_equivalence_power(self, group1: List[float], group2: List[float],
                                   equivalence_margin: float) -> float:
        """Calculate power for equivalence testing"""
        
        n1, n2 = len(group1), len(group2)
        pooled_var = ((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1 + n2 - 2)
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Approximate power calculation for TOST
        t_critical = stats.t.ppf(1 - self.alpha, n1 + n2 - 2)
        power = stats.norm.cdf((equivalence_margin - t_critical * se) / se)
        
        return power
    
    def _calculate_power_for_size(self, effect_size: float, sample_size: int) -> float:
        """Calculate power for given effect size and sample size"""
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Critical value for two-tailed test
        t_critical = stats.t.ppf(1 - self.alpha/2, sample_size - 1)
        
        # Power calculation
        power = 1 - stats.nct.cdf(t_critical, sample_size - 1, ncp) + \
                stats.nct.cdf(-t_critical, sample_size - 1, ncp)
        
        return power
    
    def _interpret_power(self, power: float) -> str:
        """Interpret power value"""
        if power >= 0.8:
            return "Adequate"
        elif power >= 0.6:
            return "Moderate"
        else:
            return "Insufficient"
    
    def _holm_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni correction"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected_p = [0] * n
        
        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_values[idx] * (n - i), 1.0)
        
        return corrected_p
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected_p = [0] * n
        
        for i, idx in enumerate(sorted_indices):
            corrected_p[idx] = min(p_values[idx] * n / (i + 1), 1.0)
        
        return corrected_p

# Export for use in validation framework
statistical_analyzer = StatisticalAnalyzer()
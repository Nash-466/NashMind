"""
Solution Evaluator for ARC Prize 2025
Evaluates and analyzes solutions for ARC tasks
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from collections import defaultdict

from ..arc.grid_operations import Grid
from ..arc.pattern_detector import PatternDetector


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating a solution"""
    task_id: str
    is_correct: bool
    pixel_accuracy: float
    structural_similarity: float
    color_accuracy: float
    shape_match: bool
    execution_time: float
    solver_used: str
    confidence: float
    error_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvaluation:
    """Evaluation results for multiple tasks"""
    total_tasks: int
    correct_tasks: int
    accuracy: float
    average_pixel_accuracy: float
    average_time: float
    solver_performance: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[str, Dict[str, float]]
    error_patterns: Dict[str, int]


class SolutionEvaluator:
    """Evaluates ARC solutions against ground truth"""
    
    def __init__(self):
        self.evaluation_history = []
        self.error_patterns = defaultdict(int)
    
    def evaluate_solution(self,
                         predicted: np.ndarray,
                         ground_truth: np.ndarray,
                         task_id: str = "",
                         solver_used: str = "",
                         execution_time: float = 0.0,
                         confidence: float = 0.0) -> EvaluationMetrics:
        """
        Evaluate a single solution against ground truth
        
        Args:
            predicted: Predicted output grid
            ground_truth: True output grid
            task_id: Task identifier
            solver_used: Name of solver used
            execution_time: Time taken to solve
            confidence: Solver confidence
            
        Returns:
            EvaluationMetrics with detailed analysis
        """
        
        # Check if completely correct
        is_correct = np.array_equal(predicted, ground_truth)
        
        # Calculate pixel accuracy
        pixel_accuracy = self._calculate_pixel_accuracy(predicted, ground_truth)
        
        # Calculate structural similarity
        structural_similarity = self._calculate_structural_similarity(
            predicted, ground_truth
        )
        
        # Calculate color accuracy
        color_accuracy = self._calculate_color_accuracy(predicted, ground_truth)
        
        # Check shape match
        shape_match = predicted.shape == ground_truth.shape
        
        # Perform error analysis
        error_analysis = self._analyze_errors(predicted, ground_truth)
        
        # Create metrics
        metrics = EvaluationMetrics(
            task_id=task_id,
            is_correct=is_correct,
            pixel_accuracy=pixel_accuracy,
            structural_similarity=structural_similarity,
            color_accuracy=color_accuracy,
            shape_match=shape_match,
            execution_time=execution_time,
            solver_used=solver_used,
            confidence=confidence,
            error_analysis=error_analysis
        )
        
        # Track evaluation
        self.evaluation_history.append(metrics)
        
        # Track error patterns
        if not is_correct:
            for error_type in error_analysis.get('error_types', []):
                self.error_patterns[error_type] += 1
        
        return metrics
    
    def _calculate_pixel_accuracy(self,
                                 predicted: np.ndarray,
                                 ground_truth: np.ndarray) -> float:
        """Calculate pixel-level accuracy"""
        
        if predicted.shape != ground_truth.shape:
            # Different shapes - calculate based on overlap
            min_h = min(predicted.shape[0], ground_truth.shape[0])
            min_w = min(predicted.shape[1], ground_truth.shape[1])
            
            if min_h == 0 or min_w == 0:
                return 0.0
            
            # Compare overlapping region
            pred_region = predicted[:min_h, :min_w]
            truth_region = ground_truth[:min_h, :min_w]
            
            correct_pixels = np.sum(pred_region == truth_region)
            total_pixels = ground_truth.size
            
            # Penalize for size mismatch
            size_penalty = abs(predicted.size - ground_truth.size) / ground_truth.size
            accuracy = (correct_pixels / total_pixels) * (1 - min(size_penalty, 0.5))
            
        else:
            # Same shape - direct comparison
            correct_pixels = np.sum(predicted == ground_truth)
            total_pixels = ground_truth.size
            accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return accuracy
    
    def _calculate_structural_similarity(self,
                                        predicted: np.ndarray,
                                        ground_truth: np.ndarray) -> float:
        """Calculate structural similarity between grids"""
        
        score = 0.0
        num_checks = 0
        
        # Convert to Grid objects
        pred_grid = Grid(predicted)
        truth_grid = Grid(ground_truth)
        
        # Check pattern similarity
        pred_detector = PatternDetector(pred_grid)
        truth_detector = PatternDetector(truth_grid)
        
        # Compare symmetries
        pred_symmetries = pred_detector.get_symmetries()
        truth_symmetries = truth_detector.get_symmetries()
        
        for sym_type in pred_symmetries:
            if pred_symmetries[sym_type] == truth_symmetries[sym_type]:
                score += 1
            num_checks += 1
        
        # Compare connected components count
        pred_components = pred_grid.get_connected_components()
        truth_components = truth_grid.get_connected_components()
        
        if len(pred_components) == len(truth_components):
            score += 1
        num_checks += 1
        
        # Compare color distribution similarity
        pred_colors = pred_grid.count_colors()
        truth_colors = truth_grid.count_colors()
        
        color_similarity = self._compare_color_distributions(pred_colors, truth_colors)
        score += color_similarity
        num_checks += 1
        
        # Calculate final score
        similarity = score / num_checks if num_checks > 0 else 0.0
        
        return similarity
    
    def _calculate_color_accuracy(self,
                                 predicted: np.ndarray,
                                 ground_truth: np.ndarray) -> float:
        """Calculate color matching accuracy"""
        
        pred_colors = set(np.unique(predicted))
        truth_colors = set(np.unique(ground_truth))
        
        # Check if same colors are present
        if pred_colors == truth_colors:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(pred_colors & truth_colors)
        union = len(pred_colors | truth_colors)
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_color_distributions(self,
                                    dist1: Dict[int, int],
                                    dist2: Dict[int, int]) -> float:
        """Compare two color distributions"""
        
        all_colors = set(dist1.keys()) | set(dist2.keys())
        
        if not all_colors:
            return 1.0
        
        total_diff = 0
        total_count = sum(dist1.values()) + sum(dist2.values())
        
        for color in all_colors:
            count1 = dist1.get(color, 0)
            count2 = dist2.get(color, 0)
            total_diff += abs(count1 - count2)
        
        similarity = 1 - (total_diff / total_count) if total_count > 0 else 0.0
        
        return max(0, similarity)
    
    def _analyze_errors(self,
                       predicted: np.ndarray,
                       ground_truth: np.ndarray) -> Dict[str, Any]:
        """Analyze errors in the prediction"""
        
        analysis = {
            'error_types': [],
            'shape_error': None,
            'color_errors': {},
            'pattern_errors': []
        }
        
        # Shape analysis
        if predicted.shape != ground_truth.shape:
            analysis['error_types'].append('shape_mismatch')
            analysis['shape_error'] = {
                'predicted': predicted.shape,
                'expected': ground_truth.shape,
                'difference': (predicted.shape[0] - ground_truth.shape[0],
                             predicted.shape[1] - ground_truth.shape[1])
            }
        
        # Color analysis
        pred_colors = set(np.unique(predicted))
        truth_colors = set(np.unique(ground_truth))
        
        missing_colors = truth_colors - pred_colors
        extra_colors = pred_colors - truth_colors
        
        if missing_colors:
            analysis['error_types'].append('missing_colors')
            analysis['color_errors']['missing'] = list(missing_colors)
        
        if extra_colors:
            analysis['error_types'].append('extra_colors')
            analysis['color_errors']['extra'] = list(extra_colors)
        
        # Pattern analysis (if same shape)
        if predicted.shape == ground_truth.shape:
            # Find regions with errors
            error_mask = predicted != ground_truth
            if np.any(error_mask):
                analysis['error_types'].append('pixel_errors')
                
                # Identify error patterns
                error_positions = np.argwhere(error_mask)
                
                # Check if errors are clustered
                if len(error_positions) > 0:
                    # Simple clustering check
                    y_coords = error_positions[:, 0]
                    x_coords = error_positions[:, 1]
                    
                    y_range = np.max(y_coords) - np.min(y_coords)
                    x_range = np.max(x_coords) - np.min(x_coords)
                    
                    if y_range < predicted.shape[0] / 3 and x_range < predicted.shape[1] / 3:
                        analysis['pattern_errors'].append('localized_errors')
                    else:
                        analysis['pattern_errors'].append('distributed_errors')
        
        return analysis
    
    def evaluate_batch(self,
                      predictions: List[np.ndarray],
                      ground_truths: List[np.ndarray],
                      task_ids: Optional[List[str]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None) -> BatchEvaluation:
        """
        Evaluate multiple solutions
        
        Args:
            predictions: List of predicted grids
            ground_truths: List of ground truth grids
            task_ids: Optional list of task IDs
            metadata: Optional metadata for each solution
            
        Returns:
            BatchEvaluation with aggregate metrics
        """
        
        if not task_ids:
            task_ids = [f"task_{i}" for i in range(len(predictions))]
        
        if not metadata:
            metadata = [{} for _ in range(len(predictions))]
        
        # Evaluate each solution
        metrics_list = []
        for pred, truth, task_id, meta in zip(predictions, ground_truths, task_ids, metadata):
            metrics = self.evaluate_solution(
                pred, truth, task_id,
                solver_used=meta.get('solver_used', ''),
                execution_time=meta.get('execution_time', 0.0),
                confidence=meta.get('confidence', 0.0)
            )
            metrics_list.append(metrics)
        
        # Calculate aggregate metrics
        correct_tasks = sum(1 for m in metrics_list if m.is_correct)
        accuracy = correct_tasks / len(metrics_list) if metrics_list else 0.0
        
        avg_pixel_accuracy = np.mean([m.pixel_accuracy for m in metrics_list])
        avg_time = np.mean([m.execution_time for m in metrics_list])
        
        # Solver performance breakdown
        solver_performance = defaultdict(lambda: {'count': 0, 'correct': 0, 'accuracy': 0.0})
        for m in metrics_list:
            solver = m.solver_used
            solver_performance[solver]['count'] += 1
            if m.is_correct:
                solver_performance[solver]['correct'] += 1
        
        for solver in solver_performance:
            stats = solver_performance[solver]
            stats['accuracy'] = stats['correct'] / stats['count'] if stats['count'] > 0 else 0.0
        
        # Create batch evaluation
        batch_eval = BatchEvaluation(
            total_tasks=len(metrics_list),
            correct_tasks=correct_tasks,
            accuracy=accuracy,
            average_pixel_accuracy=avg_pixel_accuracy,
            average_time=avg_time,
            solver_performance=dict(solver_performance),
            difficulty_breakdown={},  # Could be extended with difficulty analysis
            error_patterns=dict(self.error_patterns)
        )
        
        return batch_eval
    
    def generate_report(self, 
                       evaluation: EvaluationMetrics,
                       verbose: bool = True) -> str:
        """
        Generate a text report for an evaluation
        
        Args:
            evaluation: Evaluation metrics
            verbose: Include detailed error analysis
            
        Returns:
            Formatted report string
        """
        
        report = []
        report.append(f"{'='*60}")
        report.append(f"Task: {evaluation.task_id}")
        report.append(f"{'='*60}")
        report.append(f"Result: {'✓ CORRECT' if evaluation.is_correct else '✗ INCORRECT'}")
        report.append(f"Solver: {evaluation.solver_used}")
        report.append(f"Confidence: {evaluation.confidence:.2%}")
        report.append(f"Execution Time: {evaluation.execution_time:.2f}s")
        report.append("")
        report.append("Metrics:")
        report.append(f"  Pixel Accuracy: {evaluation.pixel_accuracy:.2%}")
        report.append(f"  Structural Similarity: {evaluation.structural_similarity:.2%}")
        report.append(f"  Color Accuracy: {evaluation.color_accuracy:.2%}")
        report.append(f"  Shape Match: {'Yes' if evaluation.shape_match else 'No'}")
        
        if verbose and evaluation.error_analysis:
            report.append("")
            report.append("Error Analysis:")
            
            if evaluation.error_analysis.get('error_types'):
                report.append(f"  Error Types: {', '.join(evaluation.error_analysis['error_types'])}")
            
            if evaluation.error_analysis.get('shape_error'):
                shape_err = evaluation.error_analysis['shape_error']
                report.append(f"  Shape: {shape_err['predicted']} vs {shape_err['expected']}")
            
            if evaluation.error_analysis.get('color_errors'):
                color_err = evaluation.error_analysis['color_errors']
                if color_err.get('missing'):
                    report.append(f"  Missing Colors: {color_err['missing']}")
                if color_err.get('extra'):
                    report.append(f"  Extra Colors: {color_err['extra']}")
        
        report.append(f"{'='*60}")
        
        return '\n'.join(report)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all evaluations"""
        
        if not self.evaluation_history:
            return {}
        
        total = len(self.evaluation_history)
        correct = sum(1 for e in self.evaluation_history if e.is_correct)
        
        return {
            'total_evaluations': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0,
            'average_pixel_accuracy': np.mean([e.pixel_accuracy for e in self.evaluation_history]),
            'average_execution_time': np.mean([e.execution_time for e in self.evaluation_history]),
            'error_patterns': dict(self.error_patterns)
        }
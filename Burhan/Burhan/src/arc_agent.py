"""
ARC Agent - Main orchestrator for solving ARC Prize 2025 tasks
Integrates multiple solvers and strategies for robust solutions
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import logging
from enum import Enum

from .arc.task_loader import ARCTask, ARCExample
from .arc.grid_operations import Grid
from .strategies.strategy_selector import StrategySelector, TaskCategory
from .solvers.ensemble_solver import EnsembleSolver, SolverType
from .solvers.program_synthesis import ProgramSynthesisEngine
from .solvers.csp_solver import CSPSolver
from .solvers.pattern_solver import PatternSolver


class AgentMode(Enum):
    """Operating modes for the agent"""
    FAST = "fast"  # Quick solving with basic strategies
    BALANCED = "balanced"  # Balance between speed and accuracy
    COMPREHENSIVE = "comprehensive"  # Try all strategies thoroughly
    ADAPTIVE = "adaptive"  # Adapt strategy based on task analysis


@dataclass
class SolutionAttempt:
    """Records a solution attempt"""
    task_id: str
    solution: Optional[np.ndarray]
    confidence: float
    solver_used: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False


@dataclass
class TaskSolution:
    """Complete solution for a task"""
    task_id: str
    test_solutions: List[np.ndarray]
    attempts: List[SolutionAttempt]
    total_time: float
    strategy_used: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ARCAgent:
    """
    Main ARC Agent that orchestrates solving strategies
    """
    
    def __init__(self, 
                 mode: AgentMode = AgentMode.BALANCED,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ARC Agent
        
        Args:
            mode: Operating mode for the agent
            config: Configuration dictionary
        """
        self.mode = mode
        self.config = config or {}
        
        # Initialize components
        self.strategy_selector = StrategySelector()
        self.ensemble_solver = None
        self._initialize_solvers()
        
        # Performance tracking
        self.solutions_cache = {}
        self.performance_stats = {
            'tasks_attempted': 0,
            'tasks_solved': 0,
            'average_time': 0,
            'solver_accuracy': {}
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_solvers(self):
        """Initialize solver ensemble based on mode"""
        
        # Configure solvers based on mode
        if self.mode == AgentMode.FAST:
            solver_config = {
                'timeout_per_solver': 5.0,
                'solvers': [SolverType.SIMPLE_PATTERN, SolverType.PATTERN],
                'use_simple_pattern': True
            }
        elif self.mode == AgentMode.BALANCED:
            solver_config = {
                'timeout_per_solver': 10.0,
                'solvers': [SolverType.SIMPLE_PATTERN, SolverType.PATTERN, SolverType.PROGRAM_SYNTHESIS],
                'use_simple_pattern': True
            }
        elif self.mode == AgentMode.COMPREHENSIVE:
            solver_config = {
                'timeout_per_solver': 20.0,
                'solvers': [SolverType.SIMPLE_PATTERN, SolverType.PATTERN, SolverType.PROGRAM_SYNTHESIS, SolverType.CSP],
                'use_simple_pattern': True
            }
        else:  # ADAPTIVE
            solver_config = {
                'timeout_per_solver': 15.0,
                'solvers': [SolverType.SIMPLE_PATTERN, SolverType.PATTERN, SolverType.PROGRAM_SYNTHESIS, SolverType.CSP],
                'adaptive': True,
                'use_simple_pattern': True
            }
        
        # Merge with user config
        solver_config.update(self.config.get('solver_config', {}))
        
        # Create ensemble solver
        self.ensemble_solver = EnsembleSolver(solver_config)
    
    def solve_task(self, task: ARCTask) -> TaskSolution:
        """
        Solve a complete ARC task
        
        Args:
            task: ARC task to solve
            
        Returns:
            TaskSolution with results
        """
        start_time = time.time()
        self.performance_stats['tasks_attempted'] += 1
        
        # Check cache
        if task.task_id in self.solutions_cache:
            self.logger.info(f"Using cached solution for task {task.task_id}")
            return self.solutions_cache[task.task_id]
        
        # Prepare training data
        train_inputs = [ex.input for ex in task.train_examples]
        train_outputs = [ex.output for ex in task.train_examples]
        
        # Analyze task and select strategy
        recommendation = self.strategy_selector.select_strategy(
            train_inputs, train_outputs
        )
        
        self.logger.info(f"Task {task.task_id}: Using strategy {recommendation.primary_strategy}")
        self.logger.info(f"Expected difficulty: {recommendation.expected_difficulty.name}")
        
        # Configure ensemble based on recommendation
        if self.mode == AgentMode.ADAPTIVE:
            self._adapt_solvers(recommendation)
        
        # Solve test examples
        test_solutions = []
        attempts = []
        
        for i, test_example in enumerate(task.test_examples):
            self.logger.info(f"Solving test example {i+1}/{len(task.test_examples)}")
            
            # Try to solve
            attempt_start = time.time()
            
            try:
                # Use ensemble solver
                result = self.ensemble_solver.solve(
                    train_inputs,
                    train_outputs,
                    test_example.input
                )
                
                if result.final_solution is not None:
                    test_solutions.append(result.final_solution)
                    success = True
                    confidence = result.consensus_score
                    solver_used = result.strategy_used
                else:
                    # Fallback: try simple pattern matching
                    self.logger.warning(f"Ensemble failed, trying fallback for test {i+1}")
                    fallback_solution = self._fallback_solve(
                        train_inputs, train_outputs, test_example.input
                    )
                    test_solutions.append(fallback_solution)
                    success = fallback_solution is not None
                    confidence = 0.3 if success else 0.0
                    solver_used = "fallback"
                
            except Exception as e:
                self.logger.error(f"Error solving test {i+1}: {e}")
                test_solutions.append(None)
                success = False
                confidence = 0.0
                solver_used = "error"
            
            attempt_time = time.time() - attempt_start
            
            # Record attempt
            attempt = SolutionAttempt(
                task_id=task.task_id,
                solution=test_solutions[-1],
                confidence=confidence,
                solver_used=solver_used,
                execution_time=attempt_time,
                success=success
            )
            attempts.append(attempt)
        
        # Create solution
        total_time = time.time() - start_time
        overall_confidence = np.mean([a.confidence for a in attempts])
        
        solution = TaskSolution(
            task_id=task.task_id,
            test_solutions=test_solutions,
            attempts=attempts,
            total_time=total_time,
            strategy_used=recommendation.primary_strategy,
            confidence=overall_confidence,
            metadata={
                'recommendation': recommendation,
                'mode': self.mode.value,
                'num_test_examples': len(task.test_examples)
            }
        )
        
        # Update stats
        if all(s is not None for s in test_solutions):
            self.performance_stats['tasks_solved'] += 1
        
        self.performance_stats['average_time'] = (
            (self.performance_stats['average_time'] * 
             (self.performance_stats['tasks_attempted'] - 1) + total_time) /
            self.performance_stats['tasks_attempted']
        )
        
        # Cache solution
        self.solutions_cache[task.task_id] = solution
        
        return solution
    
    def _adapt_solvers(self, recommendation):
        """Adapt solver configuration based on task analysis"""
        
        # Adjust timeout based on expected difficulty
        difficulty_timeouts = {
            1: 5.0,   # TRIVIAL
            2: 8.0,   # SIMPLE
            3: 12.0,  # MODERATE
            4: 18.0,  # COMPLEX
            5: 25.0   # VERY_COMPLEX
        }
        
        timeout = difficulty_timeouts.get(
            recommendation.expected_difficulty.value, 15.0
        )
        
        # Select solvers based on task category
        if recommendation.primary_strategy == "pattern_matching":
            solvers = [SolverType.PATTERN]
        elif recommendation.primary_strategy == "program_synthesis":
            solvers = [SolverType.PROGRAM_SYNTHESIS, SolverType.PATTERN]
        elif recommendation.primary_strategy == "constraint_satisfaction":
            solvers = [SolverType.CSP, SolverType.PATTERN]
        else:
            solvers = [SolverType.PATTERN, SolverType.PROGRAM_SYNTHESIS]
        
        # Reconfigure ensemble
        new_config = {
            'timeout_per_solver': timeout,
            'solvers': solvers
        }
        self.ensemble_solver = EnsembleSolver(new_config)
    
    def _fallback_solve(self, 
                       train_inputs: List[np.ndarray],
                       train_outputs: List[np.ndarray],
                       test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Simple fallback solver for when main strategies fail
        """
        try:
            # Strategy 1: If all outputs are same, return first output
            if all(np.array_equal(train_outputs[0], out) for out in train_outputs[1:]):
                self.logger.info("Fallback: All outputs identical")
                return train_outputs[0].copy()
            
            # Strategy 2: If all inputs same size as outputs, check for simple transform
            if all(inp.shape == out.shape for inp, out in zip(train_inputs, train_outputs)):
                # Check for simple color mapping
                color_maps = []
                for inp, out in zip(train_inputs, train_outputs):
                    if inp.shape == out.shape:
                        color_map = {}
                        for i in range(inp.shape[0]):
                            for j in range(inp.shape[1]):
                                color_map[inp[i,j]] = out[i,j]
                        color_maps.append(color_map)
                
                # If consistent color mapping
                if all(cm == color_maps[0] for cm in color_maps[1:]):
                    self.logger.info("Fallback: Simple color mapping")
                    result = test_input.copy()
                    for i in range(result.shape[0]):
                        for j in range(result.shape[1]):
                            if result[i,j] in color_maps[0]:
                                result[i,j] = color_maps[0][result[i,j]]
                    return result
            
            # Strategy 3: Check for scaling
            scale_factors = []
            for inp, out in zip(train_inputs, train_outputs):
                if (out.shape[0] % inp.shape[0] == 0 and 
                    out.shape[1] % inp.shape[1] == 0):
                    h_scale = out.shape[0] // inp.shape[0]
                    w_scale = out.shape[1] // inp.shape[1]
                    if h_scale == w_scale:
                        # Verify it's actually scaled
                        scaled = np.repeat(np.repeat(inp, h_scale, axis=0), w_scale, axis=1)
                        if np.array_equal(scaled, out):
                            scale_factors.append(h_scale)
            
            if scale_factors and all(s == scale_factors[0] for s in scale_factors):
                self.logger.info(f"Fallback: Scaling by {scale_factors[0]}")
                factor = scale_factors[0]
                return np.repeat(np.repeat(test_input, factor, axis=0), factor, axis=1)
            
        except Exception as e:
            self.logger.error(f"Fallback solver error: {e}")
        
        return None
    
    def evaluate_solution(self, 
                         task: ARCTask,
                         solution: TaskSolution) -> Dict[str, Any]:
        """
        Evaluate a solution against ground truth if available
        
        Args:
            task: Original task
            solution: Generated solution
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            'task_id': task.task_id,
            'num_test_examples': len(task.test_examples),
            'solutions_found': sum(1 for s in solution.test_solutions if s is not None),
            'confidence': solution.confidence,
            'execution_time': solution.total_time
        }
        
        # If we have ground truth
        correct = 0
        for i, (test_ex, sol) in enumerate(zip(task.test_examples, solution.test_solutions)):
            if test_ex.output is not None and sol is not None:
                if np.array_equal(test_ex.output, sol):
                    correct += 1
                    self.logger.info(f"Test {i+1}: Correct!")
                else:
                    self.logger.info(f"Test {i+1}: Incorrect")
        
        if any(ex.output is not None for ex in task.test_examples):
            metrics['accuracy'] = correct / len(task.test_examples)
            metrics['correct'] = correct
        
        return metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return self.performance_stats.copy()
    
    def reset_cache(self):
        """Clear solution cache"""
        self.solutions_cache.clear()
        self.logger.info("Solution cache cleared")
"""
Ensemble Solver for ARC Prize 2025
Combines multiple solving strategies for robust solutions
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import time
from collections import Counter

from ..arc.grid_operations import Grid
from .program_synthesis import ProgramSynthesisEngine
from .csp_solver import CSPSolver
from .pattern_solver import PatternSolver
from .simple_pattern_solver import SimplePatternSolver


class SolverType(Enum):
    """Types of solvers in the ensemble"""
    PROGRAM_SYNTHESIS = "program_synthesis"
    CSP = "csp"
    PATTERN = "pattern"
    SIMPLE_PATTERN = "simple_pattern"
    HYBRID = "hybrid"


@dataclass
class SolverResult:
    """Result from a single solver"""
    solver_type: SolverType
    solution: Optional[np.ndarray]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Combined result from ensemble"""
    final_solution: Optional[np.ndarray]
    individual_results: List[SolverResult]
    consensus_score: float
    total_time: float
    strategy_used: str


class SolverWrapper:
    """Wrapper for individual solvers with timeout and error handling"""
    
    def __init__(self, 
                 solver_type: SolverType,
                 solver_instance: Any,
                 timeout: float = 10.0):
        self.solver_type = solver_type
        self.solver = solver_instance
        self.timeout = timeout
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> SolverResult:
        """Solve with timeout and error handling"""
        
        start_time = time.time()
        
        try:
            # Execute solver with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.solver.solve,
                    train_inputs,
                    train_outputs,
                    test_input
                )
                
                try:
                    solution = future.result(timeout=self.timeout)
                    execution_time = time.time() - start_time
                    
                    if solution is not None:
                        confidence = self._calculate_confidence(
                            solution, train_inputs, train_outputs
                        )
                    else:
                        confidence = 0.0
                    
                    return SolverResult(
                        solver_type=self.solver_type,
                        solution=solution,
                        confidence=confidence,
                        execution_time=execution_time
                    )
                    
                except concurrent.futures.TimeoutError:
                    return SolverResult(
                        solver_type=self.solver_type,
                        solution=None,
                        confidence=0.0,
                        execution_time=self.timeout,
                        metadata={'error': 'timeout'}
                    )
                    
        except Exception as e:
            return SolverResult(
                solver_type=self.solver_type,
                solution=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _calculate_confidence(self,
                            solution: np.ndarray,
                            train_inputs: List[np.ndarray],
                            train_outputs: List[np.ndarray]) -> float:
        """Calculate confidence score for solution"""
        
        # Basic confidence based on solver type
        base_confidence = {
            SolverType.PROGRAM_SYNTHESIS: 0.8,
            SolverType.CSP: 0.7,
            SolverType.PATTERN: 0.75,
            SolverType.SIMPLE_PATTERN: 0.9,
            SolverType.HYBRID: 0.85
        }.get(self.solver_type, 0.5)
        
        # Adjust based on solution characteristics
        if solution is not None:
            # Check if solution size is reasonable
            avg_output_size = np.mean([o.size for o in train_outputs])
            size_ratio = solution.size / avg_output_size if avg_output_size > 0 else 1.0
            
            if 0.5 <= size_ratio <= 2.0:
                base_confidence *= 1.1
            else:
                base_confidence *= 0.9
            
            # Check if colors are reasonable
            solution_colors = set(np.unique(solution))
            train_colors = set()
            for out in train_outputs:
                train_colors.update(np.unique(out))
            
            if solution_colors.issubset(train_colors):
                base_confidence *= 1.1
            else:
                base_confidence *= 0.8
        
        return min(1.0, base_confidence)


class VotingMechanism:
    """Voting mechanism for solution selection"""
    
    def __init__(self, voting_type: str = "weighted"):
        self.voting_type = voting_type
    
    def vote(self, results: List[SolverResult]) -> Optional[np.ndarray]:
        """Select solution through voting"""
        
        if not results:
            return None
        
        # Filter out None solutions
        valid_results = [r for r in results if r.solution is not None]
        
        if not valid_results:
            return None
        
        if len(valid_results) == 1:
            return valid_results[0].solution
        
        if self.voting_type == "weighted":
            return self._weighted_vote(valid_results)
        elif self.voting_type == "majority":
            return self._majority_vote(valid_results)
        elif self.voting_type == "confidence":
            return self._confidence_vote(valid_results)
        else:
            return valid_results[0].solution
    
    def _weighted_vote(self, results: List[SolverResult]) -> np.ndarray:
        """Weighted voting based on confidence scores"""
        
        # Group solutions by similarity
        solution_groups = self._group_similar_solutions(results)
        
        # Calculate weighted score for each group
        group_scores = {}
        for group_id, group_results in solution_groups.items():
            total_weight = sum(r.confidence for r in group_results)
            group_scores[group_id] = total_weight
        
        # Select group with highest score
        best_group = max(group_scores, key=group_scores.get)
        best_results = solution_groups[best_group]
        
        # Return solution with highest confidence in best group
        return max(best_results, key=lambda r: r.confidence).solution
    
    def _majority_vote(self, results: List[SolverResult]) -> np.ndarray:
        """Simple majority voting"""
        
        # Group identical solutions
        solution_counts = Counter()
        solution_map = {}
        
        for result in results:
            # Create hashable representation
            solution_key = tuple(result.solution.flatten())
            solution_counts[solution_key] += 1
            solution_map[solution_key] = result.solution
        
        # Return most common solution
        most_common = solution_counts.most_common(1)[0][0]
        return solution_map[most_common]
    
    def _confidence_vote(self, results: List[SolverResult]) -> np.ndarray:
        """Select solution with highest confidence"""
        return max(results, key=lambda r: r.confidence).solution
    
    def _group_similar_solutions(self, 
                                results: List[SolverResult]) -> Dict[int, List[SolverResult]]:
        """Group solutions by similarity"""
        
        groups = {}
        group_id = 0
        
        for result in results:
            found_group = False
            
            for gid, group_results in groups.items():
                # Check if solution is similar to group
                representative = group_results[0].solution
                
                if self._solutions_similar(result.solution, representative):
                    groups[gid].append(result)
                    found_group = True
                    break
            
            if not found_group:
                groups[group_id] = [result]
                group_id += 1
        
        return groups
    
    def _solutions_similar(self, sol1: np.ndarray, sol2: np.ndarray) -> bool:
        """Check if two solutions are similar"""
        
        # Same shape
        if sol1.shape != sol2.shape:
            return False
        
        # Calculate similarity
        matches = np.sum(sol1 == sol2)
        total = sol1.size
        similarity = matches / total if total > 0 else 0
        
        return similarity > 0.9


class HybridSolver:
    """Hybrid solver combining multiple strategies"""
    
    def __init__(self):
        self.program_synthesis = ProgramSynthesisEngine()
        self.pattern_solver = PatternSolver()
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using hybrid approach"""
        
        # Try pattern detection first (faster)
        pattern_solution = self.pattern_solver.solve(
            train_inputs, train_outputs, test_input
        )
        
        if pattern_solution is not None:
            return pattern_solution
        
        # Fall back to program synthesis
        return self.program_synthesis.solve(
            train_inputs, train_outputs, test_input
        )


class EnsembleSolver:
    """Main ensemble solver combining multiple strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize component solvers
        self.solvers = self._initialize_solvers(config)
        
        # Voting mechanism
        self.voting = VotingMechanism(
            voting_type=config.get('voting_type', 'weighted')
        )
        
        # Configuration
        self.parallel = config.get('parallel', True)
        self.timeout_per_solver = config.get('timeout_per_solver', 10.0)
        self.min_consensus = config.get('min_consensus', 0.5)
        self.adaptive = config.get('adaptive', True)
    
    def _initialize_solvers(self, config: Dict[str, Any]) -> List[SolverWrapper]:
        """Initialize component solvers"""
        solvers = []
        
        # Program Synthesis Solver
        if config.get('use_program_synthesis', True):
            solver = ProgramSynthesisEngine(
                config.get('program_synthesis_config', {})
            )
            wrapper = SolverWrapper(
                SolverType.PROGRAM_SYNTHESIS,
                solver,
                config.get('timeout_per_solver', 10.0)
            )
            solvers.append(wrapper)
        
        # CSP Solver
        if config.get('use_csp', True):
            solver = CSPSolver(
                config.get('csp_config', {})
            )
            wrapper = SolverWrapper(
                SolverType.CSP,
                solver,
                config.get('timeout_per_solver', 10.0)
            )
            solvers.append(wrapper)
        
        # Simple Pattern Solver (try first as it's fastest)
        if config.get('use_simple_pattern', True):
            solver = SimplePatternSolver(
                config.get('simple_pattern_config', {})
            )
            wrapper = SolverWrapper(
                SolverType.SIMPLE_PATTERN,
                solver,
                config.get('timeout_per_solver', 5.0)
            )
            solvers.insert(0, wrapper)  # Add to beginning for priority
        
        # Pattern Solver
        if config.get('use_pattern', True):
            solver = PatternSolver(
                config.get('pattern_config', {})
            )
            wrapper = SolverWrapper(
                SolverType.PATTERN,
                solver,
                config.get('timeout_per_solver', 10.0)
            )
            solvers.append(wrapper)
        
        # Hybrid Solver
        if config.get('use_hybrid', True):
            solver = HybridSolver()
            wrapper = SolverWrapper(
                SolverType.HYBRID,
                solver,
                config.get('timeout_per_solver', 15.0)
            )
            solvers.append(wrapper)
        
        return solvers
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> EnsembleResult:
        """Solve using ensemble approach"""
        
        start_time = time.time()
        
        # Select solvers based on task characteristics
        if self.adaptive:
            selected_solvers = self._select_solvers(
                train_inputs, train_outputs, test_input
            )
        else:
            selected_solvers = self.solvers
        
        # Execute solvers
        if self.parallel:
            results = self._solve_parallel(
                selected_solvers, train_inputs, train_outputs, test_input
            )
        else:
            results = self._solve_sequential(
                selected_solvers, train_inputs, train_outputs, test_input
            )
        
        # Vote on solution
        final_solution = self.voting.vote(results)
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(results)
        
        # Determine strategy used
        if final_solution is not None:
            winning_result = next(
                (r for r in results if r.solution is not None and 
                 np.array_equal(r.solution, final_solution)), None
            )
            strategy_used = winning_result.solver_type.value if winning_result else "unknown"
        else:
            strategy_used = "none"
        
        total_time = time.time() - start_time
        
        return EnsembleResult(
            final_solution=final_solution,
            individual_results=results,
            consensus_score=consensus_score,
            total_time=total_time,
            strategy_used=strategy_used
        )
    
    def _select_solvers(self,
                       train_inputs: List[np.ndarray],
                       train_outputs: List[np.ndarray],
                       test_input: np.ndarray) -> List[SolverWrapper]:
        """Select appropriate solvers based on task characteristics"""
        
        selected = []
        
        # Analyze task characteristics
        characteristics = self._analyze_task(train_inputs, train_outputs, test_input)
        
        # Select solvers based on characteristics
        for solver in self.solvers:
            if self._should_use_solver(solver.solver_type, characteristics):
                selected.append(solver)
        
        # Ensure at least one solver is selected
        if not selected:
            selected = [self.solvers[0]]
        
        return selected
    
    def _analyze_task(self,
                     train_inputs: List[np.ndarray],
                     train_outputs: List[np.ndarray],
                     test_input: np.ndarray) -> Dict[str, Any]:
        """Analyze task characteristics"""
        
        characteristics = {
            'size_change': False,
            'color_change': False,
            'pattern_repetition': False,
            'symmetry': False,
            'complexity': 'low'
        }
        
        # Check for size changes
        for inp, out in zip(train_inputs, train_outputs):
            if inp.shape != out.shape:
                characteristics['size_change'] = True
                break
        
        # Check for color changes
        for inp, out in zip(train_inputs, train_outputs):
            if set(np.unique(inp)) != set(np.unique(out)):
                characteristics['color_change'] = True
                break
        
        # Check for patterns
        for out in train_outputs:
            grid = Grid(out)
            from ..arc.pattern_detector import PatternDetector
            detector = PatternDetector(grid)
            
            if detector.find_repeating_patterns():
                characteristics['pattern_repetition'] = True
            
            symmetries = detector.get_symmetries()
            if any(symmetries.values()):
                characteristics['symmetry'] = True
        
        # Estimate complexity
        avg_size = np.mean([o.size for o in train_outputs])
        if avg_size > 100:
            characteristics['complexity'] = 'high'
        elif avg_size > 25:
            characteristics['complexity'] = 'medium'
        
        return characteristics
    
    def _should_use_solver(self,
                         solver_type: SolverType,
                         characteristics: Dict[str, Any]) -> bool:
        """Determine if solver should be used for task"""
        
        if solver_type == SolverType.SIMPLE_PATTERN:
            # Best for simple repetition and tiling tasks
            return (characteristics['size_change'] and
                   characteristics['pattern_repetition'])
        
        elif solver_type == SolverType.PATTERN:
            # Good for pattern-based tasks
            return (characteristics['pattern_repetition'] or 
                   characteristics['symmetry'] or
                   characteristics['size_change'])
        
        elif solver_type == SolverType.PROGRAM_SYNTHESIS:
            # Good for complex transformations
            return (characteristics['complexity'] in ['medium', 'high'] or
                   characteristics['color_change'])
        
        elif solver_type == SolverType.CSP:
            # Good for constraint-based tasks
            return (characteristics['color_change'] or
                   not characteristics['size_change'])
        
        elif solver_type == SolverType.HYBRID:
            # Always useful as fallback
            return True
        
        return True
    
    def _solve_parallel(self,
                       solvers: List[SolverWrapper],
                       train_inputs: List[np.ndarray],
                       train_outputs: List[np.ndarray],
                       test_input: np.ndarray) -> List[SolverResult]:
        """Execute solvers in parallel"""
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(solvers)) as executor:
            futures = []
            
            for solver in solvers:
                future = executor.submit(
                    solver.solve,
                    train_inputs,
                    train_outputs,
                    test_input
                )
                futures.append((solver.solver_type, future))
            
            for solver_type, future in futures:
                try:
                    result = future.result(timeout=self.timeout_per_solver * 2)
                    results.append(result)
                except Exception as e:
                    results.append(SolverResult(
                        solver_type=solver_type,
                        solution=None,
                        confidence=0.0,
                        execution_time=0.0,
                        metadata={'error': str(e)}
                    ))
        
        return results
    
    def _solve_sequential(self,
                        solvers: List[SolverWrapper],
                        train_inputs: List[np.ndarray],
                        train_outputs: List[np.ndarray],
                        test_input: np.ndarray) -> List[SolverResult]:
        """Execute solvers sequentially"""
        
        results = []
        
        for solver in solvers:
            result = solver.solve(train_inputs, train_outputs, test_input)
            results.append(result)
            
            # Early stopping if high confidence solution found
            if result.confidence > 0.95:
                break
        
        return results
    
    def _calculate_consensus(self, results: List[SolverResult]) -> float:
        """Calculate consensus score among solvers"""
        
        valid_results = [r for r in results if r.solution is not None]
        
        if len(valid_results) <= 1:
            return 0.0 if not valid_results else 1.0
        
        # Group similar solutions
        groups = {}
        for result in valid_results:
            found_group = False
            for group_id, group_results in groups.items():
                if self._solutions_match(result.solution, group_results[0].solution):
                    groups[group_id].append(result)
                    found_group = True
                    break
            
            if not found_group:
                groups[len(groups)] = [result]
        
        # Calculate consensus as ratio of largest group
        largest_group_size = max(len(g) for g in groups.values())
        consensus = largest_group_size / len(valid_results)
        
        return consensus
    
    def _solutions_match(self, sol1: np.ndarray, sol2: np.ndarray) -> bool:
        """Check if two solutions match exactly"""
        return sol1.shape == sol2.shape and np.array_equal(sol1, sol2)
    
    def get_solution(self,
                    train_inputs: List[np.ndarray],
                    train_outputs: List[np.ndarray],
                    test_input: np.ndarray) -> Optional[np.ndarray]:
        """Get solution (simplified interface)"""
        
        result = self.solve(train_inputs, train_outputs, test_input)
        return result.final_solution
"""
Unified Interfaces for ARC Project Components
Provides consistent interfaces and base classes for all components
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from error_manager import safe_execute, ErrorSeverity

@dataclass
class TaskData:
    """Standardized task data structure"""
    task_id: str
    train_pairs: List[Dict[str, np.ndarray]]
    test_input: np.ndarray
    test_output: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """Standardized analysis result structure"""
    component_name: str
    analysis_type: str
    confidence: float
    features: Dict[str, Any]
    patterns: List[str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResult:
    """Standardized prediction result structure"""
    prediction: np.ndarray
    confidence: float
    method_used: str
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None

class BaseAnalyzer(ABC):
    """Base class for all analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    @abstractmethod
    def analyze(self, task_data: TaskData) -> AnalysisResult:
        """Analyze task data and return results"""
        pass
    
    @safe_execute("BaseAnalyzer", "initialize", ErrorSeverity.MEDIUM, True)
    def initialize(self) -> bool:
        """Initialize the analyzer"""
        self.is_initialized = True
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'initialized': self.is_initialized
        }

class BaseStrategy(ABC):
    """Base class for all strategies"""
    
    def __init__(self, name: str, cost: float = 1.0):
        self.name = name
        self.cost = cost
        self.success_count = 0
        self.failure_count = 0
    
    @abstractmethod
    def apply(self, grid: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """Apply strategy to grid"""
        pass
    
    @safe_execute("BaseStrategy", "validate", ErrorSeverity.LOW, 0.0)
    def validate(self, train_pairs: List[Dict[str, np.ndarray]]) -> float:
        """Validate strategy on training pairs"""
        if not train_pairs:
            return 0.0
        
        correct = 0
        for pair in train_pairs:
            try:
                result = self.apply(pair['input'], {})
                if np.array_equal(result, pair['output']):
                    correct += 1
            except Exception:
                continue
        
        return correct / len(train_pairs)
    
    def update_stats(self, success: bool):
        """Update success/failure statistics"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

class BaseEngine(ABC):
    """Base class for all engines"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_active = False
        self.performance_metrics = {}
    
    @abstractmethod
    def process(self, task_data: TaskData) -> PredictionResult:
        """Process task data and return prediction"""
        pass
    
    @safe_execute("BaseEngine", "start", ErrorSeverity.MEDIUM, False)
    def start(self) -> bool:
        """Start the engine"""
        self.is_active = True
        return True
    
    @safe_execute("BaseEngine", "stop", ErrorSeverity.LOW, True)
    def stop(self) -> bool:
        """Stop the engine"""
        self.is_active = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'active': self.is_active,
            'metrics': self.performance_metrics.copy()
        }

class ComponentRegistry:
    """Registry for all system components"""
    
    def __init__(self):
        self.analyzers: Dict[str, BaseAnalyzer] = {}
        self.strategies: Dict[str, BaseStrategy] = {}
        self.engines: Dict[str, BaseEngine] = {}
    
    def register_analyzer(self, analyzer: BaseAnalyzer):
        """Register an analyzer"""
        self.analyzers[analyzer.name] = analyzer
    
    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy"""
        self.strategies[strategy.name] = strategy
    
    def register_engine(self, engine: BaseEngine):
        """Register an engine"""
        self.engines[engine.name] = engine
    
    def get_analyzer(self, name: str) -> Optional[BaseAnalyzer]:
        """Get analyzer by name"""
        return self.analyzers.get(name)
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get strategy by name"""
        return self.strategies.get(name)
    
    def get_engine(self, name: str) -> Optional[BaseEngine]:
        """Get engine by name"""
        return self.engines.get(name)
    
    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components"""
        return {
            'analyzers': list(self.analyzers.keys()),
            'strategies': list(self.strategies.keys()),
            'engines': list(self.engines.keys())
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'total_analyzers': len(self.analyzers),
            'total_strategies': len(self.strategies),
            'total_engines': len(self.engines),
            'active_engines': sum(1 for e in self.engines.values() if e.is_active),
            'initialized_analyzers': sum(1 for a in self.analyzers.values() if a.is_initialized)
        }

# Global component registry
component_registry = ComponentRegistry()

# Utility functions for data conversion
def convert_to_task_data(raw_task: Dict[str, Any], task_id: str = None) -> TaskData:
    """Convert raw task data to standardized TaskData format"""
    if task_id is None:
        task_id = raw_task.get('id', 'unknown')
    
    train_pairs = []
    for pair in raw_task.get('train', []):
        train_pairs.append({
            'input': np.array(pair['input']),
            'output': np.array(pair['output'])
        })
    
    test_data = raw_task.get('test', [{}])[0]
    test_input = np.array(test_data.get('input', []))
    test_output = None
    if 'output' in test_data:
        test_output = np.array(test_data['output'])
    
    return TaskData(
        task_id=task_id,
        train_pairs=train_pairs,
        test_input=test_input,
        test_output=test_output,
        metadata=raw_task.get('metadata', {})
    )

def validate_grid(grid: Any) -> bool:
    """Validate that input is a proper grid"""
    try:
        if not isinstance(grid, np.ndarray):
            grid = np.array(grid)
        return grid.ndim == 2 and grid.size > 0
    except Exception:
        return False

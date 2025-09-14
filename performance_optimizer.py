"""
Performance Optimization System for ARC Project
Provides memory management, caching, and performance monitoring
"""
from __future__ import annotations
import gc
import time
import threading

# Safe import for psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil for fallback
    class psutil:
        class Process:
            def memory_info(self):
                return type('MemoryInfo', (), {'rss': 100 * 1024 * 1024})()  # 100MB default

        @staticmethod
        def cpu_percent():
            return 0.0
from collections.abc import Callable
from typing import Any, Dict, Optional, Callable, Tuple
from functools import wraps, lru_cache
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hits: int = 0
    cache_misses: int = 0

class MemoryManager:
    """Memory management and monitoring"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.current_usage = 0
        self.peak_usage = 0
        self.cleanup_threshold = 0.8  # 80% of max memory
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            usage_mb = memory_info.rss / 1024 / 1024
            self.current_usage = usage_mb
            if usage_mb > self.peak_usage:
                self.peak_usage = usage_mb
            return usage_mb
        except Exception:
            return 0.0
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        current = self.get_memory_usage()
        return current > (self.max_memory_mb * self.cleanup_threshold)
    
    def cleanup(self) -> float:
        """Perform memory cleanup"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        freed = before - after
        logger.info(f"Memory cleanup: freed {freed:.2f} MB")
        return freed
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        return {
            'current_mb': self.current_usage,
            'peak_mb': self.peak_usage,
            'max_mb': self.max_memory_mb,
            'usage_percent': (self.current_usage / self.max_memory_mb) * 100
        }

class CacheManager:
    """Intelligent caching system"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key], True
            else:
                self.miss_count += 1
                return None, False
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_oldest(self):
        """Evict oldest item from cache"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'usage_percent': (len(self.cache) / self.max_size) * 100
        }

class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.metrics_history: list[PerformanceMetrics] = []
        self.max_history = 1000
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
    
    def measure_performance(self, func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Measure function performance"""
        start_time = time.time()
        start_memory = self.memory_manager.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = e
            success = False
        
        end_time = time.time()
        end_memory = self.memory_manager.get_memory_usage()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=psutil.cpu_percent()
        )
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        if not success:
            raise result
        
        return result, metrics
    
    def get_average_metrics(self, last_n: int = 100) -> Optional[PerformanceMetrics]:
        """Get average metrics for last N operations"""
        if not self.metrics_history:
            return None
        
        recent_metrics = self.metrics_history[-last_n:]
        
        avg_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        
        return PerformanceMetrics(
            execution_time=avg_time,
            memory_usage=avg_memory,
            cpu_usage=avg_cpu
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system performance status"""
        return {
            'memory': self.memory_manager.get_stats(),
            'cache': self.cache_manager.get_stats(),
            'metrics_count': len(self.metrics_history),
            'average_metrics': self.get_average_metrics().__dict__ if self.get_average_metrics() else None
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

def cached(cache_key_func: Optional[Callable] = None, ttl: Optional[float] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result, hit = performance_monitor.cache_manager.get(cache_key)
            if hit:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            performance_monitor.cache_manager.put(cache_key, result)
            
            return result
        return wrapper
    return decorator

def monitor_performance(func: Callable) -> Callable:
    """Decorator for monitoring function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result, metrics = performance_monitor.measure_performance(func, *args, **kwargs)
        
        # Log performance if execution time is high
        if metrics.execution_time > 1.0:  # More than 1 second
            logger.warning(f"{func.__name__} took {metrics.execution_time:.2f}s")
        
        # Check memory usage
        if performance_monitor.memory_manager.should_cleanup():
            performance_monitor.memory_manager.cleanup()
        
        return result
    return wrapper

def optimize_memory(func: Callable) -> Callable:
    """Decorator for memory optimization"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory before execution
        if performance_monitor.memory_manager.should_cleanup():
            performance_monitor.memory_manager.cleanup()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after heavy operations
            if performance_monitor.memory_manager.get_memory_usage() > 1000:  # > 1GB
                gc.collect()
    
    return wrapper

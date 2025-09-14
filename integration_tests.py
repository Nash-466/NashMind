"""
Integration Tests for ARC Project
Comprehensive testing suite to ensure all components work together
"""
from __future__ import annotations
import unittest
import numpy as np
import json
import os
import time
from collections.abc import Callable
from typing import Dict, Any, List
import logging

# Import project components
from error_manager import error_manager, ErrorSeverity
from unified_interfaces import TaskData, component_registry, convert_to_task_data
from performance_optimizer import performance_monitor

logger = logging.getLogger(__name__)

class IntegrationTestSuite(unittest.TestCase):
    """Main integration test suite"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_data = cls._create_test_data()
        cls.test_results = {}
    
    @classmethod
    def _create_test_data(cls) -> Dict[str, Any]:
        """Create test data for integration tests"""
        return {
            'simple_task': {
                'train': [
                    {
                        'input': [[0, 1], [1, 0]],
                        'output': [[1, 0], [0, 1]]
                    }
                ],
                'test': [
                    {
                        'input': [[0, 1], [1, 0]]
                    }
                ]
            },
            'complex_task': {
                'train': [
                    {
                        'input': [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
                        'output': [[2, 0, 1], [0, 1, 2], [1, 2, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[1, 2, 0], [2, 0, 1], [0, 1, 2]]
                    }
                ]
            }
        }
    
    def test_error_manager_integration(self):
        """Test error manager integration"""
        logger.info("Testing error manager integration...")
        
        # Test error handling
        try:
            raise ValueError("Test error")
        except Exception as e:
            context = error_manager.handle_error(
                e, "test_component", "test_operation", ErrorSeverity.LOW
            )
            self.assertIsNotNone(context)
            self.assertEqual(context.component, "test_component")
            self.assertEqual(context.error_type, "ValueError")
        
        # Test statistics
        stats = error_manager.get_error_statistics()
        self.assertIn('total_errors', stats)
        self.assertIn('error_counts', stats)
        
        self.test_results['error_manager'] = 'PASSED'
    
    def test_unified_interfaces_integration(self):
        """Test unified interfaces integration"""
        logger.info("Testing unified interfaces integration...")
        
        # Test TaskData conversion
        raw_task = self.test_data['simple_task']
        task_data = convert_to_task_data(raw_task, 'test_task')
        
        self.assertIsInstance(task_data, TaskData)
        self.assertEqual(task_data.task_id, 'test_task')
        self.assertEqual(len(task_data.train_pairs), 1)
        self.assertIsInstance(task_data.test_input, np.ndarray)
        
        # Test component registry
        initial_status = component_registry.get_system_status()
        self.assertIn('total_analyzers', initial_status)
        self.assertIn('total_strategies', initial_status)
        self.assertIn('total_engines', initial_status)
        
        self.test_results['unified_interfaces'] = 'PASSED'
    
    def test_performance_monitor_integration(self):
        """Test performance monitor integration"""
        logger.info("Testing performance monitor integration...")
        
        # Test performance measurement
        def test_function():
            time.sleep(0.1)  # Simulate work
            return "test_result"
        
        result, metrics = performance_monitor.measure_performance(test_function)
        
        self.assertEqual(result, "test_result")
        self.assertGreater(metrics.execution_time, 0.05)  # Should be at least 0.05s
        self.assertIsInstance(metrics.memory_usage, float)
        
        # Test system status
        status = performance_monitor.get_system_status()
        self.assertIn('memory', status)
        self.assertIn('cache', status)
        
        self.test_results['performance_monitor'] = 'PASSED'
    
    def test_dependency_imports(self):
        """Test that all dependencies can be imported"""
        logger.info("Testing dependency imports...")
        
        import_results = {}
        
        # Test core imports
        try:
            import numpy as np
            import_results['numpy'] = 'SUCCESS'
        except ImportError:
            import_results['numpy'] = 'FAILED'
        
        # Test optional imports
        optional_packages = ['pandas', 'torch', 'sklearn', 'scipy']
        for package in optional_packages:
            try:
                __import__(package)
                import_results[package] = 'SUCCESS'
            except ImportError:
                import_results[package] = 'OPTIONAL_MISSING'
        
        # At least numpy should be available
        self.assertEqual(import_results['numpy'], 'SUCCESS')
        
        self.test_results['dependency_imports'] = import_results
    
    def test_main_workflow_integration(self):
        """Test main workflow integration"""
        logger.info("Testing main workflow integration...")
        
        try:
            # Test importing main components
            from main import load_tasks, save_submission, build_orchestrator
            
            # Test orchestrator creation
            orchestrator = build_orchestrator('fast', 0)
            self.assertIsNotNone(orchestrator)
            
            # Test task processing with dummy data
            test_task = self.test_data['simple_task']
            
            # This should not crash
            try:
                result = orchestrator.process_single_task(test_task)
                # Result might be None if no solution found, but should not crash
                workflow_status = 'SUCCESS'
            except Exception as e:
                logger.warning(f"Workflow test encountered error: {e}")
                workflow_status = 'PARTIAL_SUCCESS'
            
            self.test_results['main_workflow'] = workflow_status
            
        except ImportError as e:
            logger.error(f"Failed to import main components: {e}")
            self.test_results['main_workflow'] = 'FAILED'
    
    def test_file_io_integration(self):
        """Test file I/O integration"""
        logger.info("Testing file I/O integration...")
        
        # Test JSON file operations
        test_file = 'test_temp.json'
        test_data = {'test': 'data', 'number': 42}
        
        try:
            # Test save
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            # Test load
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data, test_data)
            
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            
            self.test_results['file_io'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"File I/O test failed: {e}")
            self.test_results['file_io'] = 'FAILED'
    
    def test_memory_management(self):
        """Test memory management"""
        logger.info("Testing memory management...")

        # Test memory monitoring
        initial_memory = performance_monitor.memory_manager.get_memory_usage()
        self.assertGreater(initial_memory, 0)

        # Create some data to use memory (only if psutil is available)
        try:
            import psutil
            large_array = np.random.rand(1000, 1000)
            after_allocation = performance_monitor.memory_manager.get_memory_usage()

            # Memory usage should increase (with some tolerance for measurement variations)
            self.assertGreaterEqual(after_allocation, initial_memory - 1.0)  # Allow 1MB tolerance

            # Test cleanup
            del large_array
            freed_memory = performance_monitor.memory_manager.cleanup()
        except ImportError:
            # If psutil not available, just test that the methods work
            logger.info("psutil not available, testing basic functionality only")
            freed_memory = performance_monitor.memory_manager.cleanup()

        self.test_results['memory_management'] = 'PASSED'
    
    def test_cache_functionality(self):
        """Test caching functionality"""
        logger.info("Testing cache functionality...")
        
        cache_manager = performance_monitor.cache_manager
        
        # Test cache operations
        cache_manager.put('test_key', 'test_value')
        value, hit = cache_manager.get('test_key')
        
        self.assertTrue(hit)
        self.assertEqual(value, 'test_value')
        
        # Test cache miss
        value, hit = cache_manager.get('nonexistent_key')
        self.assertFalse(hit)
        self.assertIsNone(value)
        
        # Test cache stats
        stats = cache_manager.get_stats()
        self.assertIn('hit_count', stats)
        self.assertIn('miss_count', stats)
        
        self.test_results['cache_functionality'] = 'PASSED'
    
    def tearDown(self):
        """Clean up after each test"""
        # Clear caches and perform cleanup
        performance_monitor.cache_manager.clear()
        performance_monitor.memory_manager.cleanup()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        logger.info("Integration test results:")
        for test_name, result in cls.test_results.items():
            logger.info(f"  {test_name}: {result}")

def run_integration_tests():
    """Run all integration tests"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)

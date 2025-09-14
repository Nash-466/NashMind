"""
Simple Pattern Solver for ARC Prize 2025
Specialized solver for simple repetition and tiling patterns
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from ..arc.grid_operations import Grid


class SimplePatternSolver:
    """Solver specialized for simple pattern repetition tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.debug = config.get('debug', False)
    
    def solve(self,
             train_inputs: List[np.ndarray],
             train_outputs: List[np.ndarray],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using simple pattern detection"""
        
        # Try different pattern strategies
        strategies = [
            self._solve_alternating_tile_pattern,
            self._solve_simple_tiling,
            self._solve_distributed_pattern,
            self._solve_scaled_tiling,
            self._solve_mirror_tiling
        ]
        
        for strategy in strategies:
            try:
                solution = strategy(train_inputs, train_outputs, test_input)
                if solution is not None:
                    # Validate on training examples
                    if self._validate_solution(strategy, train_inputs, train_outputs):
                        if self.debug:
                            print(f"Solution found using: {strategy.__name__}")
                        return solution
            except Exception as e:
                if self.debug:
                    print(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        return None
    
    def _solve_alternating_tile_pattern(self,
                                       train_inputs: List[np.ndarray],
                                       train_outputs: List[np.ndarray],
                                       test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve patterns where a small grid is tiled with alternating row/column patterns.
        Example: 2x2 -> 6x6 where rows alternate between original and swapped columns
        """
        
        # Check if this is a tiling pattern with alternating rows
        for inp, out in zip(train_inputs, train_outputs):
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Check for 3x3 tiling of 2x2 input (Task 00576224 pattern)
            if h_in == 2 and w_in == 2 and h_out == 6 and w_out == 6:
                # Build the expected output
                expected = np.zeros((6, 6), dtype=np.int8)
                
                # Pattern for 00576224: Alternating rows with column swap
                # Row 0-1: Original pattern repeated 3 times
                expected[0] = np.tile(inp[0], 3)  # First row repeated
                expected[1] = np.tile(inp[1], 3)  # Second row repeated
                
                # Row 2-3: Swapped columns pattern
                swapped = inp[:, [1, 0]]  # Swap columns
                expected[2] = np.tile(swapped[0], 3)
                expected[3] = np.tile(swapped[1], 3)
                
                # Row 4-5: Original pattern again
                expected[4] = np.tile(inp[0], 3)
                expected[5] = np.tile(inp[1], 3)
                
                if np.array_equal(expected, out):
                    # Apply same transformation to test input
                    if test_input.shape == (2, 2):
                        result = np.zeros((6, 6), dtype=np.int8)
                        
                        # Row 0-1: Original pattern
                        result[0] = np.tile(test_input[0], 3)
                        result[1] = np.tile(test_input[1], 3)
                        
                        # Row 2-3: Swapped columns
                        test_swapped = test_input[:, [1, 0]]
                        result[2] = np.tile(test_swapped[0], 3)
                        result[3] = np.tile(test_swapped[1], 3)
                        
                        # Row 4-5: Original pattern
                        result[4] = np.tile(test_input[0], 3)
                        result[5] = np.tile(test_input[1], 3)
                        
                        return result
        
        return None
    
    def _solve_simple_tiling(self,
                           train_inputs: List[np.ndarray],
                           train_outputs: List[np.ndarray],
                           test_input: np.ndarray) -> Optional[np.ndarray]:
        """Simple tiling where input is repeated in a grid pattern"""
        
        # Analyze the training examples
        tile_params = []
        
        for inp, out in zip(train_inputs, train_outputs):
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Check if output dimensions are multiples of input
            if h_out % h_in == 0 and w_out % w_in == 0:
                rows = h_out // h_in
                cols = w_out // w_in
                
                # Verify simple tiling
                expected = np.tile(inp, (rows, cols))
                if np.array_equal(expected, out):
                    tile_params.append((rows, cols))
        
        # If all training examples use same tiling
        if tile_params and len(set(tile_params)) == 1:
            rows, cols = tile_params[0]
            return np.tile(test_input, (rows, cols))
        
        return None
    
    def _solve_distributed_pattern(self,
                                  train_inputs: List[np.ndarray],
                                  train_outputs: List[np.ndarray],
                                  test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve patterns where input is distributed across output in specific positions.
        Example: 3x3 -> 9x9 with pattern in corners and center (Task 007bbfb7)
        """
        
        for inp, out in zip(train_inputs, train_outputs):
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Check for 3x3 -> 9x9 pattern
            if h_in == 3 and w_in == 3 and h_out == 9 and w_out == 9:
                # Analyze where the input pattern appears in output
                # Task 007bbfb7: Pattern appears in 3x3 blocks with specific arrangement
                
                # Create expected output
                expected = np.zeros((9, 9), dtype=np.int8)
                
                # Try different distribution patterns
                # Pattern 1: Top-left, top-right, center, bottom-left, bottom-right
                positions = [
                    (0, 0), (0, 6), (3, 3), (6, 0), (6, 6)  # Corners and center
                ]
                
                for y, x in positions:
                    expected[y:y+3, x:x+3] = inp
                
                if not np.array_equal(expected, out):
                    # Try pattern 2: Different arrangement
                    expected = np.zeros((9, 9), dtype=np.int8)
                    
                    # Place pattern in 3x3 grid of blocks
                    for i in range(3):
                        for j in range(3):
                            y_start = i * 3
                            x_start = j * 3
                            
                            # Check various transformations
                            if i == 1 and j == 1:
                                # Center might be different
                                expected[y_start:y_start+3, x_start:x_start+3] = inp
                            elif (i + j) % 2 == 0:
                                expected[y_start:y_start+3, x_start:x_start+3] = inp
                            else:
                                # Try rotated or flipped version
                                if i == 0:
                                    expected[y_start:y_start+3, x_start:x_start+3] = np.rot90(inp, 2)
                                else:
                                    expected[y_start:y_start+3, x_start:x_start+3] = np.fliplr(inp)
                    
                    if np.array_equal(expected, out):
                        # Apply same pattern to test
                        if test_input.shape == (3, 3):
                            result = np.zeros((9, 9), dtype=np.int8)
                            
                            for i in range(3):
                                for j in range(3):
                                    y_start = i * 3
                                    x_start = j * 3
                                    
                                    if i == 1 and j == 1:
                                        result[y_start:y_start+3, x_start:x_start+3] = test_input
                                    elif (i + j) % 2 == 0:
                                        result[y_start:y_start+3, x_start:x_start+3] = test_input
                                    else:
                                        if i == 0:
                                            result[y_start:y_start+3, x_start:x_start+3] = np.rot90(test_input, 2)
                                        else:
                                            result[y_start:y_start+3, x_start:x_start+3] = np.fliplr(test_input)
                            
                            return result
                
                # Try analyzing the actual pattern more carefully
                # Check where non-zero values appear in output vs input
                return self._analyze_distributed_pattern(inp, out, test_input)
        
        return None
    
    def _analyze_distributed_pattern(self,
                                    inp: np.ndarray,
                                    out: np.ndarray,
                                    test_input: np.ndarray) -> Optional[np.ndarray]:
        """Analyze how input is distributed in output for 3x3 -> 9x9 patterns"""
        
        if inp.shape != (3, 3) or out.shape != (9, 9):
            return None
        
        # Initialize result
        result = np.zeros((9, 9), dtype=np.int8)
        
        # Analyze block-by-block (3x3 blocks in 9x9 grid)
        for block_i in range(3):
            for block_j in range(3):
                y_start = block_i * 3
                x_start = block_j * 3
                block = out[y_start:y_start+3, x_start:x_start+3]
                
                # Check if this block matches input or a transformation
                if np.array_equal(block, inp):
                    # Direct copy
                    result[y_start:y_start+3, x_start:x_start+3] = test_input
                elif np.array_equal(block, np.rot90(inp)):
                    # 90 degree rotation
                    result[y_start:y_start+3, x_start:x_start+3] = np.rot90(test_input)
                elif np.array_equal(block, np.rot90(inp, 2)):
                    # 180 degree rotation
                    result[y_start:y_start+3, x_start:x_start+3] = np.rot90(test_input, 2)
                elif np.array_equal(block, np.rot90(inp, 3)):
                    # 270 degree rotation
                    result[y_start:y_start+3, x_start:x_start+3] = np.rot90(test_input, 3)
                elif np.array_equal(block, np.fliplr(inp)):
                    # Horizontal flip
                    result[y_start:y_start+3, x_start:x_start+3] = np.fliplr(test_input)
                elif np.array_equal(block, np.flipud(inp)):
                    # Vertical flip
                    result[y_start:y_start+3, x_start:x_start+3] = np.flipud(test_input)
                elif np.array_equal(block, inp.T):
                    # Transpose
                    result[y_start:y_start+3, x_start:x_start+3] = test_input.T
                elif np.all(block == 0):
                    # Empty block
                    result[y_start:y_start+3, x_start:x_start+3] = 0
                else:
                    # Unknown transformation - try to match pattern
                    # Check if it's a color-swapped version
                    continue
        
        return result
    
    def _solve_scaled_tiling(self,
                           train_inputs: List[np.ndarray],
                           train_outputs: List[np.ndarray],
                           test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve patterns where input is scaled then tiled"""
        
        for inp, out in zip(train_inputs, train_outputs):
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Try different scale factors
            for scale in [2, 3, 4]:
                scaled_h = h_in * scale
                scaled_w = w_in * scale
                
                if h_out % scaled_h == 0 and w_out % scaled_w == 0:
                    # Scale the input
                    scaled = np.repeat(np.repeat(inp, scale, axis=0), scale, axis=1)
                    
                    # Check if output is tiled version of scaled
                    rows = h_out // scaled_h
                    cols = w_out // scaled_w
                    expected = np.tile(scaled, (rows, cols))
                    
                    if np.array_equal(expected, out):
                        # Apply to test
                        test_scaled = np.repeat(np.repeat(test_input, scale, axis=0), scale, axis=1)
                        return np.tile(test_scaled, (rows, cols))
        
        return None
    
    def _solve_mirror_tiling(self,
                           train_inputs: List[np.ndarray],
                           train_outputs: List[np.ndarray],
                           test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve patterns with mirroring and tiling"""
        
        for inp, out in zip(train_inputs, train_outputs):
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            
            # Try horizontal mirroring + tiling
            if w_out == w_in * 2:
                mirrored = np.hstack([inp, np.fliplr(inp)])
                if h_out % h_in == 0:
                    rows = h_out // h_in
                    expected = np.tile(mirrored, (rows, 1))
                    if np.array_equal(expected, out):
                        test_mirrored = np.hstack([test_input, np.fliplr(test_input)])
                        return np.tile(test_mirrored, (rows, 1))
            
            # Try vertical mirroring + tiling
            if h_out == h_in * 2:
                mirrored = np.vstack([inp, np.flipud(inp)])
                if w_out % w_in == 0:
                    cols = w_out // w_in
                    expected = np.tile(mirrored, (1, cols))
                    if np.array_equal(expected, out):
                        test_mirrored = np.vstack([test_input, np.flipud(test_input)])
                        return np.tile(test_mirrored, (1, cols))
        
        return None
    
    def _validate_solution(self,
                         strategy,
                         train_inputs: List[np.ndarray],
                         train_outputs: List[np.ndarray]) -> bool:
        """Validate that strategy works on all training examples"""
        
        for inp, expected_out in zip(train_inputs, train_outputs):
            try:
                # Apply strategy to training input
                result = strategy([inp], [expected_out], inp)
                if result is None or not np.array_equal(result, expected_out):
                    return False
            except:
                return False
        
        return True
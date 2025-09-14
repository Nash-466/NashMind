"""
ARC Pattern Detection Module
Detects patterns, symmetries, and transformations in ARC grids
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy import ndimage
from collections import Counter
from .grid_operations import Grid


class PatternDetector:
    """Detects various patterns and features in ARC grids"""
    
    def __init__(self, grid: Grid):
        self.grid = grid
        self.data = grid.data
    
    # Symmetry Detection
    
    def has_horizontal_symmetry(self) -> bool:
        """Check if grid has horizontal (left-right) symmetry"""
        return np.array_equal(self.data, np.fliplr(self.data))
    
    def has_vertical_symmetry(self) -> bool:
        """Check if grid has vertical (top-bottom) symmetry"""
        return np.array_equal(self.data, np.flipud(self.data))
    
    def has_diagonal_symmetry(self) -> bool:
        """Check if grid has diagonal symmetry (main diagonal)"""
        if self.grid.height != self.grid.width:
            return False
        return np.array_equal(self.data, self.data.T)
    
    def has_anti_diagonal_symmetry(self) -> bool:
        """Check if grid has anti-diagonal symmetry"""
        if self.grid.height != self.grid.width:
            return False
        return np.array_equal(self.data, np.rot90(np.rot90(self.data.T)))
    
    def has_rotational_symmetry(self, order: int = 2) -> bool:
        """Check if grid has rotational symmetry of given order"""
        if self.grid.height != self.grid.width:
            return False
        
        angle = 360 // order
        rotated = self.data.copy()
        
        for _ in range(order - 1):
            rotated = np.rot90(rotated)
            if not np.array_equal(self.data, rotated) and angle * (_ + 1) < 360:
                return False
        
        return True
    
    def get_symmetries(self) -> Dict[str, bool]:
        """Get all symmetries of the grid"""
        return {
            'horizontal': self.has_horizontal_symmetry(),
            'vertical': self.has_vertical_symmetry(),
            'diagonal': self.has_diagonal_symmetry(),
            'anti_diagonal': self.has_anti_diagonal_symmetry(),
            'rotational_90': self.has_rotational_symmetry(4),
            'rotational_180': self.has_rotational_symmetry(2)
        }
    
    # Pattern Detection
    
    def find_repeating_patterns(self, min_size: int = 2, 
                               max_size: Optional[int] = None) -> List[Tuple[Grid, int]]:
        """Find repeating subpatterns in the grid"""
        if max_size is None:
            max_size = min(self.grid.height // 2, self.grid.width // 2)
        
        patterns = {}
        
        for h in range(min_size, min(max_size + 1, self.grid.height + 1)):
            for w in range(min_size, min(max_size + 1, self.grid.width + 1)):
                for y in range(self.grid.height - h + 1):
                    for x in range(self.grid.width - w + 1):
                        subgrid = self.grid.extract_subgrid(y, x, h, w)
                        pattern_key = tuple(subgrid.data.flatten())
                        
                        if pattern_key not in patterns:
                            patterns[pattern_key] = {
                                'grid': subgrid,
                                'count': 0,
                                'positions': []
                            }
                        
                        patterns[pattern_key]['count'] += 1
                        patterns[pattern_key]['positions'].append((y, x))
        
        # Filter patterns that appear more than once
        repeating = []
        for pattern_info in patterns.values():
            if pattern_info['count'] > 1:
                repeating.append((pattern_info['grid'], pattern_info['count']))
        
        # Sort by frequency
        repeating.sort(key=lambda x: x[1], reverse=True)
        
        return repeating
    
    def detect_periodicity(self) -> Dict[str, Optional[int]]:
        """Detect periodic patterns in horizontal and vertical directions"""
        result = {'horizontal': None, 'vertical': None}
        
        # Check horizontal periodicity
        for period in range(2, self.grid.width):
            if self.grid.width % period == 0:
                is_periodic = True
                for row in self.data:
                    for i in range(period, self.grid.width, period):
                        if not np.array_equal(row[:period], row[i:i+period]):
                            is_periodic = False
                            break
                    if not is_periodic:
                        break
                
                if is_periodic:
                    result['horizontal'] = period
                    break
        
        # Check vertical periodicity
        for period in range(2, self.grid.height):
            if self.grid.height % period == 0:
                is_periodic = True
                for col_idx in range(self.grid.width):
                    col = self.data[:, col_idx]
                    for i in range(period, self.grid.height, period):
                        if not np.array_equal(col[:period], col[i:i+period]):
                            is_periodic = False
                            break
                    if not is_periodic:
                        break
                
                if is_periodic:
                    result['vertical'] = period
                    break
        
        return result
    
    # Shape Detection
    
    def find_rectangles(self, color: Optional[int] = None) -> List[Dict]:
        """Find rectangular regions of a specific color or any non-background color"""
        rectangles = []
        colors_to_check = [color] if color is not None else [c for c in self.grid.unique_colors if c != 0]
        
        for check_color in colors_to_check:
            mask = (self.data == check_color).astype(int)
            
            # Use connected components to find separate rectangles
            labeled, num_features = ndimage.label(mask)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                
                # Check if component is rectangular
                bbox = self._get_bbox_from_mask(component)
                if bbox:
                    y_min, x_min, y_max, x_max = bbox
                    rect_region = component[y_min:y_max, x_min:x_max]
                    
                    # A perfect rectangle should be all True in the bbox
                    if np.all(rect_region):
                        rectangles.append({
                            'color': check_color,
                            'position': (y_min, x_min),
                            'height': y_max - y_min,
                            'width': x_max - x_min,
                            'area': (y_max - y_min) * (x_max - x_min)
                        })
        
        return rectangles
    
    def find_lines(self, min_length: int = 3) -> List[Dict]:
        """Find straight lines (horizontal, vertical, diagonal) in the grid"""
        lines = []
        
        for color in self.grid.unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Horizontal lines
            for y in range(self.grid.height):
                x = 0
                while x < self.grid.width:
                    if self.data[y, x] == color:
                        # Found start of potential line
                        length = 1
                        while x + length < self.grid.width and self.data[y, x + length] == color:
                            length += 1
                        
                        if length >= min_length:
                            lines.append({
                                'type': 'horizontal',
                                'color': color,
                                'start': (y, x),
                                'end': (y, x + length - 1),
                                'length': length
                            })
                        
                        x += length
                    else:
                        x += 1
            
            # Vertical lines
            for x in range(self.grid.width):
                y = 0
                while y < self.grid.height:
                    if self.data[y, x] == color:
                        # Found start of potential line
                        length = 1
                        while y + length < self.grid.height and self.data[y + length, x] == color:
                            length += 1
                        
                        if length >= min_length:
                            lines.append({
                                'type': 'vertical',
                                'color': color,
                                'start': (y, x),
                                'end': (y + length - 1, x),
                                'length': length
                            })
                        
                        y += length
                    else:
                        y += 1
            
            # Diagonal lines (top-left to bottom-right)
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    if self.data[y, x] == color:
                        length = 1
                        while (y + length < self.grid.height and 
                               x + length < self.grid.width and 
                               self.data[y + length, x + length] == color):
                            length += 1
                        
                        if length >= min_length:
                            lines.append({
                                'type': 'diagonal',
                                'color': color,
                                'start': (y, x),
                                'end': (y + length - 1, x + length - 1),
                                'length': length
                            })
            
            # Anti-diagonal lines (top-right to bottom-left)
            for y in range(self.grid.height):
                for x in range(self.grid.width):
                    if self.data[y, x] == color:
                        length = 1
                        while (y + length < self.grid.height and 
                               x - length >= 0 and 
                               self.data[y + length, x - length] == color):
                            length += 1
                        
                        if length >= min_length:
                            lines.append({
                                'type': 'anti-diagonal',
                                'color': color,
                                'start': (y, x),
                                'end': (y + length - 1, x - length + 1),
                                'length': length
                            })
        
        return lines
    
    def get_connected_components(self, connectivity: int = 4, include_background: bool = False) -> List[Dict]:
        """Find and analyze connected components"""
        components = []
        
        for color in self.grid.unique_colors:
            if color == 0 and not include_background:  # Skip background unless requested
                continue
            
            mask = (self.data == color).astype(int)
            
            if connectivity == 4:
                structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
            else:  # 8-connectivity
                structure = np.ones((3,3))
            
            labeled, num_features = ndimage.label(mask, structure=structure)
            
            for i in range(1, num_features + 1):
                component_mask = (labeled == i)
                bbox = self._get_bbox_from_mask(component_mask)
                
                if bbox:
                    y_min, x_min, y_max, x_max = bbox
                    area = np.sum(component_mask)
                    
                    # Calculate centroid
                    positions = np.argwhere(component_mask)
                    centroid = np.mean(positions, axis=0)
                    
                    components.append({
                        'color': color,
                        'bbox': bbox,
                        'area': int(area),
                        'centroid': tuple(centroid),
                        'positions': [(int(y), int(x)) for y, x in positions],
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    })
        
        return components
    
    # Transformation Detection
    
    def compare_with(self, other: 'Grid') -> Dict[str, any]:
        """Compare this grid with another to detect transformations"""
        comparison = {
            'same_size': self.grid.shape == other.shape,
            'size_change': None,
            'colors_added': set(),
            'colors_removed': set(),
            'colors_changed': {},
            'is_scaled': False,
            'scale_factor': None,
            'is_rotated': False,
            'rotation_angle': None,
            'is_flipped': False,
            'flip_type': None,
            'pattern_repeated': False
        }
        
        # Check size changes
        if not comparison['same_size']:
            comparison['size_change'] = {
                'height_ratio': other.height / self.grid.height,
                'width_ratio': other.width / self.grid.width
            }
            
            # Check if it's a simple scale
            if (other.height % self.grid.height == 0 and 
                other.width % self.grid.width == 0 and
                other.height // self.grid.height == other.width // self.grid.width):
                
                factor = other.height // self.grid.height
                scaled = self.grid.scale(factor)
                if scaled == other:
                    comparison['is_scaled'] = True
                    comparison['scale_factor'] = factor
        
        # Check color changes
        self_colors = self.grid.unique_colors
        other_colors = other.unique_colors
        
        comparison['colors_added'] = other_colors - self_colors
        comparison['colors_removed'] = self_colors - other_colors
        
        # Check for simple transformations
        if comparison['same_size']:
            # Check rotations
            for angle in [90, 180, 270]:
                if self.grid.rotate(angle) == other:
                    comparison['is_rotated'] = True
                    comparison['rotation_angle'] = angle
                    break
            
            # Check flips
            if self.grid.flip_horizontal() == other:
                comparison['is_flipped'] = True
                comparison['flip_type'] = 'horizontal'
            elif self.grid.flip_vertical() == other:
                comparison['is_flipped'] = True
                comparison['flip_type'] = 'vertical'
        
        # Check for pattern repetition
        if other.height >= self.grid.height and other.width >= self.grid.width:
            if (other.height % self.grid.height == 0 and 
                other.width % self.grid.width == 0):
                
                rows = other.height // self.grid.height
                cols = other.width // self.grid.width
                tiled = self.grid.tile(rows, cols)
                
                if tiled == other:
                    comparison['pattern_repeated'] = True
                    comparison['repetition'] = {'rows': rows, 'cols': cols}
        
        return comparison
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Helper to get bounding box from boolean mask"""
        if not np.any(mask):
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(y_min), int(x_min), int(y_max + 1), int(x_max + 1))
    
    def get_pattern_statistics(self) -> Dict:
        """Get comprehensive statistics about patterns in the grid"""
        stats = {
            'grid_shape': self.grid.shape,
            'unique_colors': len(self.grid.unique_colors),
            'color_distribution': self.grid.count_colors(),
            'symmetries': self.get_symmetries(),
            'periodicity': self.detect_periodicity(),
            'num_rectangles': len(self.find_rectangles()),
            'num_lines': len(self.find_lines()),
            'num_components': len(self.find_connected_components()),
            'dominant_color': None,
            'sparsity': None
        }
        
        # Find dominant color
        color_counts = stats['color_distribution']
        if color_counts:
            stats['dominant_color'] = max(color_counts, key=color_counts.get)
        
        # Calculate sparsity (assuming 0 is background)
        total_cells = self.grid.height * self.grid.width
        non_zero_cells = total_cells - color_counts.get(0, 0)
        stats['sparsity'] = non_zero_cells / total_cells if total_cells > 0 else 0
        
        return stats
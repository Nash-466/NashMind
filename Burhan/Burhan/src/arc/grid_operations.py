"""
ARC Grid Operations Module
Provides comprehensive grid manipulation operations for ARC tasks
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any, Callable
from scipy import ndimage
from collections import Counter
import copy


class Grid:
    """Class representing an ARC grid with manipulation operations"""
    
    def __init__(self, data: np.ndarray):
        """Initialize grid with 2D numpy array"""
        if isinstance(data, list):
            data = np.array(data, dtype=np.int8)
        elif not isinstance(data, np.ndarray):
            raise TypeError("Grid data must be a numpy array or list")
        
        if data.ndim != 2:
            raise ValueError("Grid must be 2-dimensional")
        
        # Ensure valid color range (0-9)
        if not np.all((data >= 0) & (data <= 9)):
            raise ValueError("Grid values must be in range 0-9")
        
        self.data = data.astype(np.int8)
    
    def __repr__(self) -> str:
        return f"Grid(shape={self.shape}, unique_colors={self.unique_colors})"
    
    def __str__(self) -> str:
        """Pretty print the grid"""
        return '\n'.join([' '.join(str(cell) for cell in row) for row in self.data])
    
    def __eq__(self, other) -> bool:
        """Check if two grids are equal"""
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def unique_colors(self) -> Set[int]:
        return set(np.unique(self.data))
    
    def copy(self) -> 'Grid':
        """Create a deep copy of the grid"""
        return Grid(self.data.copy())
    
    # Basic Operations
    
    def rotate(self, degrees: int) -> 'Grid':
        """Rotate grid by specified degrees (90, 180, 270)"""
        if degrees not in [90, 180, 270]:
            raise ValueError("Rotation must be 90, 180, or 270 degrees")
        
        k = degrees // 90
        return Grid(np.rot90(self.data, k))
    
    def flip_horizontal(self) -> 'Grid':
        """Flip grid horizontally (left-right)"""
        return Grid(np.fliplr(self.data))
    
    def flip_vertical(self) -> 'Grid':
        """Flip grid vertically (up-down)"""
        return Grid(np.flipud(self.data))
    
    def transpose(self) -> 'Grid':
        """Transpose the grid (swap rows and columns)"""
        return Grid(self.data.T)
    
    # Advanced Operations
    
    def scale(self, factor: int) -> 'Grid':
        """Scale grid by integer factor (each cell becomes factor x factor cells)"""
        if factor <= 0:
            raise ValueError("Scale factor must be positive")
        
        new_height = self.height * factor
        new_width = self.width * factor
        scaled = np.zeros((new_height, new_width), dtype=np.int8)
        
        for i in range(self.height):
            for j in range(self.width):
                scaled[i*factor:(i+1)*factor, j*factor:(j+1)*factor] = self.data[i, j]
        
        return Grid(scaled)
    
    def tile(self, rows: int, cols: int) -> 'Grid':
        """Tile grid in a rows x cols pattern"""
        if rows <= 0 or cols <= 0:
            raise ValueError("Tile dimensions must be positive")
        
        return Grid(np.tile(self.data, (rows, cols)))
    
    def mirror(self, axis: str = 'horizontal') -> 'Grid':
        """Mirror grid along specified axis"""
        if axis == 'horizontal':
            # Mirror left to right
            mirrored = np.hstack([self.data, np.fliplr(self.data)])
        elif axis == 'vertical':
            # Mirror top to bottom
            mirrored = np.vstack([self.data, np.flipud(self.data)])
        elif axis == 'both':
            # Mirror in all four directions
            top = np.hstack([self.data, np.fliplr(self.data)])
            bottom = np.hstack([np.flipud(self.data), np.flipud(np.fliplr(self.data))])
            mirrored = np.vstack([top, bottom])
        else:
            raise ValueError("Axis must be 'horizontal', 'vertical', or 'both'")
        
        return Grid(mirrored)
    
    def overlay(self, other: 'Grid', x: int = 0, y: int = 0, 
                transparent_color: int = 0) -> 'Grid':
        """Overlay another grid on this grid at position (x, y)"""
        result = self.copy()
        
        # Calculate the valid region for overlay
        y_start = max(0, y)
        y_end = min(self.height, y + other.height)
        x_start = max(0, x)
        x_end = min(self.width, x + other.width)
        
        # Calculate corresponding region in other grid
        other_y_start = max(0, -y)
        other_y_end = other_y_start + (y_end - y_start)
        other_x_start = max(0, -x)
        other_x_end = other_x_start + (x_end - x_start)
        
        # Apply overlay
        for i in range(y_end - y_start):
            for j in range(x_end - x_start):
                color = other.data[other_y_start + i, other_x_start + j]
                if color != transparent_color:
                    result.data[y_start + i, x_start + j] = color
        
        return result
    
    def mask(self, mask_grid: 'Grid', mask_value: int = 0) -> 'Grid':
        """Apply a mask to the grid (keep values where mask is non-zero)"""
        if self.shape != mask_grid.shape:
            raise ValueError("Grid and mask must have the same shape")
        
        result = np.where(mask_grid.data != 0, self.data, mask_value)
        return Grid(result)
    
    # Color Operations
    
    def replace_color(self, old_color: int, new_color: int) -> 'Grid':
        """Replace all occurrences of old_color with new_color"""
        result = self.data.copy()
        result[result == old_color] = new_color
        return Grid(result)
    
    def map_colors(self, color_map: Dict[int, int]) -> 'Grid':
        """Map colors according to the provided dictionary"""
        result = self.data.copy()
        for old_color, new_color in color_map.items():
            result[self.data == old_color] = new_color
        return Grid(result)
    
    def filter_color(self, color: int, background: int = 0) -> 'Grid':
        """Keep only specified color, replace others with background"""
        result = np.where(self.data == color, self.data, background)
        return Grid(result)
    
    def count_colors(self) -> Dict[int, int]:
        """Count occurrences of each color"""
        unique, counts = np.unique(self.data, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_color_positions(self, color: int) -> List[Tuple[int, int]]:
        """Get all positions of a specific color"""
        positions = np.argwhere(self.data == color)
        return [(int(y), int(x)) for y, x in positions]
    
    # Pattern Operations
    
    def extract_subgrid(self, y: int, x: int, height: int, width: int) -> 'Grid':
        """Extract a subgrid from the specified position"""
        if y < 0 or x < 0 or y + height > self.height or x + width > self.width:
            raise ValueError("Subgrid bounds exceed grid dimensions")
        
        return Grid(self.data[y:y+height, x:x+width])
    
    def replace_subgrid(self, subgrid: 'Grid', y: int, x: int) -> 'Grid':
        """Replace a region with the provided subgrid"""
        result = self.copy()
        
        # Calculate the valid region
        y_end = min(self.height, y + subgrid.height)
        x_end = min(self.width, x + subgrid.width)
        
        result.data[y:y_end, x:x_end] = subgrid.data[:y_end-y, :x_end-x]
        return result
    
    def find_pattern(self, pattern: 'Grid') -> List[Tuple[int, int]]:
        """Find all occurrences of a pattern in the grid"""
        positions = []
        p_h, p_w = pattern.shape
        
        if p_h > self.height or p_w > self.width:
            return positions
        
        for y in range(self.height - p_h + 1):
            for x in range(self.width - p_w + 1):
                subgrid = self.data[y:y+p_h, x:x+p_w]
                if np.array_equal(subgrid, pattern.data):
                    positions.append((y, x))
        
        return positions
    
    def get_bounding_box(self, color: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of non-background colors or specific color"""
        if color is not None:
            mask = self.data == color
        else:
            # Assume 0 is background
            mask = self.data != 0
        
        if not np.any(mask):
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(y_min), int(x_min), int(y_max + 1), int(x_max + 1))
    
    def crop_to_content(self, background: int = 0) -> 'Grid':
        """Crop grid to remove background borders"""
        bbox = self.get_bounding_box()
        if bbox is None:
            return self.copy()
        
        y_min, x_min, y_max, x_max = bbox
        return Grid(self.data[y_min:y_max, x_min:x_max])
    
    def pad(self, top: int, bottom: int, left: int, right: int, 
            fill_value: int = 0) -> 'Grid':
        """Pad grid with specified values on each side"""
        padded = np.pad(self.data, 
                       ((top, bottom), (left, right)), 
                       constant_values=fill_value)
        return Grid(padded)
    
    def resize(self, new_height: int, new_width: int, fill_value: int = 0) -> 'Grid':
        """Resize grid to new dimensions, centering content"""
        if new_height <= 0 or new_width <= 0:
            raise ValueError("New dimensions must be positive")
        
        result = np.full((new_height, new_width), fill_value, dtype=np.int8)
        
        # Calculate centering offsets
        y_offset = (new_height - self.height) // 2
        x_offset = (new_width - self.width) // 2
        
        # Calculate valid regions
        src_y_start = max(0, -y_offset)
        src_y_end = min(self.height, new_height - y_offset)
        src_x_start = max(0, -x_offset)
        src_x_end = min(self.width, new_width - x_offset)
        
        dst_y_start = max(0, y_offset)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, x_offset)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            self.data[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return Grid(result)
    
    def apply_function(self, func: Callable[[int], int]) -> 'Grid':
        """Apply a function to each cell value"""
        vectorized = np.vectorize(func)
        result = vectorized(self.data)
        return Grid(result)
    
    def get_connected_components(self, connectivity: int = 4) -> List['Grid']:
        """Get all connected components as separate grids"""
        components = []
        
        for color in self.unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Create binary mask for this color
            mask = (self.data == color).astype(int)
            
            # Label connected components
            if connectivity == 4:
                structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
            else:  # 8-connectivity
                structure = np.ones((3,3))
            
            labeled, num_features = ndimage.label(mask, structure=structure)
            
            # Extract each component
            for i in range(1, num_features + 1):
                component_mask = (labeled == i)
                bbox = self._get_bbox_from_mask(component_mask)
                if bbox:
                    y_min, x_min, y_max, x_max = bbox
                    component_data = np.zeros_like(self.data[y_min:y_max, x_min:x_max])
                    component_data[component_mask[y_min:y_max, x_min:x_max]] = color
                    components.append(Grid(component_data))
        
        return components
    
    def _get_bbox_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Helper to get bounding box from boolean mask"""
        if not np.any(mask):
            return None
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return (int(y_min), int(x_min), int(y_max + 1), int(x_max + 1))
    
    def to_numpy(self) -> np.ndarray:
        """Convert grid to numpy array"""
        return self.data.copy()
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'Grid':
        """Create grid from numpy array"""
        return cls(array)
    
    @classmethod
    def create_empty(cls, height: int, width: int, fill_value: int = 0) -> 'Grid':
        """Create an empty grid with specified dimensions"""
        return cls(np.full((height, width), fill_value, dtype=np.int8))
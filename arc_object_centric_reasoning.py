from __future__ import annotations
import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple
from collections import deque

class Object:
    """     ARC."""
    def __init__(self, object_id: int, pixels: List[Tuple[int, int]], color: int, grid_shape: Tuple[int, int]):
        self.id = object_id
        self.pixels = pixels
        self.color = color
        self.grid_shape = grid_shape
        self.properties = self._calculate_properties()

    def _calculate_properties(self) -> Dict:
        if not self.pixels:
            return {"size": 0, "bbox": (0,0,0,0), "centroid": (0,0), "aspect_ratio": 0}

        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        return {
            "size": len(self.pixels),
            "bbox": (min_r, min_c, max_r, max_c), # (min_row, min_col, max_row, max_col)
            "centroid": (np.mean(rows), np.mean(cols)),
            "aspect_ratio": width / height if height > 0 else 0,
            "color": self.color
        }

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "color": self.color,
            "properties": self.properties
        }

class ObjectCentricReasoning:
    """    ."""
    def __init__(self):
        pass

    def segment_and_analyze(self, grid: np.ndarray) -> List[Object]:
        """
              .
             (BFS/DFS).
        """
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        object_id_counter = 0

        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and not visited[r, c]:
                    object_id_counter += 1
                    current_color = grid[r, c]
                    pixels = []
                    q = deque([(r, c)])
                    visited[r, c] = True

                    while q:
                        curr_r, curr_c = q.popleft()
                        pixels.append((curr_r, curr_c))

                        #  (4-connected)
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == current_color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    objects.append(Object(object_id_counter, pixels, current_color, grid.shape))
        return objects

    def find_object_relations(self, objects: List[Object]) -> List[Dict]:
        """
            .
        """
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j: continue

                # :   ()
                bbox1 = obj1.properties["bbox"]
                bbox2 = obj2.properties["bbox"]

                #    
                h_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                #    
                v_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))

                if h_overlap > 0 and v_overlap > 0: # 
                    relations.append({"type": "overlap", "obj1_id": obj1.id, "obj2_id": obj2.id})
                elif h_overlap > 0 and (abs(bbox1[2] - bbox2[0]) <= 1 or abs(bbox2[2] - bbox1[0]) <= 1): #  
                    relations.append({"type": "vertical_neighbor", "obj1_id": obj1.id, "obj2_id": obj2.id})
                elif v_overlap > 0 and (abs(bbox1[3] - bbox2[1]) <= 1 or abs(bbox2[3] - bbox1[1]) <= 1): #  
                    relations.append({"type": "horizontal_neighbor", "obj1_id": obj1.id, "obj2_id": obj2.id})

        return relations




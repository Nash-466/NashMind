from __future__ import annotations
import numpy as np
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Set, Optional, Union
from collections import defaultdict, deque
import itertools
import json
import math
import copy
from dataclasses import dataclass
from enum import Enum

# Advanced imports for ARC-AGI-2 capabilities
try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Enhanced ARC Hierarchical Reasoning System for ARC Prize 2025
# Designed to handle ARC-AGI-2 challenges with advanced logical reasoning

# ============================================================================
# ADVANCED REASONING ENUMS AND DATA STRUCTURES
# ============================================================================

class ReasoningLevel(Enum):
    """Levels of reasoning abstraction"""
    PERCEPTUAL = 1      # Basic pattern recognition
    CONCEPTUAL = 2      # Object and relationship understanding
    LOGICAL = 3         # Rule-based reasoning
    ABSTRACT = 4        # High-level abstraction
    META = 5           # Meta-reasoning about reasoning

class InferenceType(Enum):
    """Types of logical inference"""
    DEDUCTIVE = "deductive"      # From general to specific
    INDUCTIVE = "inductive"      # From specific to general
    ABDUCTIVE = "abductive"      # Best explanation
    ANALOGICAL = "analogical"    # By analogy
    CAUSAL = "causal"           # Cause-effect reasoning

@dataclass
class ReasoningStep:
    """Represents a single step in reasoning process"""
    level: ReasoningLevel
    inference_type: InferenceType
    premise: Any
    conclusion: Any
    confidence: float
    justification: str

@dataclass
class LogicalRule:
    """Represents a logical rule"""
    condition: Callable[[Any], bool]
    action: Callable[[Any], Any]
    confidence: float
    description: str
    context_sensitive: bool = False

@dataclass
class AbstractConcept:
    """Represents an abstract concept"""
    name: str
    properties: Dict[str, Any]
    relationships: List[str]
    abstraction_level: int
    instances: List[Any]

# ============================================================================
# ADVANCED REASONING COMPONENTS
# ============================================================================

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

                #      


                # "contains", "aligned_horizontally", "aligned_vertically"

        return relations


class UltraAdvancedGridCalculusEngine:
    """    (   CPU)"""
    def analyze_grid_comprehensive(self, grid: np.ndarray) -> Dict:
        #      
        return {
            "shape": grid.shape,
            "unique_colors": len(np.unique(grid)),
            "mean_color": np.mean(grid),
            "std_color": np.std(grid),
            "density": np.sum(grid != 0) / grid.size
        }

class UltraComprehensivePatternAnalyzer:
    """   (   CPU)"""
    def analyze_ultra_comprehensive_patterns(self, grid: np.ndarray) -> Dict:
        #     
        patterns = {
            "horizontal_lines": self._detect_lines(grid, axis=0),
            "vertical_lines": self._detect_lines(grid, axis=1),
            "squares": self._detect_squares(grid)
        }
        return patterns

    def _detect_lines(self, grid: np.ndarray, axis: int) -> int:
        count = 0
        if axis == 0: # Horizontal
            for row in grid:
                if np.all(row == row[0]) and row[0] != 0: # Check for solid color line
                    count += 1
        else: # Vertical
            for col_idx in range(grid.shape[1]):
                col = grid[:, col_idx]
                if np.all(col == col[0]) and col[0] != 0:
                    count += 1
        return count

    def _detect_squares(self, grid: np.ndarray) -> int:
        count = 0
        h, w = grid.shape
        for size in range(2, min(h, w) + 1):
            for r in range(h - size + 1):
                for c in range(w - size + 1):
                    subgrid = grid[r:r+size, c:c+size]
                    if np.all(subgrid == subgrid[0,0]) and subgrid[0,0] != 0:
                        count += 1
        return count

class HierarchicalReasoningEngine:
    """      ARC (CPU-Optimized)"""

    def __init__(self):
        self.calculus_engine = UltraAdvancedGridCalculusEngine()
        self.pattern_analyzer = UltraComprehensivePatternAnalyzer()
        self.object_reasoning = ObjectCentricReasoning() #   
        self.knowledge_graph = defaultdict(set) #      
        self.transformation_rules = {}

    def solve_task(self, task: Dict) -> List[np.ndarray]:
        """
          ARC    .
        Args:
            task (Dict):  ARC   'train'  'test' .
        Returns:
            List[np.ndarray]:     .
        """
        train_examples = task.get('train', [])
        test_examples = task.get("test", [])

        #  1:        
        abstract_representations = []
        for example in train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example["output"])

            input_features = self._extract_level1_features(input_grid)

            output_features = self._extract_level1_features(output_grid)

            input_patterns = self._extract_level2_patterns(input_grid)
            output_patterns = self._extract_level2_patterns(output_grid)

            input_objects = self.object_reasoning.segment_and_analyze(input_grid)
            output_objects = self.object_reasoning.segment_and_analyze(output_grid)

            input_object_relations = self.object_reasoning.find_object_relations(input_objects)
            output_object_relations = self.object_reasoning.find_object_relations(output_objects)

            #     (   )
            self._build_knowledge_graph(input_grid, input_features, input_patterns, input_objects, input_object_relations, "input")
            self._build_knowledge_graph(output_grid, output_features, output_patterns, output_objects, output_object_relations, "output")

            abstract_representations.append({
                "input_grid": input_grid,
                "output_grid": output_grid,
                "input_features": input_features,
                "output_features": output_features,
                "input_patterns": input_patterns,
                "output_patterns": output_patterns,
                "input_objects": [obj.to_dict() for obj in input_objects],
                "output_objects": [obj.to_dict() for obj in output_objects],
                "input_object_relations": input_object_relations,
                "output_object_relations": output_object_relations,
            })

        #  2:     (    )
        inferred_transformations = self._infer_abstract_transformations(abstract_representations)
        self._learn_transformation_rules(inferred_transformations)

        #  3:     
        predictions = []
        for example in test_examples:
            test_input_grid = np.array(example['input'])
            predicted_output = self._apply_rules_to_test(test_input_grid)
            predictions.append(predicted_output)

        return predictions

    def _extract_level1_features(self, grid: np.ndarray) -> Dict:
        """ 1:    (CPU-friendly)"""
        return self.calculus_engine.analyze_grid_comprehensive(grid)

    def _extract_level2_patterns(self, grid: np.ndarray) -> Dict:
        """ 2:     (CPU-friendly)"""
        return self.pattern_analyzer.analyze_ultra_comprehensive_patterns(grid)

    def _build_knowledge_graph(self, grid: np.ndarray, features: Dict, patterns: Dict, objects: List[Object], relations: List[Dict], prefix: str):
        """
              .
        :     .
        :      .
        """
        grid_id = hash(grid.tobytes()) #   

        #    ()
        unique_colors = np.unique(grid)
        for color in unique_colors:
            node_name = f"color_{color}"
            self.knowledge_graph[f"grid_{grid_id}"].add(node_name)
            self.knowledge_graph[node_name].add(f"in_grid_{grid_id}")

        #    
        for k, v in features.items():
            node_name = f"feature_{k}_{int(v)}" if isinstance(v, np.integer) else f"feature_{k}_{v}"
            self.knowledge_graph[f"grid_{grid_id}"].add(node_name)

        for k, v in patterns.items():
            if isinstance(v, (int, float, str, np.integer)): #  
                node_name = f"pattern_{k}_{int(v) if isinstance(v, np.integer) else v}"
                self.knowledge_graph[f"grid_{grid_id}"].add(node_name)
            elif isinstance(v, dict): #   ( \'squares\': count)
                for sub_k, sub_v in v.items():
                    node_name = f"pattern_{k}_{sub_k}_{int(sub_v) if isinstance(sub_v, np.integer) else (float(sub_v) if isinstance(sub_v, np.floating) else sub_v)}"
                    self.knowledge_graph[f"grid_{grid_id}"].add(node_name)

        #      
        for obj in objects:
            obj_node_name = f"object_{obj.id}_grid_{grid_id}"
            self.knowledge_graph[f"grid_{grid_id}"].add(obj_node_name)
            for prop_k, prop_v in obj.properties.items():
                #        
                if isinstance(prop_v, (list, tuple, np.ndarray)):
                    prop_v = str(prop_v)
                elif isinstance(prop_v, np.integer):
                    prop_v = int(prop_v)
                elif isinstance(prop_v, np.floating):
                    prop_v = float(prop_v)
                prop_node_name = f"obj_{obj.id}_prop_{prop_k}_{prop_v}"
                self.knowledge_graph[obj_node_name].add(prop_node_name)

        #      
        for rel in relations:
            rel_node_name = f"relation_{rel['type']}_obj1_{rel['obj1_id']}_obj2_{rel['obj2_id']}_grid_{grid_id}"
            self.knowledge_graph[f"grid_{grid_id}"].add(rel_node_name)
            self.knowledge_graph[f"object_{rel['obj1_id']}_grid_{grid_id}"].add(rel_node_name)
            self.knowledge_graph[f"object_{rel['obj2_id']}_grid_{grid_id}"].add(rel_node_name)

        #     ( :  )
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                current_color = grid[r, c]
                #  
                if r + 1 < h: # 
                    neighbor_color = grid[r+1, c]
                    self.knowledge_graph[f"color_{int(current_color)}"].add(f"adjacent_to_color_{int(neighbor_color)}")
                if c + 1 < w: # 
                    neighbor_color = grid[r, c+1]
                    self.knowledge_graph[f"color_{int(current_color)}"].add(f"adjacent_to_color_{int(neighbor_color)}")

    def _infer_abstract_transformations(self, representations: List[Dict]) -> List[Dict]:
        """ 3:    (CPU-friendly)"""
        inferred = []
        for rep in representations:
            input_grid = rep['input_grid']
            output_grid = rep["output_grid"]
            input_features = rep["input_features"]
            output_features = rep["output_features"]
            input_patterns = rep["input_patterns"]
            output_patterns = rep["output_patterns"]
            input_objects = rep["input_objects"]
            output_objects = rep["output_objects"]
            input_object_relations = rep["input_object_relations"]
            output_object_relations = rep["output_object_relations"]

            transformation = {}

            # 1.      (  )
            if input_grid.shape != output_grid.shape:
                transformation["resize"] = {
                    "from": tuple(int(s) for s in input_grid.shape),
                    "to": tuple(int(s) for s in output_grid.shape)
                }
            if input_grid.shape == output_grid.shape:
                if np.array_equal(np.rot90(input_grid, k=1), output_grid):
                    transformation["rotate"] = 90
                elif np.array_equal(np.rot90(input_grid, k=2), output_grid):
                    transformation["rotate"] = 180
                elif np.array_equal(np.rot90(input_grid, k=3), output_grid):
                    transformation["rotate"] = 270
                elif np.array_equal(np.fliplr(input_grid), output_grid):
                    transformation["flip_horizontal"] = True
                elif np.array_equal(np.flipud(input_grid), output_grid):
                    transformation["flip_vertical"] = True
            input_colors = set(np.unique(input_grid))
            output_colors = set(np.unique(output_grid))
            if input_colors != output_colors:
                transformation["color_change"] = {
                    "added": [int(c) for c in (output_colors - input_colors)],
                    "removed": [int(c) for c in (input_colors - output_colors)]
                }

            # 2.     ( )
            object_transformations = []
            #     
            matched_objects = self._match_objects(input_objects, output_objects)

            for input_obj_dict, output_obj_dict in matched_objects:
                obj_trans = {"input_obj_id": input_obj_dict["id"], "output_obj_id": output_obj_dict["id"]}
                #    
                if input_obj_dict["properties"]["color"] != output_obj_dict["properties"]["color"]:
                    obj_trans["color_change"] = {
                        "from": input_obj_dict["properties"]["color"],
                        "to": output_obj_dict["properties"]["color"]
                    }
                if input_obj_dict["properties"]["size"] != output_obj_dict["properties"]["size"]:
                    obj_trans["size_change"] = {
                        "from": input_obj_dict["properties"]["size"],
                        "to": output_obj_dict["properties"]["size"]
                    }
                #        (bbox, centroid, aspect_ratio)
                object_transformations.append(obj_trans)

            #   
            added_objects = [obj for obj in output_objects if obj["id"] not in [m[1]["id"] for m in matched_objects]]
            removed_objects = [obj for obj in input_objects if obj["id"] not in [m[0]["id"] for m in matched_objects]]

            if added_objects: transformation["added_objects"] = added_objects
            if removed_objects: transformation["removed_objects"] = removed_objects

            transformation["object_transformations"] = object_transformations

            # 3.       ( )
            #       
            #           CPU
            #       
            # if input_object_relations != output_object_relations:
            #     transformation["relation_change"] = True #  

            inferred.append(transformation)
        return inferred

    def _match_objects(self, input_objects: List[Dict], output_objects: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
                 .
        (   CPU-friendly)
        """
        matched = []
        #         
        unmatched_input = list(input_objects)
        unmatched_output = list(output_objects)

        #   :     
        for i_obj in list(unmatched_input): #       
            for o_obj in list(unmatched_output):
                if i_obj["properties"]["color"] == o_obj["properties"]["color"] and \
                   abs(i_obj["properties"]["size"] - o_obj["properties"]["size"]) < 2: #   
                    matched.append((i_obj, o_obj))
                    unmatched_input.remove(i_obj)
                    unmatched_output.remove(o_obj)
                    break #      
        return matched

    def _learn_transformation_rules(self, transformations: List[Dict]):
        """ 4:    (CPU-friendly)"""
        #       .
        #        .
        #        .
        if transformations:
            #  :         
            #          
            #         
            all_network_transforms = []
            all_object_transforms = []
            all_added_objects = []
            all_removed_objects = []

            for t in transformations:
                net_trans = {k: v for k, v in t.items() if k not in ["object_transformations", "added_objects", "removed_objects"]}
                if net_trans: all_network_transforms.append(json.dumps(net_trans, sort_keys=True))
                if "object_transformations" in t: all_object_transforms.extend(t["object_transformations"])
                if "added_objects" in t: all_added_objects.extend(t["added_objects"])
                if "removed_objects" in t: all_removed_objects.extend(t["removed_objects"])

            #       
            if all_network_transforms:
                from collections import Counter
                most_common_net_transform = Counter(all_network_transforms).most_common(1)
                if most_common_net_transform and most_common_net_transform[0][1] == len(transformations): #    
                    self.transformation_rules["main_network_rule"] = json.loads(most_common_net_transform[0][0])

            #    (    )
            #        
            self.transformation_rules["object_rules"] = all_object_transforms
            self.transformation_rules["added_object_patterns"] = all_added_objects
            self.transformation_rules["removed_object_patterns"] = all_removed_objects

    def _apply_rules_to_test(self, input_grid: np.ndarray) -> np.ndarray:
        """       (CPU-friendly)"""
        output_grid = np.copy(input_grid)

        # 1.    
        if "main_network_rule" in self.transformation_rules:
            rule = self.transformation_rules["main_network_rule"]
            if "resize" in rule:
                target_shape = rule["resize"]["to"]
                #    ()
                if target_shape[0] > output_grid.shape[0] or target_shape[1] > output_grid.shape[1]:
                    new_grid = np.zeros(target_shape, dtype=output_grid.dtype)
                    for r_new in range(target_shape[0]):
                        for c_new in range(target_shape[1]):
                            r_old = int(r_new * output_grid.shape[0] / target_shape[0])
                            c_old = int(c_new * output_grid.shape[1] / target_shape[1])
                            new_grid[r_new, c_new] = output_grid[r_old, c_old]
                    output_grid = new_grid
                else:
                    new_grid = np.zeros(target_shape, dtype=output_grid.dtype)
                    for r_old in range(target_shape[0]):
                        for c_old in range(target_shape[1]):
                            r_new = int(r_old * target_shape[0] / output_grid.shape[0])
                            c_new = int(c_old * target_shape[1] / output_grid.shape[1])
                            new_grid[r_new, c_new] = output_grid[r_old, c_old]
                    output_grid = new_grid

            if "rotate" in rule:
                output_grid = np.rot90(output_grid, k=rule["rotate"] // 90)
            if "flip_horizontal" in rule and rule["flip_horizontal"]:
                output_grid = np.fliplr(output_grid)
            if "flip_vertical" in rule and rule["flip_vertical"]:
                output_grid = np.flipud(output_grid)
            # Color change rule needs more sophisticated logic based on object-level changes

        # 2.     ( )
        #          
        #           
        current_objects = self.object_reasoning.segment_and_analyze(output_grid) #     
        transformed_grid = np.zeros_like(output_grid, dtype=output_grid.dtype)

        for obj in current_objects:
            #       
            applied_obj_rule = None
            for obj_rule in self.transformation_rules.get("object_rules", []):
                #   :         
                if obj_rule.get("input_obj_id") is not None and obj_rule.get("output_obj_id") is not None: #    
                    #        
                    #         
                    if "color_change" in obj_rule and obj.color == obj_rule["color_change"]["from"]:
                        applied_obj_rule = obj_rule
                        break

            if applied_obj_rule:
                #   
                if "color_change" in applied_obj_rule:
                    new_color = applied_obj_rule["color_change"]["to"]
                    for r, c in obj.pixels:
                        transformed_grid[r, c] = new_color
                else:
                    #         
                    for r, c in obj.pixels:
                        transformed_grid[r, c] = obj.color
            else:
                #          
                for r, c in obj.pixels:
                    transformed_grid[r, c] = obj.color

        #    (     )
        #          
        #     added_object_patterns

        return transformed_grid

    def _find_paths_in_graph(self, start_node: str, end_node: str, max_depth: int = 5) -> List[List[str]]:
        """
               (   BFS).
           .
        """
        paths = []
        queue = deque([(start_node, [start_node])])
        visited = {start_node}

        while queue:
            current_node, path = queue.popleft()

            if current_node == end_node:
                paths.append(path)
                continue

            if len(path) >= max_depth: #    
                continue

            for neighbor in self.knowledge_graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return paths


# ============================================================================
# ADVANCED LOGICAL REASONING ENGINE FOR ARC-AGI-2
# ============================================================================

class AdvancedLogicalReasoningEngine:
    """Advanced logical reasoning engine capable of handling ARC-AGI-2 challenges"""

    def __init__(self):
        self.reasoning_history: List[ReasoningStep] = []
        self.logical_rules: List[LogicalRule] = []
        self.abstract_concepts: Dict[str, AbstractConcept] = {}
        self.inference_chains: List[List[ReasoningStep]] = []
        self.confidence_threshold = 0.7

        # Initialize with basic logical rules
        self._initialize_basic_logical_rules()
        self._initialize_abstract_concepts()

    def _initialize_basic_logical_rules(self):
        """Initialize basic logical reasoning rules"""

        # Symmetry rules
        self.logical_rules.append(LogicalRule(
            condition=lambda x: self._has_symmetry(x),
            action=lambda x: self._apply_symmetry_transformation(x),
            confidence=0.9,
            description="If pattern has symmetry, apply symmetric transformation",
            context_sensitive=False
        ))

        # Color consistency rules
        self.logical_rules.append(LogicalRule(
            condition=lambda x: self._has_color_pattern(x),
            action=lambda x: self._apply_color_transformation(x),
            confidence=0.8,
            description="If color pattern exists, apply consistent color transformation",
            context_sensitive=True
        ))

        # Size scaling rules
        self.logical_rules.append(LogicalRule(
            condition=lambda x: self._has_size_pattern(x),
            action=lambda x: self._apply_size_transformation(x),
            confidence=0.75,
            description="If size pattern exists, apply scaling transformation",
            context_sensitive=True
        ))

        # Positional rules
        self.logical_rules.append(LogicalRule(
            condition=lambda x: self._has_positional_pattern(x),
            action=lambda x: self._apply_positional_transformation(x),
            confidence=0.8,
            description="If positional pattern exists, apply position-based transformation",
            context_sensitive=True
        ))

    def _initialize_abstract_concepts(self):
        """Initialize abstract concepts for high-level reasoning"""

        # Container concept
        self.abstract_concepts['container'] = AbstractConcept(
            name='container',
            properties={'encloses': True, 'has_boundary': True, 'can_hold_objects': True},
            relationships=['contains', 'surrounds', 'protects'],
            abstraction_level=3,
            instances=[]
        )

        # Path concept
        self.abstract_concepts['path'] = AbstractConcept(
            name='path',
            properties={'connects': True, 'has_direction': True, 'allows_movement': True},
            relationships=['connects', 'leads_to', 'traverses'],
            abstraction_level=3,
            instances=[]
        )

        # Transformer concept
        self.abstract_concepts['transformer'] = AbstractConcept(
            name='transformer',
            properties={'modifies': True, 'has_input': True, 'has_output': True},
            relationships=['transforms', 'processes', 'converts'],
            abstraction_level=4,
            instances=[]
        )

        # Pattern concept
        self.abstract_concepts['pattern'] = AbstractConcept(
            name='pattern',
            properties={'repeats': True, 'has_structure': True, 'predictable': True},
            relationships=['repeats', 'generates', 'follows_rule'],
            abstraction_level=4,
            instances=[]
        )

    def perform_multi_level_reasoning(self, input_data: Any, target_level: ReasoningLevel = ReasoningLevel.META) -> Dict[str, Any]:
        """Perform multi-level reasoning from perceptual to meta-level"""
        try:
            reasoning_results = {}
            current_data = input_data

            # Level 1: Perceptual reasoning
            if target_level.value >= ReasoningLevel.PERCEPTUAL.value:
                perceptual_result = self._perform_perceptual_reasoning(current_data)
                reasoning_results['perceptual'] = perceptual_result
                current_data = perceptual_result

            # Level 2: Conceptual reasoning
            if target_level.value >= ReasoningLevel.CONCEPTUAL.value:
                conceptual_result = self._perform_conceptual_reasoning(current_data)
                reasoning_results['conceptual'] = conceptual_result
                current_data = conceptual_result

            # Level 3: Logical reasoning
            if target_level.value >= ReasoningLevel.LOGICAL.value:
                logical_result = self._perform_logical_reasoning(current_data)
                reasoning_results['logical'] = logical_result
                current_data = logical_result

            # Level 4: Abstract reasoning
            if target_level.value >= ReasoningLevel.ABSTRACT.value:
                abstract_result = self._perform_abstract_reasoning(current_data)
                reasoning_results['abstract'] = abstract_result
                current_data = abstract_result

            # Level 5: Meta-reasoning
            if target_level.value >= ReasoningLevel.META.value:
                meta_result = self._perform_meta_reasoning(reasoning_results)
                reasoning_results['meta'] = meta_result

            return {
                'results': reasoning_results,
                'final_conclusion': self._synthesize_conclusions(reasoning_results),
                'confidence': self._calculate_overall_confidence(reasoning_results),
                'reasoning_chain': self.reasoning_history[-10:]  # Last 10 steps
            }

        except Exception as e:
            return {'error': str(e), 'results': {}}

    def _perform_perceptual_reasoning(self, data: Any) -> Dict[str, Any]:
        """Perform perceptual-level reasoning (basic pattern recognition)"""
        try:
            if isinstance(data, np.ndarray):
                grid = data
            elif isinstance(data, dict) and 'grid' in data:
                grid = data['grid']
            else:
                return {'error': 'Invalid input format'}

            # Basic perceptual analysis
            perceptual_features = {
                'shape': grid.shape,
                'unique_colors': len(np.unique(grid)),
                'density': np.count_nonzero(grid) / grid.size,
                'symmetries': self._detect_symmetries(grid),
                'patterns': self._detect_basic_patterns(grid)
            }

            # Record reasoning step
            step = ReasoningStep(
                level=ReasoningLevel.PERCEPTUAL,
                inference_type=InferenceType.INDUCTIVE,
                premise=f"Grid with shape {grid.shape}",
                conclusion=f"Detected {len(perceptual_features)} perceptual features",
                confidence=0.8,
                justification="Basic pattern recognition from visual features"
            )
            self.reasoning_history.append(step)

            return {
                'features': perceptual_features,
                'grid': grid,
                'confidence': 0.8
            }

        except Exception as e:
            return {'error': str(e)}

    def _perform_conceptual_reasoning(self, perceptual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform conceptual-level reasoning (object and relationship understanding)"""
        try:
            if 'grid' not in perceptual_data:
                return {'error': 'No grid data available'}

            grid = perceptual_data['grid']
            features = perceptual_data.get('features', {})

            # Object detection and analysis
            objects = self._detect_objects(grid)
            relationships = self._analyze_object_relationships(objects, grid)

            # Conceptual understanding
            concepts = []
            for obj in objects:
                concept = self._map_object_to_concept(obj, grid)
                if concept:
                    concepts.append(concept)

            # Record reasoning step
            step = ReasoningStep(
                level=ReasoningLevel.CONCEPTUAL,
                inference_type=InferenceType.ABDUCTIVE,
                premise=f"Objects: {len(objects)}, Relationships: {len(relationships)}",
                conclusion=f"Identified {len(concepts)} conceptual mappings",
                confidence=0.75,
                justification="Object-relationship analysis and concept mapping"
            )
            self.reasoning_history.append(step)

            return {
                'objects': objects,
                'relationships': relationships,
                'concepts': concepts,
                'grid': grid,
                'confidence': 0.75
            }

        except Exception as e:
            return {'error': str(e)}

    def _perform_logical_reasoning(self, conceptual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform logical-level reasoning (rule-based reasoning)"""
        try:
            if 'objects' not in conceptual_data:
                return {'error': 'No conceptual data available'}

            objects = conceptual_data['objects']
            relationships = conceptual_data.get('relationships', [])
            concepts = conceptual_data.get('concepts', [])
            grid = conceptual_data['grid']

            # Apply logical rules
            applicable_rules = []
            rule_results = []

            for rule in self.logical_rules:
                if rule.condition(conceptual_data):
                    applicable_rules.append(rule)
                    result = rule.action(conceptual_data)
                    rule_results.append({
                        'rule': rule.description,
                        'result': result,
                        'confidence': rule.confidence
                    })

            # Logical inference
            inferences = self._perform_logical_inference(objects, relationships, concepts)

            # Record reasoning step
            step = ReasoningStep(
                level=ReasoningLevel.LOGICAL,
                inference_type=InferenceType.DEDUCTIVE,
                premise=f"Applied {len(applicable_rules)} logical rules",
                conclusion=f"Generated {len(inferences)} logical inferences",
                confidence=0.8,
                justification="Rule-based logical reasoning and inference"
            )
            self.reasoning_history.append(step)

            return {
                'applicable_rules': applicable_rules,
                'rule_results': rule_results,
                'inferences': inferences,
                'grid': grid,
                'confidence': 0.8
            }

        except Exception as e:
            return {'error': str(e)}

    def _perform_abstract_reasoning(self, logical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform abstract-level reasoning (high-level abstraction)"""
        try:
            if 'inferences' not in logical_data:
                return {'error': 'No logical data available'}

            inferences = logical_data['inferences']
            rule_results = logical_data.get('rule_results', [])
            grid = logical_data['grid']

            # Abstract pattern extraction
            abstract_patterns = self._extract_abstract_patterns(inferences, rule_results)

            # High-level transformations
            transformations = self._identify_abstract_transformations(abstract_patterns, grid)

            # Generalization
            generalizations = self._perform_generalization(abstract_patterns, transformations)

            # Record reasoning step
            step = ReasoningStep(
                level=ReasoningLevel.ABSTRACT,
                inference_type=InferenceType.ANALOGICAL,
                premise=f"Abstract patterns: {len(abstract_patterns)}",
                conclusion=f"Generated {len(generalizations)} generalizations",
                confidence=0.7,
                justification="High-level abstraction and generalization"
            )
            self.reasoning_history.append(step)

            return {
                'abstract_patterns': abstract_patterns,
                'transformations': transformations,
                'generalizations': generalizations,
                'grid': grid,
                'confidence': 0.7
            }

        except Exception as e:
            return {'error': str(e)}

    def _perform_meta_reasoning(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-level reasoning (reasoning about reasoning)"""
        try:
            # Analyze reasoning quality
            reasoning_quality = self._analyze_reasoning_quality(all_results)

            # Identify reasoning gaps
            gaps = self._identify_reasoning_gaps(all_results)

            # Suggest improvements
            improvements = self._suggest_reasoning_improvements(gaps, reasoning_quality)

            # Meta-level insights
            insights = self._generate_meta_insights(all_results, reasoning_quality)

            # Record reasoning step
            step = ReasoningStep(
                level=ReasoningLevel.META,
                inference_type=InferenceType.ABDUCTIVE,
                premise="Analysis of reasoning process",
                conclusion=f"Generated {len(insights)} meta-insights",
                confidence=0.6,
                justification="Meta-reasoning about the reasoning process"
            )
            self.reasoning_history.append(step)

            return {
                'reasoning_quality': reasoning_quality,
                'gaps': gaps,
                'improvements': improvements,
                'insights': insights,
                'confidence': 0.6
            }

        except Exception as e:
            return {'error': str(e)}

    def _synthesize_conclusions(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize conclusions from all reasoning levels"""
        try:
            conclusions = {}

            # Extract key findings from each level
            if 'perceptual' in reasoning_results:
                conclusions['perceptual_findings'] = reasoning_results['perceptual'].get('features', {})

            if 'conceptual' in reasoning_results:
                conclusions['conceptual_findings'] = {
                    'objects': len(reasoning_results['conceptual'].get('objects', [])),
                    'relationships': len(reasoning_results['conceptual'].get('relationships', [])),
                    'concepts': len(reasoning_results['conceptual'].get('concepts', []))
                }

            if 'logical' in reasoning_results:
                conclusions['logical_findings'] = {
                    'applicable_rules': len(reasoning_results['logical'].get('applicable_rules', [])),
                    'inferences': len(reasoning_results['logical'].get('inferences', []))
                }

            if 'abstract' in reasoning_results:
                conclusions['abstract_findings'] = {
                    'patterns': len(reasoning_results['abstract'].get('abstract_patterns', [])),
                    'transformations': len(reasoning_results['abstract'].get('transformations', [])),
                    'generalizations': len(reasoning_results['abstract'].get('generalizations', []))
                }

            if 'meta' in reasoning_results:
                conclusions['meta_findings'] = reasoning_results['meta'].get('insights', [])

            # Overall assessment
            conclusions['overall_assessment'] = self._generate_overall_assessment(conclusions)

            return conclusions

        except Exception as e:
            return {'error': str(e)}

    def _calculate_overall_confidence(self, reasoning_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in reasoning results"""
        try:
            confidences = []

            for level_name, level_data in reasoning_results.items():
                if isinstance(level_data, dict) and 'confidence' in level_data:
                    confidences.append(level_data['confidence'])

            if not confidences:
                return 0.0

            # Weighted average with higher weight for logical and abstract levels
            weights = {
                'perceptual': 0.1,
                'conceptual': 0.2,
                'logical': 0.3,
                'abstract': 0.3,
                'meta': 0.1
            }

            weighted_sum = 0.0
            total_weight = 0.0

            for i, (level_name, level_data) in enumerate(reasoning_results.items()):
                if isinstance(level_data, dict) and 'confidence' in level_data:
                    weight = weights.get(level_name, 0.2)
                    weighted_sum += level_data['confidence'] * weight
                    total_weight += weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        except Exception:
            return 0.0

    # ============================================================================
    # HELPER METHODS FOR ADVANCED REASONING
    # ============================================================================

    def _has_symmetry(self, data: Any) -> bool:
        """Check if data has symmetry patterns"""
        try:
            if isinstance(data, dict) and 'grid' in data:
                grid = data['grid']
                return (np.array_equal(grid, np.fliplr(grid)) or
                       np.array_equal(grid, np.flipud(grid)) or
                       np.array_equal(grid, np.rot90(grid, 2)))
            return False
        except Exception:
            return False

    def _has_color_pattern(self, data: Any) -> bool:
        """Check if data has color patterns"""
        try:
            if isinstance(data, dict) and 'grid' in data:
                grid = data['grid']
                unique_colors = len(np.unique(grid))
                return unique_colors > 2 and unique_colors <= 8
            return False
        except Exception:
            return False

    def _has_size_pattern(self, data: Any) -> bool:
        """Check if data has size patterns"""
        try:
            if isinstance(data, dict) and 'objects' in data:
                objects = data['objects']
                if len(objects) > 1:
                    sizes = [obj.properties['size'] for obj in objects]
                    return len(set(sizes)) > 1  # Different sizes exist
            return False
        except Exception:
            return False

    def _has_positional_pattern(self, data: Any) -> bool:
        """Check if data has positional patterns"""
        try:
            if isinstance(data, dict) and 'objects' in data:
                objects = data['objects']
                if len(objects) > 1:
                    centroids = [obj.properties['centroid'] for obj in objects]
                    # Check for alignment or regular spacing
                    return self._check_positional_regularity(centroids)
            return False
        except Exception:
            return False

    def _check_positional_regularity(self, centroids: List[Tuple[float, float]]) -> bool:
        """Check if centroids show regular positioning"""
        if len(centroids) < 2:
            return False

        # Check for horizontal or vertical alignment
        rows = [c[0] for c in centroids]
        cols = [c[1] for c in centroids]

        row_variance = np.var(rows)
        col_variance = np.var(cols)

        # Low variance indicates alignment
        return row_variance < 1.0 or col_variance < 1.0

    def _apply_symmetry_transformation(self, data: Any) -> Dict[str, Any]:
        """Apply symmetry-based transformation"""
        try:
            if isinstance(data, dict) and 'grid' in data:
                grid = data['grid']

                # Determine symmetry type and apply transformation
                if np.array_equal(grid, np.fliplr(grid)):
                    return {'transformation': 'horizontal_symmetry', 'result': np.fliplr(grid)}
                elif np.array_equal(grid, np.flipud(grid)):
                    return {'transformation': 'vertical_symmetry', 'result': np.flipud(grid)}
                elif np.array_equal(grid, np.rot90(grid, 2)):
                    return {'transformation': 'rotational_symmetry', 'result': np.rot90(grid, 2)}

            return {'transformation': 'none', 'result': None}
        except Exception:
            return {'transformation': 'error', 'result': None}

    def _apply_color_transformation(self, data: Any) -> Dict[str, Any]:
        """Apply color-based transformation"""
        try:
            if isinstance(data, dict) and 'grid' in data:
                grid = data['grid']
                unique_colors = np.unique(grid)

                # Simple color mapping transformation
                color_map = {color: (color + 1) % 10 for color in unique_colors}
                transformed_grid = np.copy(grid)

                for old_color, new_color in color_map.items():
                    transformed_grid[grid == old_color] = new_color

                return {'transformation': 'color_mapping', 'result': transformed_grid, 'mapping': color_map}

            return {'transformation': 'none', 'result': None}
        except Exception:
            return {'transformation': 'error', 'result': None}

    def _apply_size_transformation(self, data: Any) -> Dict[str, Any]:
        """Apply size-based transformation"""
        try:
            if isinstance(data, dict) and 'objects' in data:
                objects = data['objects']

                # Analyze size patterns and suggest transformations
                sizes = [obj.properties['size'] for obj in objects]
                size_pattern = self._analyze_size_pattern(sizes)

                return {'transformation': 'size_scaling', 'pattern': size_pattern}

            return {'transformation': 'none', 'result': None}
        except Exception:
            return {'transformation': 'error', 'result': None}

    def _apply_positional_transformation(self, data: Any) -> Dict[str, Any]:
        """Apply position-based transformation"""
        try:
            if isinstance(data, dict) and 'objects' in data:
                objects = data['objects']
                centroids = [obj.properties['centroid'] for obj in objects]

                # Analyze positional pattern
                position_pattern = self._analyze_positional_pattern(centroids)

                return {'transformation': 'positional_shift', 'pattern': position_pattern}

            return {'transformation': 'none', 'result': None}
        except Exception:
            return {'transformation': 'error', 'result': None}

    def _detect_symmetries(self, grid: np.ndarray) -> Dict[str, bool]:
        """Detect various types of symmetries"""
        return {
            'horizontal': np.array_equal(grid, np.fliplr(grid)),
            'vertical': np.array_equal(grid, np.flipud(grid)),
            'rotational_90': np.array_equal(grid, np.rot90(grid)),
            'rotational_180': np.array_equal(grid, np.rot90(grid, 2)),
            'rotational_270': np.array_equal(grid, np.rot90(grid, 3))
        }

    def _detect_basic_patterns(self, grid: np.ndarray) -> List[str]:
        """Detect basic patterns in the grid"""
        patterns = []

        # Repetition patterns
        if self._has_repetition_pattern(grid):
            patterns.append('repetition')

        # Gradient patterns
        if self._has_gradient_pattern(grid):
            patterns.append('gradient')

        # Checkerboard patterns
        if self._has_checkerboard_pattern(grid):
            patterns.append('checkerboard')

        # Border patterns
        if self._has_border_pattern(grid):
            patterns.append('border')

        return patterns

    def _detect_objects(self, grid: np.ndarray) -> List[Object]:
        """Detect objects in the grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        object_id = 0

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    # Found new object, perform flood fill
                    pixels = []
                    color = grid[i, j]
                    self._flood_fill(grid, visited, i, j, color, pixels)

                    if pixels:
                        obj = Object(object_id, pixels, color, grid.shape)
                        objects.append(obj)
                        object_id += 1

        return objects

    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, i: int, j: int, color: int, pixels: List[Tuple[int, int]]):
        """Flood fill algorithm for object detection"""
        if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or
            visited[i, j] or grid[i, j] != color):
            return

        visited[i, j] = True
        pixels.append((i, j))

        # Check 4-connected neighbors
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            self._flood_fill(grid, visited, i + di, j + dj, color, pixels)

    def _analyze_object_relationships(self, objects: List[Object], grid: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze relationships between objects"""
        relationships = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                relationship = self._analyze_object_pair(obj1, obj2, grid)
                if relationship['strength'] > 0.3:
                    relationships.append(relationship)

        return relationships

    def _analyze_object_pair(self, obj1: Object, obj2: Object, grid: np.ndarray) -> Dict[str, Any]:
        """Analyze relationship between two objects"""
        # Calculate distance
        c1 = obj1.properties['centroid']
        c2 = obj2.properties['centroid']
        distance = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

        # Analyze spatial relationship
        spatial_rel = self._classify_spatial_relationship(c1, c2, distance)

        # Analyze size relationship
        size_rel = self._classify_size_relationship(obj1.properties['size'], obj2.properties['size'])

        # Analyze color relationship
        color_rel = self._classify_color_relationship(obj1.color, obj2.color)

        # Calculate overall relationship strength
        strength = self._calculate_relationship_strength(spatial_rel, size_rel, color_rel, distance)

        return {
            'object1_id': obj1.id,
            'object2_id': obj2.id,
            'spatial': spatial_rel,
            'size': size_rel,
            'color': color_rel,
            'distance': distance,
            'strength': strength
        }

    def _map_object_to_concept(self, obj: Object, grid: np.ndarray) -> Optional[str]:
        """Map object to abstract concept"""
        size = obj.properties['size']
        bbox = obj.properties['bbox']

        # Container concept
        if self._is_container_like(obj, grid):
            return 'container'

        # Path concept
        elif self._is_path_like(obj, grid):
            return 'path'

        # Point concept
        elif size == 1:
            return 'point'

        # Line concept
        elif self._is_line_like(obj):
            return 'line'

        # Block concept
        elif self._is_block_like(obj):
            return 'block'

        return None

    # Additional helper methods for pattern analysis
    def _has_repetition_pattern(self, grid: np.ndarray) -> bool:
        """Check for repetition patterns"""
        h, w = grid.shape

        # Check for horizontal repetition
        if w >= 4:
            for period in range(1, w//2 + 1):
                if all(np.array_equal(grid[:, i:i+period], grid[:, i+period:i+2*period])
                      for i in range(0, w-2*period, period)):
                    return True

        # Check for vertical repetition
        if h >= 4:
            for period in range(1, h//2 + 1):
                if all(np.array_equal(grid[i:i+period, :], grid[i+period:i+2*period, :])
                      for i in range(0, h-2*period, period)):
                    return True

        return False

    def _has_gradient_pattern(self, grid: np.ndarray) -> bool:
        """Check for gradient patterns"""
        # Simple gradient detection
        unique_vals = np.unique(grid)
        if len(unique_vals) < 3:
            return False

        # Check for monotonic increase/decrease
        for row in grid:
            if len(np.unique(row)) > 1:
                diffs = np.diff(row)
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    return True

        for col in grid.T:
            if len(np.unique(col)) > 1:
                diffs = np.diff(col)
                if np.all(diffs >= 0) or np.all(diffs <= 0):
                    return True

        return False

    def _has_checkerboard_pattern(self, grid: np.ndarray) -> bool:
        """Check for checkerboard patterns"""
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return False

        # Check if alternating pattern exists
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                if (grid[i, j] == grid[i+1, j+1] and
                    grid[i, j+1] == grid[i+1, j] and
                    grid[i, j] != grid[i, j+1]):
                    return True

        return False

    def _has_border_pattern(self, grid: np.ndarray) -> bool:
        """Check for border patterns"""
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return False

        # Check if border is different from interior
        border_vals = set()
        interior_vals = set()

        # Collect border values
        border_vals.update(grid[0, :])  # Top
        border_vals.update(grid[-1, :])  # Bottom
        border_vals.update(grid[:, 0])  # Left
        border_vals.update(grid[:, -1])  # Right

        # Collect interior values
        interior_vals.update(grid[1:-1, 1:-1].flatten())

        return len(border_vals.intersection(interior_vals)) < min(len(border_vals), len(interior_vals))

    def _classify_spatial_relationship(self, c1: Tuple[float, float], c2: Tuple[float, float], distance: float) -> str:
        """Classify spatial relationship between two centroids"""
        if distance < 2:
            return 'adjacent'
        elif distance < 5:
            return 'nearby'
        else:
            return 'distant'

    def _classify_size_relationship(self, size1: int, size2: int) -> str:
        """Classify size relationship between two objects"""
        ratio = size1 / size2 if size2 > 0 else float('inf')

        if 0.8 <= ratio <= 1.2:
            return 'similar'
        elif ratio > 2:
            return 'much_larger'
        elif ratio > 1.2:
            return 'larger'
        elif ratio < 0.5:
            return 'much_smaller'
        else:
            return 'smaller'

    def _classify_color_relationship(self, color1: int, color2: int) -> str:
        """Classify color relationship between two objects"""
        if color1 == color2:
            return 'same'
        else:
            return 'different'

    def _calculate_relationship_strength(self, spatial_rel: str, size_rel: str, color_rel: str, distance: float) -> float:
        """Calculate overall relationship strength"""
        strength = 0.5  # Base strength

        # Adjust based on spatial relationship
        if spatial_rel == 'adjacent':
            strength += 0.3
        elif spatial_rel == 'nearby':
            strength += 0.1

        # Adjust based on size relationship
        if size_rel == 'similar':
            strength += 0.1

        # Adjust based on color relationship
        if color_rel == 'same':
            strength += 0.2

        # Adjust based on distance
        strength -= min(distance * 0.05, 0.3)

        return max(0.0, min(1.0, strength))

    def _is_container_like(self, obj: Object, grid: np.ndarray) -> bool:
        """Check if object is container-like"""
        bbox = obj.properties['bbox']
        min_r, min_c, max_r, max_c = bbox

        # Check if object forms a boundary around empty space
        if max_r - min_r < 2 or max_c - min_c < 2:
            return False

        # Check interior
        interior = grid[min_r+1:max_r, min_c+1:max_c]
        return np.any(interior == 0)  # Has empty interior

    def _is_path_like(self, obj: Object, grid: np.ndarray) -> bool:
        """Check if object is path-like"""
        bbox = obj.properties['bbox']
        min_r, min_c, max_r, max_c = bbox

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # Path-like if one dimension is much larger than the other
        return (height == 1 and width > 2) or (width == 1 and height > 2)

    def _is_line_like(self, obj: Object) -> bool:
        """Check if object is line-like"""
        bbox = obj.properties['bbox']
        min_r, min_c, max_r, max_c = bbox

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        return height == 1 or width == 1

    def _is_block_like(self, obj: Object) -> bool:
        """Check if object is block-like"""
        size = obj.properties['size']
        bbox = obj.properties['bbox']
        min_r, min_c, max_r, max_c = bbox

        height = max_r - min_r + 1
        width = max_c - min_c + 1
        expected_size = height * width

        # Block-like if fills most of its bounding box
        return size / expected_size > 0.7

    def _perform_logical_inference(self, objects: List[Object], relationships: List[Dict[str, Any]], concepts: List[str]) -> List[Dict[str, Any]]:
        """Perform logical inference based on objects, relationships, and concepts"""
        inferences = []

        # Inference 1: If objects are similar and adjacent, they might be part of a pattern
        for rel in relationships:
            if rel['spatial'] == 'adjacent' and rel['size'] == 'similar':
                inferences.append({
                    'type': 'pattern_continuation',
                    'premise': f"Objects {rel['object1_id']} and {rel['object2_id']} are similar and adjacent",
                    'conclusion': 'They likely form part of a repeating pattern',
                    'confidence': 0.7
                })

        # Inference 2: If there's a container concept, other objects might be contained
        if 'container' in concepts:
            inferences.append({
                'type': 'containment',
                'premise': 'Container concept detected',
                'conclusion': 'Some objects may be contained within others',
                'confidence': 0.6
            })

        # Inference 3: If objects have size progression, predict next size
        sizes = [obj.properties['size'] for obj in objects]
        if len(sizes) >= 3:
            diffs = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
            if len(set(diffs)) == 1:  # Constant difference
                next_size = sizes[-1] + diffs[0]
                inferences.append({
                    'type': 'size_progression',
                    'premise': f'Size progression detected: {sizes}',
                    'conclusion': f'Next size would be: {next_size}',
                    'confidence': 0.8
                })

        return inferences

    def _extract_abstract_patterns(self, inferences: List[Dict[str, Any]], rule_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract abstract patterns from inferences and rule results"""
        patterns = []

        # Pattern from inferences
        inference_types = [inf['type'] for inf in inferences]
        if 'pattern_continuation' in inference_types:
            patterns.append({
                'type': 'repetition_pattern',
                'description': 'Elements tend to repeat in regular patterns',
                'confidence': 0.7
            })

        if 'size_progression' in inference_types:
            patterns.append({
                'type': 'progression_pattern',
                'description': 'Elements follow arithmetic progression',
                'confidence': 0.8
            })

        # Pattern from rule results
        transformations = [result.get('transformation', '') for result in rule_results]
        if 'symmetry' in ' '.join(transformations):
            patterns.append({
                'type': 'symmetry_pattern',
                'description': 'System exhibits symmetrical properties',
                'confidence': 0.9
            })

        return patterns

    def _identify_abstract_transformations(self, patterns: List[Dict[str, Any]], grid: np.ndarray) -> List[Dict[str, Any]]:
        """Identify abstract transformations based on patterns"""
        transformations = []

        for pattern in patterns:
            if pattern['type'] == 'repetition_pattern':
                transformations.append({
                    'type': 'extend_pattern',
                    'description': 'Extend the repeating pattern',
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'progression_pattern':
                transformations.append({
                    'type': 'continue_progression',
                    'description': 'Continue the arithmetic progression',
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'symmetry_pattern':
                transformations.append({
                    'type': 'apply_symmetry',
                    'description': 'Apply symmetrical transformation',
                    'confidence': pattern['confidence']
                })

        return transformations

    def _perform_generalization(self, patterns: List[Dict[str, Any]], transformations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform generalization from patterns and transformations"""
        generalizations = []

        # Generalize from patterns
        pattern_types = [p['type'] for p in patterns]
        if len(set(pattern_types)) > 1:
            generalizations.append({
                'type': 'multi_pattern_system',
                'description': 'System exhibits multiple types of patterns simultaneously',
                'confidence': 0.6
            })

        # Generalize from transformations
        if len(transformations) > 1:
            generalizations.append({
                'type': 'complex_transformation',
                'description': 'Multiple transformations may need to be applied',
                'confidence': 0.7
            })

        return generalizations

    def _analyze_reasoning_quality(self, all_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the quality of reasoning at each level"""
        quality = {}

        for level_name, level_data in all_results.items():
            if isinstance(level_data, dict) and 'confidence' in level_data:
                # Quality based on confidence and completeness
                confidence = level_data['confidence']
                completeness = self._assess_completeness(level_data)
                quality[level_name] = (confidence + completeness) / 2

        return quality

    def _assess_completeness(self, level_data: Dict[str, Any]) -> float:
        """Assess completeness of reasoning at a level"""
        # Simple heuristic based on number of results
        result_counts = []

        for key, value in level_data.items():
            if isinstance(value, list):
                result_counts.append(len(value))
            elif isinstance(value, dict):
                result_counts.append(len(value))

        if not result_counts:
            return 0.0

        # Normalize to 0-1 scale
        avg_count = np.mean(result_counts)
        return min(avg_count / 5.0, 1.0)  # Assume 5 results is "complete"

    def _identify_reasoning_gaps(self, all_results: Dict[str, Any]) -> List[str]:
        """Identify gaps in reasoning"""
        gaps = []

        # Check for missing levels
        expected_levels = ['perceptual', 'conceptual', 'logical', 'abstract']
        for level in expected_levels:
            if level not in all_results:
                gaps.append(f"Missing {level} level reasoning")

        # Check for low confidence areas
        for level_name, level_data in all_results.items():
            if isinstance(level_data, dict) and 'confidence' in level_data:
                if level_data['confidence'] < 0.5:
                    gaps.append(f"Low confidence in {level_name} reasoning")

        return gaps

    def _suggest_reasoning_improvements(self, gaps: List[str], quality: Dict[str, float]) -> List[str]:
        """Suggest improvements to reasoning"""
        improvements = []

        for gap in gaps:
            if "Missing" in gap:
                improvements.append(f"Add {gap.split()[1]} level analysis")
            elif "Low confidence" in gap:
                level = gap.split()[-2]
                improvements.append(f"Gather more evidence for {level} reasoning")

        # Suggest improvements for low-quality reasoning
        for level, qual in quality.items():
            if qual < 0.6:
                improvements.append(f"Improve {level} reasoning with additional analysis")

        return improvements

    def _generate_meta_insights(self, all_results: Dict[str, Any], quality: Dict[str, float]) -> List[str]:
        """Generate meta-level insights about the reasoning process"""
        insights = []

        # Insight about reasoning depth
        num_levels = len(all_results)
        if num_levels >= 4:
            insights.append("Deep multi-level reasoning was performed")
        else:
            insights.append("Reasoning could benefit from deeper analysis")

        # Insight about reasoning quality
        avg_quality = np.mean(list(quality.values())) if quality else 0.0
        if avg_quality > 0.7:
            insights.append("High-quality reasoning achieved across levels")
        elif avg_quality > 0.5:
            insights.append("Moderate-quality reasoning with room for improvement")
        else:
            insights.append("Reasoning quality needs significant improvement")

        # Insight about reasoning consistency
        quality_variance = np.var(list(quality.values())) if len(quality) > 1 else 0.0
        if quality_variance < 0.1:
            insights.append("Consistent reasoning quality across levels")
        else:
            insights.append("Inconsistent reasoning quality across levels")

        return insights

    def _generate_overall_assessment(self, conclusions: Dict[str, Any]) -> str:
        """Generate overall assessment of the reasoning process"""
        assessments = []

        # Count findings at each level
        total_findings = 0
        for key, value in conclusions.items():
            if isinstance(value, dict):
                total_findings += sum(v for v in value.values() if isinstance(v, int))
            elif isinstance(value, list):
                total_findings += len(value)

        if total_findings > 10:
            assessments.append("Rich analysis with many findings")
        elif total_findings > 5:
            assessments.append("Moderate analysis with several findings")
        else:
            assessments.append("Limited analysis with few findings")

        # Check for meta-findings
        if 'meta_findings' in conclusions and conclusions['meta_findings']:
            assessments.append("Meta-level insights generated")

        return "; ".join(assessments)

    def _analyze_size_pattern(self, sizes: List[int]) -> Dict[str, Any]:
        """Analyze pattern in object sizes"""
        if len(sizes) < 2:
            return {'type': 'insufficient_data'}

        # Check for arithmetic progression
        diffs = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
        if len(set(diffs)) == 1:
            return {'type': 'arithmetic_progression', 'difference': diffs[0]}

        # Check for geometric progression
        if all(s > 0 for s in sizes):
            ratios = [sizes[i+1] / sizes[i] for i in range(len(sizes)-1)]
            if all(abs(r - ratios[0]) < 0.1 for r in ratios):
                return {'type': 'geometric_progression', 'ratio': ratios[0]}

        return {'type': 'irregular'}

    def _analyze_positional_pattern(self, centroids: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze pattern in object positions"""
        if len(centroids) < 2:
            return {'type': 'insufficient_data'}

        # Check for linear arrangement
        if len(centroids) >= 3:
            # Calculate if points are collinear
            x_coords = [c[1] for c in centroids]
            y_coords = [c[0] for c in centroids]

            # Check horizontal alignment
            if all(abs(y - y_coords[0]) < 1.0 for y in y_coords):
                return {'type': 'horizontal_line', 'y_position': y_coords[0]}

            # Check vertical alignment
            if all(abs(x - x_coords[0]) < 1.0 for x in x_coords):
                return {'type': 'vertical_line', 'x_position': x_coords[0]}

        # Check for regular spacing
        distances = []
        for i in range(len(centroids) - 1):
            c1, c2 = centroids[i], centroids[i+1]
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            distances.append(dist)

        if len(set(round(d, 1) for d in distances)) == 1:
            return {'type': 'regular_spacing', 'distance': distances[0]}

        return {'type': 'irregular'}


#    (     )
if __name__ == "__main__":
    #  ARC 
    dummy_task = {
        "train": [
            {
                "input": [[1,1,1],[1,0,1],[1,1,1]],
                "output": [[1,1,1],[1,1,1],[1,1,1]]
            },
            {
                "input": [[2,2],[2,0]],
                "output": [[2,2],[2,2]]
            }
        ],
        "test": [
            {
                "input": [[3,3,3],[3,0,3],[3,3,3]]
            }
        ]
    }

    engine = HierarchicalReasoningEngine()
    predictions = engine.solve_task(dummy_task)

    print("Predictions for test tasks:")
    for pred in predictions:
        print(pred)

    #     
    # engine._build_knowledge_graph(np.array(dummy_task["train"][0]["input"]), {}, {}, "input_example_0")
    # print("\nKnowledge Graph (simplified view):")
    # for node, neighbors in engine.knowledge_graph.items():
    #     print(f"{node}: {neighbors}")

    # paths = engine._find_paths_in_graph("color_1", "color_0")
    # print(f"Paths from color_1 to color_0: {paths}")





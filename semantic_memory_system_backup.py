from __future__ import annotations
import numpy as np
import time
import math
import copy
import json
import hashlib
from collections.abc import Callable
from typing import Dict, List, Any, Tuple, Optional, Union, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import random

# Advanced imports for semantic processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ============================================================================
# ADVANCED SEMANTIC MEMORY SYSTEM FOR ARC-AGI-2
# ============================================================================

class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"         # Specific experiences
    SEMANTIC = "semantic"         # General knowledge
    PROCEDURAL = "procedural"     # How-to knowledge
    WORKING = "working"           # Temporary active memory
    META = "meta"                 # Knowledge about knowledge

class KnowledgeLevel(Enum):
    """Levels of knowledge abstraction"""
    CONCRETE = 1      # Specific instances
    CATEGORICAL = 2   # Categories and classes
    RELATIONAL = 3    # Relationships and patterns
    ABSTRACT = 4      # Abstract principles
    META = 5         # Meta-knowledge

@dataclass
class MemoryNode:
    """Represents a node in semantic memory"""
    node_id: str
    content: Any
    memory_type: MemoryType
    knowledge_level: KnowledgeLevel
    activation_level: float
    creation_time: float
    last_accessed: float
    access_count: int
    confidence: float
    tags: Set[str] = field(default_factory=set)
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticPattern:
    """Represents a semantic pattern"""
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    instances: List[str]  # Memory node IDs
    generalization_level: int
    confidence: float
    usage_frequency: int
    
@dataclass
class KnowledgeCluster:
    """Represents a cluster of related knowledge"""
    cluster_id: str
    center_node_id: str
    member_node_ids: List[str]
    cluster_features: Dict[str, Any]
    coherence_score: float
    
class AdvancedSemanticMemorySystem:
    """Advanced semantic memory system for ARC-AGI-2 challenges"""
    
    def __init__(self):
        # Core memory structures
        self.memory_nodes: Dict[str, MemoryNode] = {}
        self.semantic_patterns: Dict[str, SemanticPattern] = {}
        self.knowledge_clusters: Dict[str, KnowledgeCluster] = {}
        
        # Memory management
        self.max_memory_nodes = 10000
        self.activation_decay_rate = 0.95
        self.connection_threshold = 0.3
        self.clustering_threshold = 0.7
        
        # Indexing structures
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> node_ids
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)  # type -> node_ids
        self.level_index: Dict[KnowledgeLevel, Set[str]] = defaultdict(set)  # level -> node_ids
        
        # Semantic processing
        self.feature_extractors: Dict[str, callable] = {}
        self.similarity_functions: Dict[str, callable] = {}
        
        # Performance tracking
        self.retrieval_stats: Dict[str, Any] = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'average_retrieval_time': 0.0,
            'cache_hits': 0
        }
        
        # Initialize system
        self._initialize_feature_extractors()
        self._initialize_similarity_functions()
        
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        
        self.feature_extractors = {
            'grid_features': self._extract_grid_features,
            'pattern_features': self._extract_pattern_features,
            'transformation_features': self._extract_transformation_features,
            'context_features': self._extract_context_features,
            'semantic_features': self._extract_semantic_features
        }
    
    def _initialize_similarity_functions(self):
        """Initialize similarity calculation functions"""
        
        self.similarity_functions = {
            'grid_similarity': self._calculate_grid_similarity,
            'pattern_similarity': self._calculate_pattern_similarity,
            'semantic_similarity': self._calculate_semantic_similarity,
            'structural_similarity': self._calculate_structural_similarity
        }
    
    def store_memory(self, content: Any, memory_type: MemoryType, 
                    knowledge_level: KnowledgeLevel, tags: Set[str] = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Store new memory in the system"""
        try:
            # Generate unique node ID
            node_id = self._generate_node_id(content, memory_type)
            
            # Create memory node
            memory_node = MemoryNode(
                node_id=node_id,
                content=content,
                memory_type=memory_type,
                knowledge_level=knowledge_level,
                activation_level=1.0,
                creation_time=time.time(),
                last_accessed=time.time(),
                access_count=1,
                confidence=0.8,
                tags=tags or set(),
                metadata=metadata or {}
            )
            
            # Store in main memory
            self.memory_nodes[node_id] = memory_node
            
            # Update indices
            self._update_indices(memory_node)
            
            # Find and create connections
            self._create_connections(memory_node)
            
            # Update semantic patterns
            self._update_semantic_patterns(memory_node)
            
            # Trigger clustering if needed
            if len(self.memory_nodes) % 100 == 0:  # Cluster every 100 nodes
                self._update_knowledge_clusters()
            
            # Memory management
            self._manage_memory_capacity()
            
            return node_id
            
        except Exception as e:
            return f"Error storing memory: {str(e)}"
    
    def retrieve_memory(self, query: Any, memory_type: MemoryType = None,
                       knowledge_level: KnowledgeLevel = None, 
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        try:
            start_time = time.time()
            self.retrieval_stats['total_retrievals'] += 1
            
            # Extract query features
            query_features = self._extract_query_features(query)
            
            # Find candidate nodes
            candidates = self._find_candidate_nodes(query, memory_type, knowledge_level)
            
            # Calculate similarities
            similarities = []
            for node_id in candidates:
                node = self.memory_nodes[node_id]
                similarity = self._calculate_node_similarity(query_features, node)
                similarities.append((node_id, similarity))
            
            # Sort by similarity and select top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:max_results]
            
            # Prepare results
            results = []
            for node_id, similarity in top_results:
                node = self.memory_nodes[node_id]
                
                # Update access statistics
                node.last_accessed = time.time()
                node.access_count += 1
                node.activation_level = min(node.activation_level + 0.1, 1.0)
                
                results.append({
                    'node_id': node_id,
                    'content': node.content,
                    'similarity': similarity,
                    'confidence': node.confidence,
                    'memory_type': node.memory_type.value,
                    'knowledge_level': node.knowledge_level.value,
                    'tags': list(node.tags),
                    'metadata': node.metadata
                })
            
            # Update statistics
            retrieval_time = time.time() - start_time
            self._update_retrieval_stats(retrieval_time, len(results) > 0)
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def semantic_search(self, concept: str, context: Dict[str, Any] = None,
                       similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Perform semantic search for concept"""
        try:
            # Find semantically related memories
            related_nodes = []
            
            for node_id, node in self.memory_nodes.items():
                # Calculate semantic similarity
                semantic_sim = self._calculate_concept_similarity(concept, node, context)
                
                if semantic_sim >= similarity_threshold:
                    related_nodes.append({
                        'node_id': node_id,
                        'content': node.content,
                        'semantic_similarity': semantic_sim,
                        'confidence': node.confidence,
                        'tags': list(node.tags)
                    })
            
            # Sort by semantic similarity
            related_nodes.sort(key=lambda x: x['semantic_similarity'], reverse=True)
            
            return related_nodes
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def associative_retrieval(self, seed_node_id: str, 
                            association_strength: float = 0.3,
                            max_depth: int = 3) -> List[Dict[str, Any]]:
        """Retrieve memories through associative connections"""
        try:
            if seed_node_id not in self.memory_nodes:
                return []
            
            visited = set()
            results = []
            queue = deque([(seed_node_id, 1.0, 0)])  # (node_id, strength, depth)
            
            while queue:
                current_id, current_strength, depth = queue.popleft()
                
                if current_id in visited or depth > max_depth:
                    continue
                
                visited.add(current_id)
                current_node = self.memory_nodes[current_id]
                
                # Add to results if not the seed node
                if current_id != seed_node_id:
                    results.append({
                        'node_id': current_id,
                        'content': current_node.content,
                        'association_strength': current_strength,
                        'depth': depth,
                        'confidence': current_node.confidence
                    })
                
                # Add connected nodes to queue
                for connected_id, connection_strength in current_node.connections.items():
                    if (connected_id not in visited and 
                        connection_strength >= association_strength):
                        
                        new_strength = current_strength * connection_strength
                        queue.append((connected_id, new_strength, depth + 1))
            
            # Sort by association strength
            results.sort(key=lambda x: x['association_strength'], reverse=True)
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def consolidate_knowledge(self, consolidation_threshold: float = 0.8) -> Dict[str, Any]:
        """Consolidate related knowledge into higher-level abstractions"""
        try:
            consolidations = []
            
            # Find highly connected node clusters
            high_connectivity_clusters = self._find_high_connectivity_clusters(consolidation_threshold)
            
            for cluster in high_connectivity_clusters:
                # Extract common patterns
                common_patterns = self._extract_common_patterns(cluster)
                
                # Create consolidated knowledge node
                if common_patterns:
                    consolidated_content = self._create_consolidated_content(common_patterns)
                    
                    consolidated_id = self.store_memory(
                        content=consolidated_content,
                        memory_type=MemoryType.SEMANTIC,
                        knowledge_level=KnowledgeLevel.ABSTRACT,
                        tags={'consolidated', 'high_level'},
                        metadata={'source_nodes': cluster, 'consolidation_time': time.time()}
                    )
                    
                    consolidations.append({
                        'consolidated_id': consolidated_id,
                        'source_nodes': cluster,
                        'patterns': common_patterns
                    })
            
            return {
                'consolidations_created': len(consolidations),
                'consolidations': consolidations,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def forget_unused_memories(self, usage_threshold: float = 0.1,
                             age_threshold: float = 86400 * 30) -> Dict[str, Any]:  # 30 days
        """Remove unused or old memories to manage capacity"""
        try:
            current_time = time.time()
            nodes_to_remove = []
            
            for node_id, node in self.memory_nodes.items():
                # Check usage criteria
                age = current_time - node.creation_time
                time_since_access = current_time - node.last_accessed
                
                should_forget = (
                    node.activation_level < usage_threshold or
                    (age > age_threshold and time_since_access > age_threshold / 2)
                )
                
                # Don't forget high-confidence or frequently accessed memories
                if (should_forget and 
                    node.confidence < 0.9 and 
                    node.access_count < 10):
                    nodes_to_remove.append(node_id)
            
            # Remove selected nodes
            removed_count = 0
            for node_id in nodes_to_remove:
                if self._remove_memory_node(node_id):
                    removed_count += 1
            
            return {
                'nodes_removed': removed_count,
                'nodes_evaluated': len(self.memory_nodes),
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}

    # ============================================================================
    # HELPER METHODS FOR SEMANTIC MEMORY OPERATIONS
    # ============================================================================

    def _generate_node_id(self, content: Any, memory_type: MemoryType) -> str:
        """Generate unique node ID"""
        try:
            content_str = json.dumps(content, sort_keys=True, default=str)
            hash_input = f"{memory_type.value}_{content_str}_{time.time()}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
        except Exception:
            return f"node_{int(time.time() * 1000000) % 1000000}"

    def _update_indices(self, memory_node: MemoryNode):
        """Update indexing structures"""
        node_id = memory_node.node_id

        # Update tag index
        for tag in memory_node.tags:
            self.tag_index[tag].add(node_id)

        # Update type index
        self.type_index[memory_node.memory_type].add(node_id)

        # Update level index
        self.level_index[memory_node.knowledge_level].add(node_id)

    def _create_connections(self, new_node: MemoryNode):
        """Create connections between new node and existing nodes"""
        try:
            new_features = self._extract_node_features(new_node)

            for existing_id, existing_node in self.memory_nodes.items():
                if existing_id == new_node.node_id:
                    continue

                # Calculate similarity
                existing_features = self._extract_node_features(existing_node)
                similarity = self._calculate_feature_similarity(new_features, existing_features)

                # Create connection if similarity is above threshold
                if similarity >= self.connection_threshold:
                    new_node.connections[existing_id] = similarity
                    existing_node.connections[new_node.node_id] = similarity

        except Exception:
            pass  # Continue without connections if error occurs

    def _extract_node_features(self, node: MemoryNode) -> Dict[str, Any]:
        """Extract features from a memory node"""
        features = {
            'memory_type': node.memory_type.value,
            'knowledge_level': node.knowledge_level.value,
            'tags': list(node.tags),
            'confidence': node.confidence,
            'activation_level': node.activation_level
        }

        # Extract content-specific features
        if isinstance(node.content, np.ndarray):
            features.update(self._extract_grid_features(node.content))
        elif isinstance(node.content, dict):
            features.update(self._extract_dict_features(node.content))
        elif isinstance(node.content, str):
            features.update(self._extract_text_features(node.content))

        return features

    def _extract_grid_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract features from grid content"""
        try:
            return {
                'shape': grid.shape,
                'unique_values': len(np.unique(grid)),
                'density': np.count_nonzero(grid) / grid.size,
                'max_value': int(np.max(grid)),
                'min_value': int(np.min(grid)),
                'has_symmetry': (np.array_equal(grid, np.fliplr(grid)) or
                               np.array_equal(grid, np.flipud(grid))),
                'mean_value': float(np.mean(grid))
            }
        except Exception:
            return {}

    def _extract_pattern_features(self, pattern: Any) -> Dict[str, Any]:
        """Extract features from pattern content"""
        if isinstance(pattern, dict):
            return {
                'pattern_type': pattern.get('type', 'unknown'),
                'complexity': pattern.get('complexity', 1),
                'confidence': pattern.get('confidence', 0.5),
                'num_elements': len(pattern.get('elements', []))
            }
        return {}

    def _extract_transformation_features(self, transformation: Any) -> Dict[str, Any]:
        """Extract features from transformation content"""
        if isinstance(transformation, dict):
            return {
                'transformation_type': transformation.get('type', 'unknown'),
                'parameters': len(transformation.get('parameters', {})),
                'reversible': transformation.get('reversible', False),
                'complexity': transformation.get('complexity', 1)
            }
        return {}

    def _extract_context_features(self, context: Any) -> Dict[str, Any]:
        """Extract features from context content"""
        if isinstance(context, dict):
            return {
                'context_size': len(context),
                'has_temporal': 'time' in context or 'timestamp' in context,
                'has_spatial': 'position' in context or 'location' in context,
                'complexity_level': context.get('complexity', 'medium')
            }
        return {}

    def _extract_semantic_features(self, content: Any) -> Dict[str, Any]:
        """Extract semantic features from content"""
        features = {}

        if isinstance(content, dict):
            # Look for semantic indicators
            if 'meaning' in content:
                features['has_explicit_meaning'] = True
            if 'concept' in content:
                features['has_concept'] = True
            if 'relationship' in content:
                features['has_relationship'] = True

        return features

    def _extract_dict_features(self, data: dict) -> Dict[str, Any]:
        """Extract features from dictionary content"""
        return {
            'dict_size': len(data),
            'has_nested': any(isinstance(v, dict) for v in data.values()),
            'has_lists': any(isinstance(v, list) for v in data.values()),
            'key_types': list(set(type(k).__name__ for k in data.keys()))
        }

    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text content"""
        return {
            'text_length': len(text),
            'word_count': len(text.split()),
            'has_numbers': any(c.isdigit() for c in text),
            'has_special_chars': any(not c.isalnum() and not c.isspace() for c in text)
        }

    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between feature sets"""
        try:
            common_keys = set(features1.keys()) & set(features2.keys())
            if not common_keys:
                return 0.0

            similarities = []
            for key in common_keys:
                val1, val2 = features1[key], features2[key]

                if isinstance(val1, bool) and isinstance(val2, bool):
                    similarities.append(1.0 if val1 == val2 else 0.0)
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == 0 and val2 == 0:
                        similarities.append(1.0)
                    else:
                        max_val = max(abs(val1), abs(val2), 1)
                        similarities.append(1.0 - abs(val1 - val2) / max_val)
                elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                    set1, set2 = set(val1), set(val2)
                    if len(set1) == 0 and len(set2) == 0:
                        similarities.append(1.0)
                    else:
                        intersection = len(set1 & set2)
                        union = len(set1 | set2)
                        similarities.append(intersection / union if union > 0 else 0.0)
                else:
                    similarities.append(1.0 if val1 == val2 else 0.0)

            return np.mean(similarities) if similarities else 0.0

        except Exception:
            return 0.0

    def _update_semantic_patterns(self, memory_node: MemoryNode):
        """Update semantic patterns based on new memory node"""
        try:
            node_features = self._extract_node_features(memory_node)

            # Find existing patterns that match
            matching_patterns = []
            for pattern_id, pattern in self.semantic_patterns.items():
                similarity = self._calculate_pattern_similarity(node_features, pattern.features)
                if similarity > 0.7:
                    matching_patterns.append((pattern_id, similarity))

            if matching_patterns:
                # Update best matching pattern
                best_pattern_id = max(matching_patterns, key=lambda x: x[1])[0]
                best_pattern = self.semantic_patterns[best_pattern_id]
                best_pattern.instances.append(memory_node.node_id)
                best_pattern.usage_frequency += 1
                best_pattern.confidence = min(best_pattern.confidence + 0.05, 1.0)
            else:
                # Create new pattern
                pattern_id = f"pattern_{len(self.semantic_patterns)}"
                new_pattern = SemanticPattern(
                    pattern_id=pattern_id,
                    pattern_type=memory_node.memory_type.value,
                    features=node_features,
                    instances=[memory_node.node_id],
                    generalization_level=memory_node.knowledge_level.value,
                    confidence=0.6,
                    usage_frequency=1
                )
                self.semantic_patterns[pattern_id] = new_pattern

        except Exception:
            pass  # Continue without pattern updates if error occurs

    def _calculate_grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids"""
        try:
            if grid1.shape != grid2.shape:
                return 0.0

            # Exact match
            if np.array_equal(grid1, grid2):
                return 1.0

            # Partial match
            matches = np.sum(grid1 == grid2)
            total = grid1.size
            return matches / total

        except Exception:
            return 0.0

    def _calculate_pattern_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between patterns"""
        return self._calculate_feature_similarity(features1, features2)

    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between concepts"""
        # Simple semantic similarity based on string similarity
        if concept1 == concept2:
            return 1.0

        # Jaccard similarity on words
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())

        if len(words1) == 0 and len(words2) == 0:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_structural_similarity(self, struct1: Any, struct2: Any) -> float:
        """Calculate structural similarity"""
        if type(struct1) != type(struct2):
            return 0.0

        if isinstance(struct1, dict) and isinstance(struct2, dict):
            common_keys = set(struct1.keys()) & set(struct2.keys())
            all_keys = set(struct1.keys()) | set(struct2.keys())
            return len(common_keys) / len(all_keys) if all_keys else 1.0

        return 1.0 if struct1 == struct2 else 0.0

    # Additional helper methods for memory management
    def _extract_query_features(self, query: Any) -> Dict[str, Any]:
        """Extract features from query"""
        if isinstance(query, np.ndarray):
            return self._extract_grid_features(query)
        elif isinstance(query, dict):
            return self._extract_dict_features(query)
        elif isinstance(query, str):
            return self._extract_text_features(query)
        else:
            return {'query_type': type(query).__name__}

    def _find_candidate_nodes(self, query: Any, memory_type: MemoryType = None,
                            knowledge_level: KnowledgeLevel = None) -> List[str]:
        """Find candidate nodes for retrieval"""
        candidates = set(self.memory_nodes.keys())

        # Filter by memory type
        if memory_type:
            type_candidates = self.type_index.get(memory_type, set())
            candidates &= type_candidates

        # Filter by knowledge level
        if knowledge_level:
            level_candidates = self.level_index.get(knowledge_level, set())
            candidates &= level_candidates

        return list(candidates)

    def _calculate_node_similarity(self, query_features: Dict[str, Any], node: MemoryNode) -> float:
        """Calculate similarity between query and memory node"""
        node_features = self._extract_node_features(node)
        base_similarity = self._calculate_feature_similarity(query_features, node_features)

        # Boost similarity based on activation level and confidence
        boost = (node.activation_level + node.confidence) / 2.0 * 0.1

        return min(base_similarity + boost, 1.0)

    def _update_retrieval_stats(self, retrieval_time: float, success: bool):
        """Update retrieval statistics"""
        self.retrieval_stats['total_retrievals'] += 1

        if success:
            self.retrieval_stats['successful_retrievals'] += 1

        # Update average retrieval time
        total = self.retrieval_stats['total_retrievals']
        current_avg = self.retrieval_stats['average_retrieval_time']
        self.retrieval_stats['average_retrieval_time'] = (
            (current_avg * (total - 1) + retrieval_time) / total
        )

    def _calculate_concept_similarity(self, concept: str, node: MemoryNode,
                                    context: Dict[str, Any] = None) -> float:
        """Calculate semantic similarity between concept and memory node"""
        base_similarity = 0.0

        # Check tags
        for tag in node.tags:
            tag_similarity = self._calculate_semantic_similarity(concept, tag)
            base_similarity = max(base_similarity, tag_similarity)

        # Check content if it's text
        if isinstance(node.content, str):
            content_similarity = self._calculate_semantic_similarity(concept, node.content)
            base_similarity = max(base_similarity, content_similarity)

        # Check metadata
        if 'concept' in node.metadata:
            metadata_similarity = self._calculate_semantic_similarity(
                concept, str(node.metadata['concept'])
            )
            base_similarity = max(base_similarity, metadata_similarity)

        return base_similarity

    def _update_knowledge_clusters(self):
        """Update knowledge clusters"""
        try:
            if len(self.memory_nodes) < 10:  # Need minimum nodes for clustering
                return

            # Extract features for all nodes
            node_features = {}
            for node_id, node in self.memory_nodes.items():
                node_features[node_id] = self._extract_node_features(node)

            # Simple clustering based on similarity
            clusters = self._perform_simple_clustering(node_features)

            # Update cluster structures
            self.knowledge_clusters.clear()
            for i, cluster_nodes in enumerate(clusters):
                if len(cluster_nodes) >= 3:  # Minimum cluster size
                    cluster_id = f"cluster_{i}"
                    center_node = self._find_cluster_center(cluster_nodes, node_features)

                    self.knowledge_clusters[cluster_id] = KnowledgeCluster(
                        cluster_id=cluster_id,
                        center_node_id=center_node,
                        member_node_ids=cluster_nodes,
                        cluster_features=self._extract_cluster_features(cluster_nodes),
                        coherence_score=self._calculate_cluster_coherence(cluster_nodes, node_features)
                    )

        except Exception:
            pass  # Continue without clustering if error occurs

    def _perform_simple_clustering(self, node_features: Dict[str, Dict[str, Any]]) -> List[List[str]]:
        """Perform simple clustering of nodes"""
        node_ids = list(node_features.keys())
        clusters = []
        visited = set()

        for node_id in node_ids:
            if node_id in visited:
                continue

            # Start new cluster
            cluster = [node_id]
            visited.add(node_id)

            # Find similar nodes
            for other_id in node_ids:
                if other_id in visited:
                    continue

                similarity = self._calculate_feature_similarity(
                    node_features[node_id], node_features[other_id]
                )

                if similarity >= self.clustering_threshold:
                    cluster.append(other_id)
                    visited.add(other_id)

            clusters.append(cluster)

        return clusters

    def _find_cluster_center(self, cluster_nodes: List[str],
                           node_features: Dict[str, Dict[str, Any]]) -> str:
        """Find the center node of a cluster"""
        if len(cluster_nodes) == 1:
            return cluster_nodes[0]

        # Find node with highest average similarity to others
        best_node = cluster_nodes[0]
        best_avg_similarity = 0.0

        for node_id in cluster_nodes:
            similarities = []
            for other_id in cluster_nodes:
                if node_id != other_id:
                    sim = self._calculate_feature_similarity(
                        node_features[node_id], node_features[other_id]
                    )
                    similarities.append(sim)

            avg_similarity = np.mean(similarities) if similarities else 0.0
            if avg_similarity > best_avg_similarity:
                best_avg_similarity = avg_similarity
                best_node = node_id

        return best_node

    def _extract_cluster_features(self, cluster_nodes: List[str]) -> Dict[str, Any]:
        """Extract features for a cluster"""
        return {
            'size': len(cluster_nodes),
            'memory_types': list(set(self.memory_nodes[nid].memory_type.value for nid in cluster_nodes)),
            'knowledge_levels': list(set(self.memory_nodes[nid].knowledge_level.value for nid in cluster_nodes)),
            'avg_confidence': np.mean([self.memory_nodes[nid].confidence for nid in cluster_nodes])
        }

    def _calculate_cluster_coherence(self, cluster_nodes: List[str],
                                   node_features: Dict[str, Dict[str, Any]]) -> float:
        """Calculate coherence score for a cluster"""
        if len(cluster_nodes) <= 1:
            return 1.0

        similarities = []
        for i, node1 in enumerate(cluster_nodes):
            for node2 in cluster_nodes[i+1:]:
                sim = self._calculate_feature_similarity(
                    node_features[node1], node_features[node2]
                )
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _manage_memory_capacity(self):
        """Manage memory capacity by removing old/unused nodes"""
        if len(self.memory_nodes) <= self.max_memory_nodes:
            return

        # Decay activation levels
        current_time = time.time()
        for node in self.memory_nodes.values():
            time_since_access = current_time - node.last_accessed
            decay_factor = self.activation_decay_rate ** (time_since_access / 3600)  # Decay per hour
            node.activation_level *= decay_factor

        # Remove nodes with lowest activation
        nodes_by_activation = sorted(
            self.memory_nodes.items(),
            key=lambda x: x[1].activation_level
        )

        nodes_to_remove = len(self.memory_nodes) - self.max_memory_nodes
        for i in range(nodes_to_remove):
            node_id, node = nodes_by_activation[i]
            # Don't remove high-confidence or recently created nodes
            if node.confidence < 0.8 and (current_time - node.creation_time) > 3600:
                self._remove_memory_node(node_id)

    def _remove_memory_node(self, node_id: str) -> bool:
        """Remove a memory node and clean up references"""
        try:
            if node_id not in self.memory_nodes:
                return False

            node = self.memory_nodes[node_id]

            # Remove from indices
            for tag in node.tags:
                self.tag_index[tag].discard(node_id)
            self.type_index[node.memory_type].discard(node_id)
            self.level_index[node.knowledge_level].discard(node_id)

            # Remove connections
            for connected_id in node.connections:
                if connected_id in self.memory_nodes:
                    self.memory_nodes[connected_id].connections.pop(node_id, None)

            # Remove from patterns
            for pattern in self.semantic_patterns.values():
                if node_id in pattern.instances:
                    pattern.instances.remove(node_id)

            # Remove from clusters
            for cluster in self.knowledge_clusters.values():
                if node_id in cluster.member_node_ids:
                    cluster.member_node_ids.remove(node_id)

            # Remove the node itself
            del self.memory_nodes[node_id]

            return True

        except Exception:
            return False

    def _find_high_connectivity_clusters(self, threshold: float) -> List[List[str]]:
        """Find clusters of highly connected nodes"""
        clusters = []
        visited = set()

        for node_id, node in self.memory_nodes.items():
            if node_id in visited:
                continue

            # Find highly connected component
            cluster = []
            queue = deque([node_id])

            while queue:
                current_id = queue.popleft()
                if current_id in visited:
                    continue

                visited.add(current_id)
                cluster.append(current_id)

                current_node = self.memory_nodes[current_id]
                for connected_id, strength in current_node.connections.items():
                    if strength >= threshold and connected_id not in visited:
                        queue.append(connected_id)

            if len(cluster) >= 3:  # Minimum cluster size
                clusters.append(cluster)

        return clusters

    def _extract_common_patterns(self, cluster_nodes: List[str]) -> List[Dict[str, Any]]:
        """Extract common patterns from a cluster of nodes"""
        patterns = []

        try:
            # Extract features from all nodes in cluster
            cluster_features = []
            for node_id in cluster_nodes:
                node = self.memory_nodes[node_id]
                features = self._extract_node_features(node)
                cluster_features.append(features)

            # Find common feature patterns
            if cluster_features:
                # Find most common memory type
                memory_types = [f.get('memory_type') for f in cluster_features]
                most_common_type = max(set(memory_types), key=memory_types.count)

                # Find most common knowledge level
                knowledge_levels = [f.get('knowledge_level') for f in cluster_features]
                most_common_level = max(set(knowledge_levels), key=knowledge_levels.count)

                # Find common tags
                all_tags = []
                for f in cluster_features:
                    all_tags.extend(f.get('tags', []))
                common_tags = [tag for tag in set(all_tags) if all_tags.count(tag) >= len(cluster_nodes) // 2]

                patterns.append({
                    'type': 'common_attributes',
                    'memory_type': most_common_type,
                    'knowledge_level': most_common_level,
                    'common_tags': common_tags,
                    'cluster_size': len(cluster_nodes)
                })

        except Exception:
            pass

        return patterns

    def _create_consolidated_content(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create consolidated content from patterns"""
        consolidated = {
            'type': 'consolidated_knowledge',
            'patterns': patterns,
            'consolidation_timestamp': time.time(),
            'abstraction_level': 'high'
        }

        # Extract key insights
        if patterns:
            pattern = patterns[0]  # Use first pattern as base
            consolidated.update({
                'primary_memory_type': pattern.get('memory_type'),
                'primary_knowledge_level': pattern.get('knowledge_level'),
                'key_concepts': pattern.get('common_tags', []),
                'cluster_size': pattern.get('cluster_size', 0)
            })

        return consolidated

    # Performance and debugging methods
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            current_time = time.time()

            # Basic counts
            total_nodes = len(self.memory_nodes)
            total_patterns = len(self.semantic_patterns)
            total_clusters = len(self.knowledge_clusters)

            # Memory type distribution
            type_distribution = {}
            for memory_type in MemoryType:
                type_distribution[memory_type.value] = len(self.type_index[memory_type])

            # Knowledge level distribution
            level_distribution = {}
            for knowledge_level in KnowledgeLevel:
                level_distribution[knowledge_level.value] = len(self.level_index[knowledge_level])

            # Activation statistics
            activations = [node.activation_level for node in self.memory_nodes.values()]
            avg_activation = np.mean(activations) if activations else 0.0

            # Connection statistics
            total_connections = sum(len(node.connections) for node in self.memory_nodes.values())
            avg_connections = total_connections / total_nodes if total_nodes > 0 else 0.0

            # Age statistics
            ages = [(current_time - node.creation_time) / 3600 for node in self.memory_nodes.values()]  # Hours
            avg_age = np.mean(ages) if ages else 0.0

            return {
                'total_nodes': total_nodes,
                'total_patterns': total_patterns,
                'total_clusters': total_clusters,
                'memory_type_distribution': type_distribution,
                'knowledge_level_distribution': level_distribution,
                'average_activation': avg_activation,
                'total_connections': total_connections,
                'average_connections_per_node': avg_connections,
                'average_age_hours': avg_age,
                'retrieval_stats': self.retrieval_stats.copy(),
                'memory_capacity_used': f"{total_nodes}/{self.max_memory_nodes}",
                'capacity_percentage': (total_nodes / self.max_memory_nodes) * 100
            }

        except Exception as e:
            return {'error': str(e)}

    def export_memory_graph(self) -> Dict[str, Any]:
        """Export memory graph for visualization"""
        try:
            nodes = []
            edges = []

            for node_id, node in self.memory_nodes.items():
                nodes.append({
                    'id': node_id,
                    'type': node.memory_type.value,
                    'level': node.knowledge_level.value,
                    'activation': node.activation_level,
                    'confidence': node.confidence,
                    'tags': list(node.tags)
                })

                for connected_id, strength in node.connections.items():
                    edges.append({
                        'source': node_id,
                        'target': connected_id,
                        'weight': strength
                    })

            return {
                'nodes': nodes,
                'edges': edges,
                'clusters': [
                    {
                        'id': cluster_id,
                        'center': cluster.center_node_id,
                        'members': cluster.member_node_ids,
                        'coherence': cluster.coherence_score
                    }
                    for cluster_id, cluster in self.knowledge_clusters.items()
                ]
            }

        except Exception as e:
            return {'error': str(e)}

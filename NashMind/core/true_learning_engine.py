#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
محرك التعلم الحقيقي - مكون جديد في NashMind
يضيف قدرات التعلم الحقيقي والذاكرة الدائمة للنظام
"""

import json
import os
import time
import hashlib
import numpy as np
from collections import defaultdict
import pickle
import re

def convert_numpy_to_python(obj):
    """تحويل NumPy arrays و int64 إلى أنواع Python الأساسية"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj

class TrueLearningEngine:
    """
    محرك التعلم الحقيقي - يتعلم من كل تجربة ويحفظها دائماً
    مدمج في نظام NashMind ACES
    """
    
    def __init__(self, memory_file="nashmind_memory.json", patterns_file="nashmind_patterns.pkl"):
        self.memory_file = memory_file
        self.patterns_file = patterns_file
        
        # الذاكرة الدائمة - تحفظ كل ما يتعلمه NashMind
        self.long_term_memory = self.load_memory()
        
        # أنماط متعلمة - يكتشفها بنفسه
        self.learned_patterns = self.load_patterns()
        
        # شبكة المفاهيم - يبنيها تدريجياً
        self.concept_network = defaultdict(list)
        
        # خبرات سابقة - يتعلم منها
        self.experiences = []
        
        # قدرات التعلم - تنمو مع الوقت
        self.learning_capabilities = {
            "pattern_recognition": 0.1,
            "logical_reasoning": 0.1,
            "creative_thinking": 0.1,
            "problem_solving": 0.1,
            "learning_efficiency": 0.1,
            "memory_consolidation": 0.1,
            "concept_formation": 0.1
        }
        
        print("🧠 تم تهيئة محرك التعلم الحقيقي في NashMind")
        print(f"📚 الذاكرة طويلة المدى: {len(self.long_term_memory)} عنصر")
        print(f"🔍 الأنماط المتعلمة: {len(self.learned_patterns)} نمط")

    def load_memory(self):
        """تحميل الذاكرة طويلة المدى من الملف"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_memory(self):
        """حفظ الذاكرة طويلة المدى"""
        # تحويل البيانات قبل الحفظ
        converted_memory = convert_numpy_to_python(self.long_term_memory)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(converted_memory, f, ensure_ascii=False, indent=2)

    def load_patterns(self):
        """تحميل الأنماط المتعلمة"""
        if os.path.exists(self.patterns_file):
            try:
                with open(self.patterns_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    def save_patterns(self):
        """حفظ الأنماط المتعلمة"""
        with open(self.patterns_file, 'wb') as f:
            pickle.dump(self.learned_patterns, f)

    def learn_from_experience(self, experience_data, context="general"):
        """التعلم من تجربة جديدة - القلب النابض للتعلم الحقيقي"""
        
        print(f"🔍 تعلم من تجربة جديدة: {str(experience_data)[:50]}...")
        
        # 1. تحليل التجربة الجديدة
        analysis = self.analyze_experience(experience_data)
        
        # 2. البحث في الذاكرة عن تجارب مشابهة
        similar_experiences = self.find_similar_experiences(analysis)
        
        # 3. اكتشاف أنماط جديدة
        new_patterns = self.discover_patterns(experience_data, similar_experiences)
        
        # 4. بناء روابط مفاهيمية
        self.build_concept_connections(analysis, similar_experiences)
        
        # 5. تحديث القدرات بناءً على التعلم
        self.update_learning_capabilities(new_patterns)
        
        # 6. حفظ التجربة في الذاكرة طويلة المدى
        memory_id = self.store_experience_in_memory(experience_data, analysis, context)
        
        # 7. حفظ التغييرات
        self.save_memory()
        self.save_patterns()
        
        return {
            "memory_id": memory_id,
            "patterns_discovered": len(new_patterns),
            "concepts_connected": len(self.concept_network),
            "learning_growth": self.calculate_learning_growth()
        }

    def analyze_experience(self, experience_data):
        """تحليل التجربة الجديدة"""
        
        # استخراج الكلمات المفتاحية
        if isinstance(experience_data, str):
            text = experience_data
        elif isinstance(experience_data, dict):
            text = str(experience_data)
        else:
            text = str(experience_data)
        
        keywords = re.findall(r'\b\w+\b', text.lower())
        
        # تحليل التعقيد
        complexity = len(keywords) / 10
        
        # تحديد النوع
        experience_type = self.classify_experience_type(text)
        
        # استخراج المفاهيم
        concepts = self.extract_concepts(keywords)
        
        return {
            "keywords": keywords,
            "complexity": complexity,
            "type": experience_type,
            "concepts": concepts,
            "timestamp": time.time(),
            "hash": hashlib.md5(text.encode()).hexdigest()
        }

    def classify_experience_type(self, text):
        """تصنيف نوع التجربة"""
        
        if any(word in text.lower() for word in ['مشكلة', 'حل', 'problem', 'solve']):
            return "problem_solving"
        elif any(word in text.lower() for word in ['تعلم', 'معرفة', 'learn', 'knowledge']):
            return "learning"
        elif any(word in text.lower() for word in ['إبداع', 'فكرة', 'creative', 'idea']):
            return "creative"
        elif any(word in text.lower() for word in ['تحليل', 'فهم', 'analyze', 'understand']):
            return "analytical"
        else:
            return "general"

    def extract_concepts(self, keywords):
        """استخراج المفاهيم من الكلمات المفتاحية"""
        
        # مفاهيم أساسية
        concept_categories = {
            "mathematical": ["رقم", "حساب", "معادلة", "number", "math", "equation"],
            "logical": ["منطق", "استنتاج", "logic", "reasoning", "inference"],
            "spatial": ["مكان", "شكل", "موقع", "space", "shape", "position"],
            "temporal": ["وقت", "زمن", "time", "temporal", "sequence"],
            "causal": ["سبب", "نتيجة", "cause", "effect", "result"]
        }
        
        concepts = []
        for category, category_words in concept_categories.items():
            if any(word in keywords for word in category_words):
                concepts.append(category)
        
        return concepts

    def find_similar_experiences(self, analysis):
        """البحث عن تجارب مشابهة في الذاكرة"""
        
        similar = []
        current_keywords = set(analysis["keywords"])
        
        for memory_id, memory_data in self.long_term_memory.items():
            if "analysis" in memory_data:
                memory_keywords = set(memory_data["analysis"].get("keywords", []))
                
                # حساب التشابه
                intersection = len(current_keywords.intersection(memory_keywords))
                union = len(current_keywords.union(memory_keywords))
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.3:  # عتبة التشابه
                        similar.append({
                            "memory_id": memory_id,
                            "similarity": similarity,
                            "data": memory_data
                        })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]

    def discover_patterns(self, experience_data, similar_experiences):
        """اكتشاف أنماط جديدة"""
        
        new_patterns = []
        
        # نمط التكرار
        if len(similar_experiences) >= 2:
            pattern_id = f"repetition_{len(self.learned_patterns)}"
            pattern = {
                "type": "repetition",
                "description": "نمط تكرار في التجارب المشابهة",
                "frequency": len(similar_experiences),
                "confidence": min(0.9, len(similar_experiences) * 0.2)
            }
            self.learned_patterns[pattern_id] = pattern
            new_patterns.append(pattern_id)
        
        # نمط التطور
        if similar_experiences:
            latest_similar = max(similar_experiences, 
                               key=lambda x: x["data"].get("analysis", {}).get("timestamp", 0))
            
            if latest_similar["data"].get("analysis", {}).get("complexity", 0) < \
               self.analyze_experience(experience_data)["complexity"]:
                pattern_id = f"evolution_{len(self.learned_patterns)}"
                pattern = {
                    "type": "evolution",
                    "description": "نمط تطور في التعقيد",
                    "growth_rate": 0.1,
                    "confidence": 0.7
                }
                self.learned_patterns[pattern_id] = pattern
                new_patterns.append(pattern_id)
        
        return new_patterns

    def build_concept_connections(self, analysis, similar_experiences):
        """بناء روابط مفاهيمية"""
        
        current_concepts = analysis["concepts"]
        
        for concept in current_concepts:
            # ربط بالمفاهيم في التجارب المشابهة
            for similar in similar_experiences:
                similar_concepts = similar["data"].get("analysis", {}).get("concepts", [])
                for similar_concept in similar_concepts:
                    if similar_concept not in self.concept_network[concept]:
                        self.concept_network[concept].append(similar_concept)
                        print(f"🔗 ربط مفاهيمي جديد: {concept} ↔ {similar_concept}")

    def update_learning_capabilities(self, new_patterns):
        """تحديث قدرات التعلم بناءً على الأنماط الجديدة"""
        
        growth_factor = len(new_patterns) * 0.01
        
        for capability in self.learning_capabilities:
            self.learning_capabilities[capability] += growth_factor
            # حد أقصى للنمو
            self.learning_capabilities[capability] = min(1.0, self.learning_capabilities[capability])

    def store_experience_in_memory(self, experience_data, analysis, context):
        """حفظ التجربة في الذاكرة طويلة المدى"""
        
        memory_id = f"EXP_{len(self.long_term_memory)}_{analysis['hash'][:8]}"
        
        memory_entry = {
            "experience_data": experience_data,
            "analysis": analysis,
            "context": context,
            "stored_at": time.time(),
            "access_count": 0,
            "importance": self.calculate_importance(analysis)
        }
        
        self.long_term_memory[memory_id] = memory_entry
        self.experiences.append(memory_id)
        
        print(f"💾 تم حفظ التجربة: {memory_id}")
        
        return memory_id

    def calculate_importance(self, analysis):
        """حساب أهمية التجربة"""
        
        importance = 0.5  # أهمية أساسية
        
        # زيادة الأهمية بناءً على التعقيد
        importance += analysis["complexity"] * 0.1
        
        # زيادة الأهمية بناءً على عدد المفاهيم
        importance += len(analysis["concepts"]) * 0.05
        
        # زيادة الأهمية للأنواع المهمة
        if analysis["type"] in ["problem_solving", "creative"]:
            importance += 0.2
        
        return min(1.0, importance)

    def calculate_learning_growth(self):
        """حساب نمو التعلم"""
        
        total_capability = sum(self.learning_capabilities.values())
        return total_capability / len(self.learning_capabilities)

    def get_learning_stats(self):
        """إحصائيات التعلم"""
        
        return {
            "total_experiences": len(self.experiences),
            "total_patterns": len(self.learned_patterns),
            "concept_connections": sum(len(connections) for connections in self.concept_network.values()),
            "learning_capabilities": self.learning_capabilities,
            "overall_learning_level": self.calculate_learning_growth()
        }

    def recall_similar_experiences(self, query):
        """استدعاء تجارب مشابهة للاستعلام"""
        
        query_analysis = self.analyze_experience(query)
        similar = self.find_similar_experiences(query_analysis)
        
        return [exp["data"] for exp in similar]

    def apply_learned_patterns(self, new_situation):
        """تطبيق الأنماط المتعلمة على موقف جديد"""
        
        applicable_patterns = []
        
        for pattern_id, pattern in self.learned_patterns.items():
            if pattern["confidence"] > 0.5:
                applicable_patterns.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "applicability": pattern["confidence"]
                })
        
        return sorted(applicable_patterns, key=lambda x: x["applicability"], reverse=True)

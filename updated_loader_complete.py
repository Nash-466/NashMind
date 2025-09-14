from __future__ import annotations
# تحديث دالة load_systems في automated_training_loop.py
# استبدل الدالة الموجودة بهذا الكود

def load_systems(self):
    """تحميل جميع الأنظمة المتاحة - النسخة المحدثة"""
    
    # قائمة جميع الأنظمة المتاحة
    all_systems = [
        # الأنظمة الأساسية
        {'module': 'enhanced_arc_solver', 'priority': 10},
        {'module': 'basic_solver', 'priority': 5},
        
        # الأنظمة المتقدمة
        {'module': 'orchestrated_meta_solver', 'priority': 9},
        {'module': 'ultra_advanced_arc_system_v2', 'priority': 9},
        {'module': 'ultimate_arc_system', 'priority': 8},
        {'module': 'perfect_arc_system_v2', 'priority': 8},
        {'module': 'perfect_arc_system', 'priority': 7},
        {'module': 'revolutionary_arc_system', 'priority': 8},
        {'module': 'enhanced_efficient_zero', 'priority': 7},
        {'module': 'deep_learning_arc_system', 'priority': 8},
        {'module': 'genius_arc_manager', 'priority': 7},
        {'module': 'advanced_simulation_engine', 'priority': 7},
        {'module': 'arc_adaptive_hybrid_system', 'priority': 8},
        {'module': 'arc_hierarchical_reasoning', 'priority': 7},
        {'module': 'arc_learning_solver', 'priority': 7},
        {'module': 'arc_revolutionary_system', 'priority': 6},
        {'module': 'arc_ultimate_system', 'priority': 7},
        {'module': 'ultimate_arc_solver', 'priority': 8},
        {'module': 'efficient_zero_engine', 'priority': 6},
        {'module': 'semantic_memory_system', 'priority': 6},
        {'module': 'symbolic_rule_engine', 'priority': 6},
        {'module': 'neural_pattern_learner', 'priority': 7},
        {'module': 'continuous_learning_system', 'priority': 7},
        {'module': 'intelligent_verification_system', 'priority': 6},
        {'module': 'true_learning_ai', 'priority': 7},
        {'module': 'ultimate_ai_system', 'priority': 8},
        {'module': 'ultra_advanced_arc_system', 'priority': 8},
        
        # الغلاف الموحد (أعلى أولوية)
        {'module': 'unified_solver_wrapper', 'priority': 15},
    ]
    
    # تحميل كل نظام
    for system_info in all_systems:
        try:
            module = __import__(system_info['module'])
            
            if hasattr(module, 'solve_task'):
                self.systems.append({
                    'name': system_info['module'],
                    'solve': module.solve_task,
                    'priority': system_info['priority']
                })
                logger.info(f"✓ تم تحميل: {system_info['module']}")
            else:
                logger.warning(f"⚠ {system_info['module']} لا يحتوي على solve_task")
                
        except Exception as e:
            logger.warning(f"✗ فشل تحميل {system_info['module']}: {e}")
    
    # ترتيب الأنظمة حسب الأولوية
    self.systems.sort(key=lambda x: x['priority'], reverse=True)
    
    logger.info(f"تم تحميل {len(self.systems)} نظام بنجاح")
    
    # إضافة نظام احتياطي إذا لم يتم تحميل أي نظام
    if not self.systems:
        logger.warning("لم يتم تحميل أي نظام! سيتم استخدام النظام الافتراضي")
        self.systems.append({
            'name': 'default_solver',
            'solve': self._default_solver,
            'priority': 1
        })

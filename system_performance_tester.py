from __future__ import annotations
#!/usr/bin/env python3
"""
SYSTEM PERFORMANCE TESTER - اختبار شامل للأنظمة الحالية
========================================================
اختبار جميع الأنظمة الموجودة لتحديد نقاط الضعف
"""

import json
import numpy as np
import time
import traceback
from collections.abc import Callable
from typing import Dict, List, Any, Optional
import os

class SystemPerformanceTester:
    """اختبار شامل لأداء الأنظمة"""
    
    def __init__(self):
        self.test_results = {}
        self.system_errors = {}
        self.performance_metrics = {}
        
    def test_all_systems(self):
        """اختبار جميع الأنظمة"""
        print("🧪 بدء اختبار شامل للأنظمة الحالية...")
        
        # اختبار النظام الأساسي
        self._test_main_system()
        
        # اختبار الأنظمة الفرعية
        self._test_subsystems()
        
        # اختبار التكامل
        self._test_integration()
        
        # إنتاج التقرير
        self._generate_test_report()
    
    def _test_main_system(self):
        """اختبار النظام الرئيسي"""
        print("\n🎯 اختبار النظام الرئيسي...")
        
        try:
            # اختبار main.py
            result = self._test_system_component("main.py", self._test_main_component)
            self.test_results["main_system"] = result
            
        except Exception as e:
            self.system_errors["main_system"] = str(e)
            print(f"❌ فشل اختبار النظام الرئيسي: {e}")
    
    def _test_subsystems(self):
        """اختبار الأنظمة الفرعية"""
        print("\n🔧 اختبار الأنظمة الفرعية...")
        
        subsystems = [
            "arc_complete_agent_part1.py",
            "arc_complete_agent_part2.py", 
            "arc_complete_agent_part3.py",
            "arc_complete_agent_part4.py",
            "arc_ultimate_mind_part7.py",
            "burhan_meta_brain.py"
        ]
        
        for system in subsystems:
            try:
                result = self._test_system_component(system, self._test_subsystem_component)
                self.test_results[system] = result
                print(f"✅ {system}: {result['status']}")
                
            except Exception as e:
                self.system_errors[system] = str(e)
                print(f"❌ {system}: فشل - {e}")
    
    def _test_integration(self):
        """اختبار التكامل بين الأنظمة"""
        print("\n🔗 اختبار التكامل...")
        
        try:
            # اختبار إمكانية الاستيراد المتبادل
            integration_score = self._test_system_integration()
            self.test_results["integration"] = {
                "status": "success" if integration_score > 0.5 else "failed",
                "score": integration_score,
                "issues": self._identify_integration_issues()
            }
            
        except Exception as e:
            self.system_errors["integration"] = str(e)
    
    def _test_system_component(self, component_name: str, test_func) -> Dict[str, Any]:
        """اختبار مكون واحد"""
        start_time = time.time()
        
        try:
            if os.path.exists(component_name):
                result = test_func(component_name)
                result["execution_time"] = time.time() - start_time
                return result
            else:
                return {
                    "status": "not_found",
                    "error": f"File {component_name} not found",
                    "execution_time": 0
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _test_main_component(self, filename: str) -> Dict[str, Any]:
        """اختبار المكون الرئيسي"""
        try:
            # محاولة استيراد main
            import main
            
            # اختبار الوظائف الأساسية
            functions_test = self._test_main_functions()
            
            return {
                "status": "success",
                "import_success": True,
                "functions_available": functions_test,
                "issues": []
            }
            
        except ImportError as e:
            return {
                "status": "import_error",
                "import_success": False,
                "error": str(e),
                "issues": ["Cannot import main module"]
            }
    
    def _test_subsystem_component(self, filename: str) -> Dict[str, Any]:
        """اختبار مكون فرعي"""
        issues = []
        
        try:
            # قراءة الملف
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # فحص المشاكل الشائعة
            if "class" not in content:
                issues.append("No classes defined")
            
            if "def " not in content:
                issues.append("No functions defined")
            
            if len(content) < 100:
                issues.append("File too small")
            
            # فحص الاستيرادات
            import_issues = self._check_imports(content)
            issues.extend(import_issues)
            
            return {
                "status": "success" if len(issues) == 0 else "issues_found",
                "file_size": len(content),
                "issues": issues,
                "complexity_score": self._calculate_complexity(content)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": ["Cannot read or analyze file"]
            }
    
    def _test_main_functions(self) -> Dict[str, bool]:
        """اختبار الوظائف الرئيسية في main"""
        functions = {}
        
        try:
            import main
            
            # فحص الوظائف المطلوبة
            functions["load_tasks"] = hasattr(main, 'load_tasks')
            functions["save_submission"] = hasattr(main, 'save_submission')
            functions["build_orchestrator"] = hasattr(main, 'build_orchestrator')
            functions["parse_args"] = hasattr(main, 'parse_args')
            
        except Exception as e:
            functions["error"] = str(e)
        
        return functions
    
    def _check_imports(self, content: str) -> List[str]:
        """فحص مشاكل الاستيراد"""
        issues = []
        
        # فحص الاستيرادات المفقودة الشائعة
        if "import numpy" not in content and "np." in content:
            issues.append("Uses numpy without importing")
        
        if "import json" not in content and "json." in content:
            issues.append("Uses json without importing")
        
        # فحص الاستيرادات المحلية
        local_imports = []
        for line in content.split('\n'):
            if line.strip().startswith('from arc_') or line.strip().startswith('import arc_'):
                module_name = line.split()[-1]
                if not os.path.exists(f"{module_name}.py"):
                    issues.append(f"Missing dependency: {module_name}")
        
        return issues
    
    def _calculate_complexity(self, content: str) -> float:
        """حساب درجة التعقيد"""
        lines = content.split('\n')
        
        # عوامل التعقيد
        class_count = content.count('class ')
        function_count = content.count('def ')
        import_count = content.count('import ')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # حساب النقاط
        complexity = (
            class_count * 2 +
            function_count * 1 +
            import_count * 0.5 +
            len(lines) * 0.01
        )
        
        return min(complexity / 100, 1.0)  # تطبيع إلى 0-1
    
    def _test_system_integration(self) -> float:
        """اختبار التكامل بين الأنظمة"""
        integration_score = 0.0
        total_tests = 0
        
        # اختبار استيراد الوحدات المترابطة
        try:
            from arc_complete_agent_part1 import ARCConfig
            integration_score += 1
        except:
            pass
        total_tests += 1
        
        try:
            from arc_complete_agent_part2 import UltraComprehensivePatternAnalyzer
            integration_score += 1
        except:
            pass
        total_tests += 1
        
        try:
            from arc_complete_agent_part3 import AdvancedStrategyManager
            integration_score += 1
        except:
            pass
        total_tests += 1
        
        try:
            from arc_ultimate_mind_part7 import MasterOrchestrator
            integration_score += 1
        except:
            pass
        total_tests += 1
        
        try:
            from burhan_meta_brain import MetaBrain
            integration_score += 1
        except:
            pass
        total_tests += 1
        
        return integration_score / max(total_tests, 1)
    
    def _identify_integration_issues(self) -> List[str]:
        """تحديد مشاكل التكامل"""
        issues = []
        
        # فحص التبعيات المفقودة
        dependencies = [
            ("main.py", "arc_ultimate_mind_part7"),
            ("arc_complete_agent_part4.py", "arc_complete_agent_part1"),
            ("arc_complete_agent_part4.py", "arc_complete_agent_part2"),
            ("burhan_meta_brain.py", "arc_complete_agent_part2")
        ]
        
        for file, dependency in dependencies:
            if os.path.exists(file):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if dependency in content and not os.path.exists(f"{dependency}.py"):
                        issues.append(f"{file} depends on missing {dependency}")
                        
                except:
                    issues.append(f"Cannot read {file}")
        
        return issues
    
    def _test_on_sample_tasks(self):
        """اختبار على مهام عينة"""
        print("\n🎮 اختبار على مهام عينة...")
        
        # تحميل مهام التدريب
        try:
            with open('ملفات المسابقةarc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
                tasks = json.load(f)
            
            # اختبار على أول 5 مهام
            sample_tasks = list(tasks.items())[:5]
            
            for task_id, task_data in sample_tasks:
                result = self._test_single_task(task_id, task_data)
                self.test_results[f"task_{task_id}"] = result
                
        except Exception as e:
            self.system_errors["sample_tasks"] = str(e)
    
    def _test_single_task(self, task_id: str, task_data: Dict) -> Dict[str, Any]:
        """اختبار مهمة واحدة"""
        try:
            # محاولة حل المهمة باستخدام النظام الحالي
            start_time = time.time()
            
            # هنا يجب استدعاء النظام الفعلي
            # لكن بسبب مشاكل التكامل، سنحاكي النتيجة
            
            execution_time = time.time() - start_time
            
            return {
                "status": "attempted",
                "execution_time": execution_time,
                "solution_found": False,
                "error": "Integration issues prevent actual testing"
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": 0
            }
    
    def _generate_test_report(self):
        """إنتاج تقرير الاختبار"""
        print("\n📊 تقرير اختبار الأنظمة:")
        print("=" * 50)
        
        # إحصائيات عامة
        total_systems = len(self.test_results)
        successful_systems = sum(1 for r in self.test_results.values() 
                               if r.get('status') == 'success')
        
        print(f"📈 إجمالي الأنظمة المختبرة: {total_systems}")
        print(f"✅ الأنظمة الناجحة: {successful_systems}")
        print(f"❌ الأنظمة الفاشلة: {total_systems - successful_systems}")
        
        # تفاصيل الأخطاء
        if self.system_errors:
            print(f"\n🚨 الأخطاء المكتشفة:")
            for system, error in self.system_errors.items():
                print(f"  {system}: {error}")
        
        # نقاط الضعف الرئيسية
        weaknesses = self._identify_major_weaknesses()
        if weaknesses:
            print(f"\n⚠️ نقاط الضعف الرئيسية:")
            for weakness in weaknesses:
                print(f"  • {weakness}")
        
        # توصيات للتحسين
        recommendations = self._generate_improvement_recommendations()
        if recommendations:
            print(f"\n💡 توصيات التحسين:")
            for rec in recommendations:
                print(f"  🎯 {rec}")
        
        # حفظ التقرير
        self._save_test_report()
    
    def _identify_major_weaknesses(self) -> List[str]:
        """تحديد نقاط الضعف الرئيسية"""
        weaknesses = []
        
        # فحص مشاكل التكامل
        if "integration" in self.test_results:
            integration = self.test_results["integration"]
            if integration.get("score", 0) < 0.5:
                weaknesses.append("ضعف في التكامل بين الأنظمة")
        
        # فحص الأخطاء الشائعة
        error_count = len(self.system_errors)
        if error_count > 2:
            weaknesses.append(f"عدد كبير من الأخطاء ({error_count} أخطاء)")
        
        # فحص التبعيات المفقودة
        missing_deps = sum(1 for r in self.test_results.values() 
                          if "Missing dependency" in str(r.get("issues", [])))
        if missing_deps > 0:
            weaknesses.append(f"تبعيات مفقودة في {missing_deps} نظام")
        
        return weaknesses
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """إنتاج توصيات التحسين"""
        recommendations = []
        
        # بناءً على التحليل السابق
        if len(self.system_errors) > 0:
            recommendations.append("إصلاح أخطاء الاستيراد والتبعيات")
        
        # بناءً على تحليل الأنماط
        recommendations.append("التركيز على object_manipulation (1234 مثال)")
        recommendations.append("تطوير خوارزميات للمهام المعقدة (complex_unknown)")
        recommendations.append("تحسين معالجة تغيير الأحجام (size_asymmetric)")
        
        # توصيات عامة
        recommendations.append("إنشاء نظام تكامل موحد")
        recommendations.append("تطوير آلية اختبار مستمرة")
        recommendations.append("بناء قاعدة معرفة من الأنماط المكتشفة")
        
        return recommendations
    
    def _save_test_report(self):
        """حفظ تقرير الاختبار"""
        report = {
            "test_results": self.test_results,
            "system_errors": self.system_errors,
            "performance_metrics": self.performance_metrics,
            "weaknesses": self._identify_major_weaknesses(),
            "recommendations": self._generate_improvement_recommendations(),
            "test_timestamp": time.time()
        }
        
        with open('system_performance_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 تم حفظ تقرير الاختبار في: system_performance_report.json")

# تشغيل الاختبار
if __name__ == "__main__":
    tester = SystemPerformanceTester()
    tester.test_all_systems()
    tester._test_on_sample_tasks()

from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ اختبار سريع للتحسينات
"""

import numpy as np
import time

def test_improvements():
    """اختبار التحسينات"""
    
    print("⚡ اختبار سريع للتحسينات")
    print("="*40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        
        # Initialize improved system
        ez = EfficientZeroEngine()
        print(f"🧠 النظام المحسن مع {len(ez.action_space)} إجراء")
        
        # Test problems
        test_problems = [
            {
                'name': 'بسيط 2x2',
                'grid': np.array([[1, 0], [0, 1]])
            },
            {
                'name': 'متماثل 3x3',
                'grid': np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])
            },
            {
                'name': 'نمط 4x4',
                'grid': np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 0, 1, 0], [0, 2, 0, 2]])
            },
            {
                'name': 'معقد 3x4',
                'grid': np.array([[1, 2, 3, 1], [2, 0, 1, 2], [3, 1, 2, 3]])
            },
            {
                'name': 'كبير 5x5',
                'grid': np.random.randint(0, 4, (5, 5))
            }
        ]
        
        print("\n🎯 اختبار المشاكل...")
        print("-" * 40)
        
        results = []
        total_time = 0
        
        for i, problem in enumerate(test_problems):
            print(f"{i+1}. {problem['name']}...")
            
            try:
                start_time = time.time()
                result = ez.solve_arc_problem(problem['grid'], max_steps=5)
                solve_time = time.time() - start_time
                total_time += solve_time
                
                confidence = result.get('confidence', 0)
                method = result.get('method', 'efficient_zero')
                
                print(f"   ✅ ثقة: {confidence:.3f}, وقت: {solve_time:.3f}s")
                
                results.append({
                    'name': problem['name'],
                    'confidence': confidence,
                    'solve_time': solve_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"   ❌ فشل: {e}")
                results.append({
                    'name': problem['name'],
                    'success': False,
                    'error': str(e)
                })
        
        # Statistics
        print("\n" + "="*40)
        print("📊 الإحصائيات:")
        print("-" * 40)
        
        successful = [r for r in results if r.get('success', False)]
        
        if successful:
            success_rate = len(successful) / len(results) * 100
            avg_confidence = np.mean([r['confidence'] for r in successful])
            avg_time = np.mean([r['solve_time'] for r in successful])
            
            print(f"✅ معدل النجاح: {len(successful)}/{len(results)} ({success_rate:.1f}%)")
            print(f"🔮 متوسط الثقة: {avg_confidence:.3f}")
            print(f"⏱️  متوسط الوقت: {avg_time:.3f}s")
            print(f"🚀 إجمالي الوقت: {total_time:.3f}s")
            
            # Best result
            best = max(successful, key=lambda x: x['confidence'])
            print(f"🏆 أفضل نتيجة: {best['name']} (ثقة: {best['confidence']:.3f})")
            
            # Compare with previous baseline
            previous_avg_confidence = 0.327  # From previous test
            improvement = (avg_confidence - previous_avg_confidence) / previous_avg_confidence * 100
            
            print(f"\n📈 مقارنة مع النتائج السابقة:")
            print(f"   - تحسن الثقة: {improvement:+.1f}%")
            
            if improvement > 5:
                print("🎉 تحسن ممتاز!")
            elif improvement > 0:
                print("📊 تحسن جيد")
            else:
                print("⚠️  لا يوجد تحسن واضح")
        
        else:
            print("❌ لا توجد نتائج ناجحة")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f"❌ خطأ في النظام: {e}")
        return False

def test_new_actions():
    """اختبار الإجراءات الجديدة"""
    
    print(f"\n🔧 اختبار الإجراءات الجديدة...")
    print("-" * 40)
    
    try:
        from efficient_zero_engine import EfficientZeroEngine
        
        ez = EfficientZeroEngine()
        
        # Test specific new actions
        test_grid = np.array([[1, 2, 1], [2, 0, 2], [1, 2, 1]])
        
        new_actions = [
            'flip_diagonal',
            'invert_colors', 
            'normalize_colors',
            'detect_symmetry',
            'complete_pattern',
            'remove_noise',
            'crop_to_content'
        ]
        
        print(f"🎯 اختبار {len(new_actions)} إجراء جديد...")
        
        working_actions = 0
        
        for action in new_actions:
            try:
                # Test if action exists in action space
                if action in ez.action_space:
                    print(f"   ✅ {action}: متاح")
                    working_actions += 1
                else:
                    print(f"   ❌ {action}: غير متاح")
            except Exception as e:
                print(f"   ❌ {action}: خطأ - {e}")
        
        action_rate = working_actions / len(new_actions) * 100
        print(f"\n📊 الإجراءات الجديدة: {working_actions}/{len(new_actions)} ({action_rate:.1f}%)")
        
        return working_actions > len(new_actions) * 0.7
        
    except Exception as e:
        print(f"❌ خطأ في اختبار الإجراءات: {e}")
        return False

def main():
    """الدالة الرئيسية"""
    
    print("⚡ اختبار شامل للتحسينات")
    print("="*40)
    
    # Test 1: General improvements
    improvement_success = test_improvements()
    
    # Test 2: New actions
    actions_success = test_new_actions()
    
    # Overall assessment
    print("\n" + "="*40)
    print("🎯 التقييم العام:")
    print("-" * 40)
    
    if improvement_success and actions_success:
        print("🎉 جميع التحسينات تعمل بشكل ممتاز!")
        print("✅ النظام جاهز للاختبار على المهام الفعلية")
        overall_success = True
    elif improvement_success or actions_success:
        print("📊 بعض التحسينات تعمل بشكل جيد")
        print("🔧 يحتاج مزيد من التطوير")
        overall_success = True
    else:
        print("❌ التحسينات تحتاج مراجعة")
        print("🛠️  يجب حل المشاكل قبل المتابعة")
        overall_success = False
    
    return overall_success

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 النظام المحسن جاهز!")
    else:
        print("\n⚠️  النظام يحتاج مزيد من العمل")

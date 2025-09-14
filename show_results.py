from __future__ import annotations
import json

with open('complete_test_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('='*80)
print('🎯 COMPLETE TEST RESULTS - ALL SYSTEMS ON 1001 ARC TASKS')
print('='*80)
print(f'\n📊 Test Overview:')
print(f'   • Total Tasks Tested: {data["task_count"]}')
print(f'   • Total Systems: {data["system_count"]}')
print(f'   • Total Tests Run: {data["total_tests"]}')
print(f'   • Test Files: {", ".join(data["task_files"])}')

print('\n' + '='*80)
print('🏆 SYSTEM PERFORMANCE RANKING:')
print('='*80)

for i, system in enumerate(data['systems'], 1):
    print(f'\n{i}. {system["name"]}')
    print(f'   ✅ Success Rate: {system["success_rate"]*100:.1f}%')
    print(f'   📈 Tasks Solved: {system["solved"]}/{system["total"]}')
    print(f'   ⏱️  Average Time: {system["avg_time"]*1000:.1f}ms')
    
    if system["errors"] > 0:
        print(f'   ⚠️  Errors: {system["errors"]}')
    if system["timeouts"] > 0:
        print(f'   ⏳ Timeouts: {system["timeouts"]}')

print('\n' + '='*80)
print('🤝 SYSTEM AGREEMENT ANALYSIS:')
print('='*80)

# Sort agreements by count
agreements = sorted(data['agreements'].items(), key=lambda x: x[1], reverse=True)
print('\nTop System Agreements (similar solutions):')
for pair, count in agreements[:10]:
    print(f'   • {pair}: {count} tasks ({count/data["task_count"]*100:.1f}%)')

print('\n' + '='*80)
print('📈 SUMMARY:')
print('='*80)

# Best system
best = data['systems'][0]
print(f'\n🥇 BEST SYSTEM: {best["name"]}')
print(f'   • Success Rate: {best["success_rate"]*100:.1f}%')
print(f'   • Speed: {best["avg_time"]*1000:.1f}ms per task')

# Fastest system
fastest = min(data['systems'], key=lambda x: x['avg_time'])
print(f'\n⚡ FASTEST SYSTEM: {fastest["name"]}')
print(f'   • Average Time: {fastest["avg_time"]*1000:.1f}ms per task')
print(f'   • Success Rate: {fastest["success_rate"]*100:.1f}%')

print('\n' + '='*80)

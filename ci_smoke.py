"""Быстрая проверка, что весь код импортируется и базовые тесты проходят.
Запуск: python ci_smoke.py
"""
import subprocess
import sys

print("🔍 Запускаю быстрые тесты для всех глав...")
print("=" * 60)

# Список глав с тестами
chapters = [
    "chapter1/test_agent.py",
    # Добавь сюда главы по мере написания:
    # "chapter2/test_paraphraser.py",
    # "chapter3/test_agent.py",
]

failed = []
for test_file in chapters:
    print(f"\n📦 Тестирую {test_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        failed.append(test_file)
        print(f"❌ {test_file} провалился")
        print(result.stdout)
    else:
        print(f"✅ {test_file} прошёл")

print("\n" + "=" * 60)
if failed:
    print(f"❌ Провалено: {len(failed)}")
    for f in failed:
        print(f"   - {f}")
    sys.exit(1)
else:
    print("✅ Все тесты прошли")
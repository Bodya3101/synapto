import time
import random
import string
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from .engine import SynaptoEngine
from .queue import ChatStreamProcessor


class ProceduralFactGenerator:
    """
    Процедурный генератор уникальных фактов по 6 категориям (включая личный профиль).
    """
    FIRST_NAMES = ["Александр", "Дмитрий", "Елена", "Михаил", "София", "Артем", "Виктория", "Игорь"]
    LAST_NAMES = ["Ковалев", "Смирнов", "Соколов", "Морозов", "Волков", "Васильев", "Зайцев"]
    MONTHS = ["января", "марта", "мая", "июля", "сентября", "ноября"]
    PROTOCOLS = ["TLS-1.3-ChaCha20", "AES-256-GCM", "WSS-SECURE", "HTTPS-ECDHE"]
    PROJECT_NAMES = ["Horizon", "CloudLog", "NervOS", "Nexus", "CoreSystem", "DataVault", "Aether"]

    # Данные для категории PersonalProfile
    USER_NAMES = ["Бодя", "Алексей", "Максим", "Денис", "Роман"]
    PET_EVENTS = ["рожала кошка", "завели щенка", "купили попугая", "построили домик для кота"]
    FAVORITE_DISHES = ["пельмени", "пицца пепперони", "паста карбонара", "борщ"]

    @classmethod
    def _random_str(cls, length: int = 6) -> str:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    @classmethod
    def generate_fact_by_category(cls, category: str) -> Tuple[str, str, str]:
        project = random.choice(cls.PROJECT_NAMES) + "_" + cls._random_str(3)
        
        if category == "Credentials":
            prompt = f"API ключ сервиса {project}:"
            completion = f" sk-prod-{cls._random_str(4)}-{cls._random_str(4)}."
        elif category == "Temporal":
            day = random.randint(1, 28)
            month = random.choice(cls.MONTHS)
            year = random.randint(2025, 2030)
            prompt = f"Дата релиза модуля {project}:"
            completion = f" {day} {month} {year} года."
        elif category == "Entities":
            name = f"{random.choice(cls.FIRST_NAMES)} {random.choice(cls.LAST_NAMES)}"
            prompt = f"Главный архитектор системы {project}:"
            completion = f" {name}."
        elif category == "Networking":
            port = random.randint(8000, 9999)
            proto = random.choice(cls.PROTOCOLS)
            prompt = f"Порт и протокол сервиса {project}:"
            completion = f" {port}-{proto}."
        elif category == "ConfigStrings":
            db = cls._random_str(4)
            prompt = f"Строка подключения БД {project}:"
            completion = f" postgresql://admin:{cls._random_str(6)}@localhost:5432/{db}."
        elif category == "PersonalProfile":
            choice_type = random.choice(["name", "age", "pet", "dish"])
            if choice_type == "name":
                prompt = "Как меня зовут?"
                completion = f" {random.choice(cls.USER_NAMES)}."
            elif choice_type == "age":
                age = random.randint(17, 35)
                prompt = "Сколько мне лет?"
                completion = f" {age} лет."
            elif choice_type == "pet":
                prompt = "Что произошло вчера с моим питомцем?"
                completion = f" Вчера {random.choice(cls.PET_EVENTS)}."
            else:
                prompt = "Какое мое любимое блюдо?"
                completion = f" {random.choice(cls.FAVORITE_DISHES)}."
        else:
            raise ValueError(f"Неизвестная категория: {category}")

        return prompt, completion, category


class RealisticNoiseGenerator:
    """
    Процедурный генератор мульти-доменного шума для забивания KV-кэша.
    """
    NOISE_TEMPLATES = [
        ("Какая обстановка по задачам в Jira?", " Все приоритетные тикеты закрыты, тестируем релизную сборку."),
        ("Во сколько начинается общее собрание?", " Синхронизация команды запланирована на 15:00 в главной переговорке."),
        ("Какая текущая загрузка процессора?", " Загрузка CPU на узле cluster-01 составляет 34 процента."),
        ("Что сказали на презентации нового проекта?", " Докладчик рассказал об основных этапах масштабирования инфраструктуры."),
        ("Какой статус контейнеров в Kubernetes?", " Все под-контейнеры находятся в состоянии Running, ошибок нет.")
    ]

    @classmethod
    def get_noise_turn(cls) -> Tuple[str, str]:
        return random.choice(cls.NOISE_TEMPLATES)


class SWERecallBenchmark:
    """
    Профессиональный бенчмарк оценщик качества динамической памяти по 6 категориям.
    """
    CATEGORIES = ["Credentials", "Temporal", "Entities", "Networking", "ConfigStrings", "PersonalProfile"]

    def __init__(self, engine: SynaptoEngine, window_tokens: int = 30):
        self.engine = engine
        self.processor = ChatStreamProcessor(engine, max_window_tokens=window_tokens)

    def run_benchmark(self, facts_per_category: int = 2) -> Dict[str, Any]:
        """
        Генерирует уникальный датасет и прогоняет тест с разшивкой по 6 категориям.
        """
        start_time = time.time()
        
        generated_facts: List[Tuple[str, str, str]] = []
        for cat in self.CATEGORIES:
            for _ in range(facts_per_category):
                generated_facts.append(ProceduralFactGenerator.generate_fact_by_category(cat))

        random.shuffle(generated_facts)
        total_facts_count = len(generated_facts)

        # 1. Потоковая зашумленная эвикция
        for prompt, completion, category in generated_facts:
            self.processor.process_turn(prompt, completion)
            noise_prompt, noise_completion = RealisticNoiseGenerator.get_noise_turn()
            self.processor.process_turn(noise_prompt, noise_completion)

        # 2. Оценка точности извлечения из весов на чистом контексте
        category_stats: Dict[str, Dict[str, int]] = {
            cat: {"passed": 0, "total": 0} for cat in self.CATEGORIES
        }
        detailed_results = []
        total_passed = 0

        for prompt, expected_completion, category in generated_facts:
            response = self.engine.generate_response(prompt)
            target_clean = expected_completion.strip().rstrip('.')
            
            is_success = target_clean in response
            category_stats[category]["total"] += 1
            
            if is_success:
                category_stats[category]["passed"] += 1
                total_passed += 1

            detailed_results.append({
                "category": category,
                "prompt": prompt,
                "target": target_clean,
                "response": response,
                "passed": is_success
            })

        overall_accuracy = (total_passed / total_facts_count) * 100
        elapsed_seconds = time.time() - start_time

        category_breakdown = {}
        for cat, stats in category_stats.items():
            cat_acc = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
            category_breakdown[cat] = {
                "accuracy_rate": cat_acc,
                "passed": stats["passed"],
                "total": stats["total"]
            }

        return {
            "overall_accuracy_rate": overall_accuracy,
            "total_passed": total_passed,
            "total_facts": total_facts_count,
            "elapsed_seconds": elapsed_seconds,
            "category_breakdown": category_breakdown,
            "details": detailed_results,
            "memory_journal": self.engine.get_memory_dump()
        }

    def export_report_markdown(self, benchmark_results: Dict[str, Any], filepath: str = "benchmark_report.md") -> None:
        """
        Сохраняет красивый Markdown-отчет о результатах теста по 6 категориям.
        """
        md = f"# Synapto SWE-Recall-Bench Report\n\n"
        md += f"**Overall Accuracy Rate:** {benchmark_results['overall_accuracy_rate']:.2f}%\n"
        md += f"**Total Passed:** {benchmark_results['total_passed']} / {benchmark_results['total_facts']}\n"
        md += f"**Execution Time:** {benchmark_results['elapsed_seconds']:.2f} seconds\n\n"
        
        md += "## Category Breakdown\n\n"
        md += "| Category | Accuracy | Passed / Total |\n"
        md += "| :--- | :---: | :---: |\n"
        for cat, stats in benchmark_results["category_breakdown"].items():
            md += f"| **{cat}** | {stats['accuracy_rate']:.1f}% | {stats['passed']} / {stats['total']} |\n"
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)
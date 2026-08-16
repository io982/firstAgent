"""Плагины для работы с векторной базой знаний."""

import hashlib
from datetime import datetime

from chapter4.src import tools as chapter4_tools
from .vectorstore import add_document, search_documents, list_documents


def _generate_id(text: str) -> str:
    """Генерирует уникальный ID для документа."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


@chapter4_tools.tool(
    "remember",
    "сохраняет важную информацию в долгосрочную память. Используй, когда пользователь просит запомнить факт, или когда узнаёшь новый факт о проекте/настройках/предпочтениях пользователя."
)
def remember(content: str = "", category: str = "general", **kwargs) -> str:
    # Fallback для выдуманных параметров
    if not content and kwargs:
        content = " | ".join(f"{k}: {v}" for k, v in kwargs.items())
    if not content:
        return "Ошибка: не указан текст для сохранения. Используй параметр content."
    # === ДЕДУПЛИКАЦИЯ: проверяем, нет ли уже очень похожей записи ===
    try:
        existing = search_documents(content, n_results=1)
        if existing:
            best_match = existing[0]
            # Если косинусное расстояние очень маленькое (< 0.1), значит это дубль
            if best_match["distance"] < 0.1:
                return f"Эта информация уже есть в памяти (ID: {best_match['id']}). Повторное сохранение не требуется."
    except Exception:
        pass  # Если поиск упал, просто сохраняем как обычно
    doc_id = _generate_id(content + datetime.now().isoformat())
    metadata = {
        "category": category,
        "timestamp": datetime.now().isoformat()
    }
    add_document(doc_id, content, metadata)
    return f"Сохранено в памяти (ID: {doc_id}, категория: {category})"


@chapter4_tools.tool(
    "recall",
    "ищет информацию в долгосрочной памяти по ключевым словам или фразе. Используй, когда помнишь, ЧТО ищешь. Параметр query ОБЯЗАТЕЛЬНО должен быть непустым."
)
def recall(query: str, max_results: int = 3) -> str:
    """
    Ищет похожую информацию в векторной базе.
    """
    # Защита от пустого query
    if not query or not query.strip():
        return "Ошибка: не указан поисковый запрос. Используй параметр query с текстом для поиска."
    results = search_documents(query, n_results=max_results)
    if not results:
        return "Ничего не найдено в памяти."
    output = []
    for i, result in enumerate(results, 1):
        distance = result["distance"]
        confidence = "высокая" if distance < 0.3 else "средняя" if distance < 0.5 else "низкая"
        output.append(f"{i}. [{confidence} уверенность] {result['text']}")
        if result["metadata"].get("category"):
            output.append(f"   Категория: {result['metadata']['category']}")
    return "\n".join(output)


@chapter4_tools.tool(
    "list_memory",
    "показывает ВСЕ сохранённые документы в памяти без поиска. Используй, когда пользователь просит 'покажи что ты помнишь', 'что у тебя в памяти', 'вспомни всё'. После получения списка НЕ вызывай recall повторно — просто ответь пользователю на основе этих данных."
)
def list_memory(limit: int = 10) -> str:
    docs = list_documents(limit=limit)
    if not docs:
        return "Память пуста."
    output = [f"Найдено {len(docs)} записей в памяти:\n"]
    for i, doc in enumerate(docs, 1):
        category = doc["metadata"].get("category", "без категории")
        # Убираем обрезку, показываем полный текст
        output.append(f"{i}. [{category}] {doc['text']}")
    return "\n".join(output)

def auto_save_important_info(user_query: str, agent_response: str):
    """
    Автоматически сохраняет важную информацию из диалога.
    Вызывается после каждого ответа агента.
    """
    query_lower = user_query.lower()
    # Фразы, которые ПРОСЯТ ЗАПОМНИТЬ (сохраняем)
    save_keywords = [
        "запомни", "сохрани", "запиши",
        "remember", "save", "note"
    ]
    # Фразы, которые ПРОСЯТ ПОКАЗАТЬ ПАМЯТЬ (НЕ сохраняем!)
    skip_keywords = [
        "покажи", "что ты помнишь", "что помнишь", "вспомни",
        "покажи память", "list_memory", "что в памяти"
    ]
    # Если запрос содержит слова для показа памяти — пропускаем
    if any(kw in query_lower for kw in skip_keywords):
        return
    # Если пользователь явно просит запомнить
    if any(kw in query_lower for kw in save_keywords):
        content = f"Запрос: {user_query}\nОтвет: {agent_response}"
        remember(content, category="explicit_request")
        return
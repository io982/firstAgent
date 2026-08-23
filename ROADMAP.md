# 🚧 В разработке

# Глава 0. Выбор модели

**Статус: ✅ ГОТОВО** — [текст в README.md](README.md#глава-0-выбор-модели)

---

# Глава 1. Первый AI Agent — ReAct

**Статус: CORE**

Создаём первого локального агента.

## 1.1. LLM vs Agent

* Что такое агент
* Agent loop
* State
* Observation
* Action
* Tool

## 1.2. ReAct

* Reason
* Act
* Observe
* Iteration
* Stopping conditions

## 1.3. Первый агент

* Python
* Ollama API
* System prompt
* User prompt
* Message history
* Agent loop

## 1.4. Первые tools

* Calculator
* Date/time
* Простые Python functions

## 1.5. Ошибки

* Invalid action
* Tool errors
* Retry
* Timeout
* Maximum iterations

### Практический результат

Небольшая модель самостоятельно:

```
получает задачу
     ↓
   думает
     ↓
выбирает tool
     ↓
получает результат
     ↓
продолжает
     ↓
  отвечает
```

---

# Глава 2. Tool Calling

**Статус: CORE**

Переходим от имитации инструментов к нормальному tool calling.

## 2.1. Function calling

* Tool schema
* Arguments
* Tool result
* Native tool calls

## 2.2. Создание Tool API

* Python functions
* JSON Schema
* Validation
* Structured output

## 2.3. Реальные tools

* Filesystem
* HTTP
* Python
* Git
* Database

## 2.4. Tool orchestration

* Несколько tools
* Tool selection
* Tool chaining
* Parallel tools

## 2.5. Безопасность

* Permissions
* Allowlist
* Timeout
* Ограничение команд

### Практический результат

```
Agent
  │
  ├── calculator
  ├── filesystem
  ├── HTTP
  ├── Python
  └── database
```

---

# Глава 3. Context и Memory

**Статус: CORE**

Учимся управлять контекстом небольших моделей.

## 3.1. Context window

* Tokens
* Context budget
* Что занимает контекст
* Context overflow

## 3.2. Short-term memory

* Conversation history
* Agent state
* Scratchpad

## 3.3. Context management

* Trimming
* Summarization
* Compression
* Selective history

## 3.4. Long-term memory

* Facts
* Preferences
* Previous tasks
* Memory retrieval

## 3.5. Performance

* Latency
* Token throughput
* Streaming
* Caching

### Главная идея

Не заставляем маленькую модель держать всё в контексте:

```
Большая информация
      ↓
   Memory
      ↓
relevant context
      ↓
   Small LLM
```

---

# Глава 4. RAG

**Статус: CORE**

Добавляем внешнюю базу знаний.

## 4.1. RAG

* Retrieval-Augmented Generation
* Почему знания лучше хранить вне модели

## 4.2. Embeddings

* Embedding models
* Vector representation
* Similarity
* Cosine similarity

## 4.3. Vector databases

* FAISS
* Chroma
* Qdrant
* SQLite-based варианты

## 4.4. Document pipeline

* Parsing
* Chunking
* Embedding
* Indexing
* Retrieval

## 4.5. RAG Agent

```
Documents
   ↓
Embeddings
   ↓
Vector DB
   ↓
Retrieval
   ↓
Relevant context
   ↓
Small LLM
   ↓
Answer
```

### Особое внимание

RAG позволяет небольшой модели работать с большим объёмом знаний без необходимости использовать огромную LLM.

---

# Глава 5. Code RAG

**Статус: CORE**

Создаём агента, который понимает код проекта.

## 5.1. Индексация проекта

* Python
* JavaScript / TypeScript
* Markdown
* Config files

## 5.2. Code chunking

* Functions
* Classes
* Modules
* Imports

## 5.3. Code embeddings

* Semantic code search
* Code retrieval

## 5.4. Repository understanding

* Project structure
* Dependencies
* Architecture

## 5.5. Code assistant

* Найти код
* Объяснить код
* Найти зависимости
* Найти потенциальные проблемы

### Практический результат

```
User
  ↓
Code Agent
  ↓
Code Search
  ↓
Relevant files
  ↓
Small Coding LLM
  ↓
Answer
```

---

# Глава 6. Hybrid Search

**Статус: CORE**

Улучшаем RAG.

## 6.1. Keyword search

* Exact match
* BM25
* Symbols
* Function names

## 6.2. Vector search

* Semantic similarity

## 6.3. Hybrid retrieval

* Keyword + vector
* Score fusion

## 6.4. Reranking

* Зачем нужен reranker
* Cross-encoder
* Local reranking

## 6.5. Улучшенный Code RAG

```
Query
  │
  ├── BM25
  │
  └── Vector Search
         │
         ▼
      Fusion
         │
         ▼
      Reranker
         │
         ▼
   Relevant chunks
         │
         ▼
        LLM
```

### Почему глава важна для слабого железа

Большую часть работы здесь выполняют поисковые алгоритмы, а не LLM. Поэтому качество системы можно сильно повысить без перехода на огромную модель.

---

# Глава 7. Multi-Agent Systems

**Статус: CORE**

Разбираемся, как создавать несколько специализированных агентов.

## 7.1. Зачем нужны агенты

* Специализация
* Делегирование
* Изоляция контекста
* Разные инструменты

## 7.2. Архитектуры

* Manager / Worker
* Router / Specialist
* Sequential agents
* Parallel agents
* Reviewer / Executor

## 7.3. Специалисты

* Researcher
* Coder
* Analyst
* Planner
* Reviewer

## 7.4. Один LLM → несколько агентов

На слабом железе:

```
Small LLM
   │
   ├── Researcher
   ├── Coder
   ├── Analyst
   └── Reviewer
```

Каждый агент получает:

* свою роль;
* свой system prompt;
* свой набор tools;
* свой контекст.

## 7.5. Model routing

На мощном железе:

```
Router
  │
  ├── Small model
  ├── General model
  ├── Reasoning model
  └── Coding model
```

---

# Глава 8. Coding Agent

**Статус: CORE**

Создаём практического агента для разработки.

## 8.1. Filesystem

* Read
* Write
* Edit
* Search

## 8.2. Git

* Status
* Diff
* Log
* Branch
* Commit

## 8.3. Execution

* Run code
* Tests
* Linters
* Capture errors

## 8.4. Coding workflow

```
User
  ↓
Planner
  ↓
Code Agent
  ↓
Read
  ↓
Edit
  ↓
Test
  ↓
Error
  ↓
Fix
  ↓
Test again
```

## 8.5. Coding models

* Small coding models
* Qwen Coder
* Другие coding models

Большие coding models:

* Qwen3-Coder
* Devstral
* Другие 14B+ модели

рассматриваются как Advanced.

---

# Глава 9. Reasoning и Planning

**Статус: CORE — архитектура / ADVANCED — тяжёлые модели**

## 9.1. Reasoning models

* Что такое reasoning
* Thinking vs обычная генерация
* Когда reasoning нужен

## 9.2. Planning

* Task decomposition
* Subtasks
* Dependencies
* Execution plan

## 9.3. Planner + Executor

```
Planner
   ↓
 Plan
   ↓
Executor
   ↓
Results
   ↓
Verifier
```

## 9.4. Reasoning models

Изучаем:

* DeepSeek-R1
* Qwen reasoning models
* Другие reasoning models

Но запуск больших версий не является обязательным.

## 9.5. Verification

* Self-check
* Critic
* Test-based verification
* Independent verification

---

# Глава 10. Vision и Multimodal Agents

**Статус: OPTIONAL / ADVANCED**

Глава не требуется для прохождения основного пути.

## 10.1. Vision

* Images
* Screenshots
* OCR
* Documents

## 10.2. Vision + tools

* Image understanding
* Action selection
* Tool calling

## 10.3. Document agents

* PDF
* Scanned documents
* Tables
* Forms

## 10.4. Computer-use agents

* Screenshot
* Mouse
* Keyboard
* Browser

## 10.5. Vision architecture

```
Image
  ↓
Vision model
  ↓
Agent
  ↓
Tool
```

### Модели

* Gemma
* Qwen-VL
* LLaVA
* Другие vision models

Большие vision models — Advanced.

---

# Глава 11. Advanced Multi-Agent Systems

**Статус: ADVANCED**

Объединяем всё изученное.

## 11.1. Dynamic Model Routing

* Model selection
* Quality / latency trade-off
* Fallback models

## 11.2. Hierarchical agents

* Manager
* Team lead
* Workers

## 11.3. Parallel agents

* Parallel research
* Parallel analysis
* Result aggregation

## 11.4. Critic architectures

* Generator
* Critic
* Reviewer
* Finalizer

## 11.5. Multi-model architecture

```
Manager
   │
   ├── Researcher → General model
   ├── Coder → Coding model
   ├── Analyst → Reasoning model
   └── Vision → Vision model
```

---

# Глава 12. Security

**Статус: CORE**

Безопасность агента обязательна, особенно если он получает доступ к компьютеру.

## 12.1. Threat model

* Prompt injection
* Indirect prompt injection
* Malicious documents
* Malicious web pages
* Tool abuse

## 12.2. Tool security

* Permissions
* Sandboxing
* Capability-based access
* Allowlists

## 12.3. Data security

* API keys
* Secrets
* Environment variables
* Sensitive files

## 12.4. Execution security

* Shell isolation
* Containers
* Resource limits
* Network restrictions

## 12.5. Secure agent architecture

```
Agent
  ↓
Policy Layer
  ↓
Permission Check
  ↓
Sandbox
  ↓
Tool
```

---

# Глава 13. Evaluation

**Статус: CORE**

Учимся измерять качество агента.

## 13.1. Metrics

* Task success
* Accuracy
* Reliability
* Latency
* Token usage

## 13.2. Tool evaluation

* Правильный tool
* Правильные arguments
* Tool error handling

## 13.3. Model comparison

Например:

```
Model A
   vs
Model B
   vs
Model C
```

на одинаковом наборе agent tasks.

## 13.4. Agent traces

* Steps
* Tool calls
* Errors
* Retries
* Final answer

## 13.5. Regression tests

* Набор задач
* Автоматический запуск
* Сравнение результатов

---

# Глава 14. Production Architecture

**Статус: ADVANCED**

Переходим от учебного агента к реальной системе.

## 14.1. Agent server

* API
* Authentication
* Sessions

## 14.2. Model server

* Ollama
* Multiple models
* Model lifecycle

## 14.3. Storage

* PostgreSQL
* Vector database
* Object storage

## 14.4. Observability

* Logs
* Metrics
* Traces
* Token usage
* Latency

## 14.5. Reliability

* Retries
* Timeouts
* Circuit breakers
* Fallback models

## 14.6. Scaling

* GPU workers
* Queues
* Parallel agents
* Multiple model instances

---

# Финальный проект

## Level 1 — Local Agent

**Статус: CORE**

Проект должен работать на слабом или среднем компьютере.

Минимальная архитектура:

```
User
  ↓
Agent
  ↓
Small LLM
  │
  ├── Tools
  ├── Memory
  ├── RAG
  ├── Hybrid Search
  └── Code Search
```

Возможности:

* ReAct
* Tool calling
* Filesystem
* Python
* Git
* Memory
* RAG
* Code RAG
* Hybrid search
* Basic coding agent
* Security
* Evaluation

---

# Level 2 — Multi-Agent Platform

**Статус: ADVANCED**

На более мощном железе:

```
User
  ↓
Manager
  ↓
Router
  │
  ├── Small model
  ├── General model
  ├── Coding model
  ├── Reasoning model
  └── Vision model
         │
         ▼
       Tools
         │
         ▼
        RAG
         │
         ▼
      Memory
         │
         ▼
     Verification
```

Добавляются:

* Dynamic model routing
* Multiple specialized models
* Reasoning models
* Coding models
* Vision
* Parallel agents
* Hierarchical agents
* Advanced verification

---

# Итоговая траектория

```
Глава 0
Выбор модели
    │
    ▼
Глава 1
ReAct Agent
    │
    ▼
Глава 2
Tool Calling
    │
    ▼
Глава 3
Context + Memory
    │
    ▼
Глава 4
RAG
    │
    ▼
Глава 5
Code RAG
    │
    ▼
Глава 6
Hybrid Search
    │
    ▼
Глава 7
Multi-Agent
    │
    ▼
Глава 8
Coding Agent
    │
    ▼
Глава 9
Reasoning + Planning
    │
    ├──────────────► Глава 10
    │                Vision
    │                [OPTIONAL]
    │
    ▼
Глава 11
Advanced Multi-Agent
    │
    ▼
Глава 12
Security
    │
    ▼
Глава 13
Evaluation
    │
    ▼
Глава 14
Production
    │
    ▼
┌─────────────────────────┐
│     FINAL PROJECT       │
│                         │
│ Local Agent Platform    │
│          +              │
│ Advanced Multi-Agent    │
└─────────────────────────┘
```

# Обозначения

**CORE** — необходимо пройти. Практические задания рассчитаны на небольшие локальные модели.

**OPTIONAL** — полезно, но не обязательно для основного курса.

**ADVANCED** — следующий уровень для мощного железа или удалённых GPU.

# Главный принцип курса

Не:

```
"Нужна большая модель → значит нужен мощный компьютер"
```

А:

```
"Сначала строим хорошую архитектуру"

             ↓

Small LLM + Tools + RAG + Memory
             ↓
          хороший
           Agent
             ↓
   при необходимости
             ↓
заменяем модель на более мощную
```

Архитектура агента должна быть независима от размера конкретной LLM.

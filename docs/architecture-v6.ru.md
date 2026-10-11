# Архитектура GhostSwitch v6

> Реестр экспертов теперь реализован пакетами `experts/qwen35-9b/<expert_id>/`. Старые пути и примеры ниже описывают этап проектирования/миграции. Актуальный стандарт: [experts/README.md](../experts/README.md), управление библиотекой и запуск: [reference.md](reference.md). Выбор специальностей сохраняется на сервере; Auto использует включённые и подготовленные адаптеры.


Документ описывает целевую архитектуру GhostSwitch v6, основанную на упрощении runtime v5.

Главное изменение v6: **GhostSwitch больше не является полноценным agent harness и не владеет workflow задачи.** Он становится специализированным model intelligence layer поверх одной общей базовой модели и набора LoRA-экспертов.

Ключевой принцип:

> **Harness owns workflow. GhostSwitch owns expertise selection.**

Внешний harness решает, **что делать дальше**. GhostSwitch решает, **какой эксперт лучше всего выполнит текущий inference-step**.

---

## 1. Что представляет собой v6

GhostSwitch v6 — это inference/runtime слой поверх общей базовой модели и набора LoRA-экспертов.

Для внешнего harness GhostSwitch должен выглядеть почти как одна обычная модель: harness передаёт текущий запрос или текущий шаг workflow, а GhostSwitch сам выбирает специализацию, активирует нужную LoRA, выполняет генерацию, проверяет завершённость результата и возвращает ответ.

V6 **не строит DAG**, **не планирует цепочки агентов**, **не управляет зависимостями между экспертами**, **не вызывает следующего эксперта сам** и **не выполняет tool loop**.

Упрощённый путь запроса:

```text
Request
  ↓
Request/API validation
  ↓
AUTO / DIRECT mode
  ↓
┌───────────────────────┐
│ AUTO                  │
│ Router выбирает       │
│ expert_id или base    │
└──────────┬────────────┘
           │
           │       ┌───────────────────────┐
           └──────→│ DIRECT                │
                   │ expert_id задан       │
                   │ внешним harness       │
                   └──────────┬────────────┘
                              ↓
                       Expert Registry
                              ↓
                         LoRA Manager
                              ↓
                    Base Model + LoRA
                              ↓
                          Inference
                              ↓
                           Verifier
                     ┌────────┼────────┐
                     │        │        │
                  ACCEPT    RETRY     FAIL
                     │        │        │
                     │   same expert   │
                     │   + feedback    │
                     └────────┴────────┘
                              ↓
                           Response
```

---

## 2. Граница ответственности

### 2.1. Что принадлежит внешнему Harness

Внешний harness отвечает за workflow и agent behavior:

- определяет следующий шаг задачи;
- решает, нужен ли ещё один inference-вызов;
- вызывает tools;
- работает с browser/shell/files/API;
- хранит agent/session state;
- управляет memory;
- выполняет planning и replanning;
- строит DAG, если он нужен;
- решает зависимости между шагами;
- решает, когда вызвать GhostSwitch повторно;
- передаёт результат одного шага в следующий;
- при необходимости вызывает GhostSwitch несколько раз для разных подзадач.

Пример:

```text
OpenClaw
  ↓
"Напиши реализацию функции"
  ↓
GhostSwitch AUTO
  ↓
Router → coding expert
  ↓
ответ
  ↓
OpenClaw решает выполнить тесты
  ↓
tool / test runner
  ↓
OpenClaw формирует новый шаг:
"Исправь найденную ошибку"
  ↓
GhostSwitch AUTO
  ↓
Router → debugging expert
```

GhostSwitch не должен знать, что это части одного agent workflow.

### 2.2. Что принадлежит GhostSwitch

GhostSwitch отвечает только за model specialization:

- принять текущий inference request;
- определить подходящую специализацию;
- выбрать expert_id;
- найти expert definition;
- проверить наличие и совместимость LoRA;
- активировать нужный адаптер;
- выполнить inference;
- проверить, можно ли принять результат;
- при необходимости выполнить ограниченный retry того же эксперта;
- вернуть результат и метаданные.

---

## 3. Основные компоненты GhostSwitch v6

```text
GhostSwitch Core
│
├── Request/API Layer
├── Router
├── Expert Registry
├── LoRA Manager
├── Inference Engine
├── Verifier
└── Response Adapter / Telemetry
```

Каждый компонент имеет узкую ответственность и не должен выполнять функции внешнего harness.

---

## 4. Request/API Layer

V6 должен иметь стабильный интерфейс, пригодный для подключения внешних harness.

Желательный внешний интерфейс — **OpenAI-compatible API** или совместимый по семантике Python API.

Минимальный внутренний контракт:

```json
{
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "mode": "auto",
  "expert_id": null,
  "max_tokens": 4096,
  "metadata": {}
}
```

### 4.1. AUTO mode

```json
{
  "mode": "auto",
  "messages": [...]
}
```

GhostSwitch запускает Router и сам выбирает экспертную специализацию.

### 4.2. DIRECT mode

```json
{
  "mode": "direct",
  "expert_id": "coding.python",
  "messages": [...]
}
```

Router пропускается. Указанный expert_id проверяется через Expert Registry, после чего выполняется inference.

DIRECT нужен для:

- внешних harness, которые хотят зафиксировать специализацию;
- тестирования конкретной LoRA;
- benchmark/evaluation;
- debugging;
- ручного вызова эксперта;
- deterministic pipelines.

AUTO остаётся основным режимом.

---

## 5. Router

Router остаётся **внутри GhostSwitch**.

Это принципиально важно: внешний harness не обязан знать, какие эксперты существуют, какие LoRA установлены и как устроена таксономия специализаций.

Router отвечает только на вопрос:

> **Какая специализация лучше всего подходит для текущего inference request?**

Router **не должен**:

- разбивать запрос на jobs;
- создавать DAG;
- задавать depends_on;
- строить multi-agent plan;
- создавать цепочки экспертов;
- определять downstream steps;
- выполнять workflow coverage;
- решать, какой эксперт будет вызван после текущего;
- управлять повторными вызовами разных экспертов.

### 5.1. Выход Router

Пример expert route:

```json
{
  "route": "expert",
  "expert_id": "coding.python",
  "confidence": 0.96
}
```

Пример base route:

```json
{
  "route": "base",
  "expert_id": null,
  "confidence": 0.91
}
```

Пример отсутствующей специализации:

```json
{
  "route": "unsupported",
  "expert_id": null,
  "missing_capability": "medical.radiology"
}
```

### 5.2. Правила Router

Сохраняются полезные принципы v5:

- специализация выбирается по требуемому результату, а не по случайным словам;
- учитывается доступность LoRA на текущем backend;
- expert_id обязан существовать в Expert Registry;
- adapter обязан быть совместим с текущей базовой моделью;
- malformed JSON Router может получить один ограниченный retry;
- при низкой уверенности допустим route `base` или `unsupported`, в зависимости от политики.

### 5.3. Что удаляется относительно v5

Удаляются:

```text
jobs[]
depends_on
routing_plan DAG
coverage recovery across full task
multi-job merge
missing work graph
needs_decomposition
max_experts_per_task
blocked downstream steps
```

Router больше не является planner.

---

## 6. Expert Registry

Expert Registry становится одной из центральных частей V6.

Эксперт — это логическая specialization-конфигурация:

```text
Expert =
    Base Model
  + LoRA Adapter
  + System Prompt
  + Capability Metadata
  + Runtime Compatibility Metadata
```

Пример:

```json
{
  "coding.python": {
    "display_name": "Python Coding Expert",
    "description": "Python implementation, debugging, FastAPI and backend development",
    "adapter": "python-v4",
    "base_model": "qwen35-9b",
    "prompt": "prompts/coding.python.txt",
    "capabilities": [
      "python",
      "fastapi",
      "backend",
      "debugging"
    ],
    "backends": ["mlx", "gguf"]
  }
}
```

### 6.1. Миграция из v5

В v5 эти данные уже частично существуют в нескольких местах:

```text
base/qwen35-9b/adapters.json
core/config/routing/v2_capability_dispatch_v5.json
base/qwen35-9b/prompts/<expert_id>.txt
```

В V6 их можно сначала оставить физически раздельными, но предоставить единый программный интерфейс `ExpertRegistry`.

Позже возможно объединение в один manifest.

### 6.2. Expert Registry должен уметь

- `get(expert_id)`;
- `list_available()`;
- `is_available(expert_id, backend)`;
- `resolve_adapter(expert_id)`;
- `get_prompt(expert_id)`;
- `check_backbone_compatibility(expert_id)`;
- отдавать Router краткие semantic descriptions экспертов.

---

## 7. LoRA Manager

LoRA Manager отвечает только за lifecycle адаптеров.

Он не знает ничего про workflow, шаги или агентов.

Поток:

```text
expert_id
  ↓
Expert Registry
  ↓
adapter reference
  ↓
compatibility check
  ↓
is loaded / cached?
  ├─ yes → activate
  └─ no  → load / prepare → activate
  ↓
inference
  ↓
detach / restore base state
```

### 7.1. Что переносится из v5

Практически без изменения сохраняются:

- одна общая базовая модель;
- одна активная экспертная LoRA на inference;
- compatibility check с backbone;
- MLX lazy load / conversion;
- локальный adapter cache;
- GGUF подготовленные LoRA;
- attach/detach;
- invalidation runtime после критического detach failure;
- backend-specific telemetry;
- `_RUN_LOCK`, пока runtime не поддерживает безопасные параллельные adapter switches.

### 7.2. Будущее развитие

Можно добавить:

- LRU adapter cache;
- adapter prefetch;
- adapter pinning;
- VRAM-aware eviction;
- per-backend adapter pools;
- metrics hot-swap latency.

Но эти функции не должны добавлять agent orchestration.

---

## 8. Inference Engine

Inference Engine должен быть максимально простым.

Вход:

```text
base model
+ optional adapter
+ messages
+ expert system prompt
+ generation parameters
```

Выход:

```text
raw model generation
+ token usage
+ stop reason
+ timing
+ backend telemetry
```

Inference Engine не выбирает эксперта и не решает, что делать после генерации.

Backend должен оставаться абстрагированным.

Поддерживаемые/возможные backend:

```text
MLX
llama.cpp / GGUF
vLLM
SGLang
Transformers
```

V6 не должен жёстко зависеть от конкретного inference backend.

---

## 9. Expert execution

Эксперт получает только данные, необходимые для текущего inference-step.

Минимально:

```text
messages
+ expert system prompt
+ selected LoRA
```

Возможно добавить общие runtime-инструкции по формату ответа.

Эксперт **не получает** внутренние понятия старого V5:

```text
step number
depends_on
prerequisites
downstream jobs
missing capabilities
routing DAG
other expert assignments
```

Эксперт не должен знать, первый он, последний или промежуточный.

Для него каждый вызов — самостоятельная конкретная задача.

---

## 10. Verifier

В V6 сохраняется отдельная проверка результата, но её роль меняется.

Verifier — это **не Final Thinker** и **не второй эксперт**.

Он не переписывает ответ и не выполняет задачу заново.

Его единственная функция — решить, можно ли принять результат текущего эксперта.

Все эксперты в MLX и GGUF получают общую языковую политику из `core/response_language.py`, даже если у адаптера нет собственного system prompt. Явное указание языка и ранее заданное предпочтение пользователя имеют приоритет; иначе пояснения следуют языку текущего запроса пользователя. Язык обучения LoRA, цитат, документов, tool results и прошлых ответов ассистента не задаёт язык ответа. Для перевода сохраняется целевой язык; код, имена инструментов, аргументы, ключи JSON и фиксированные метки сохраняют требуемый формат. Router отделяет язык ответа от предметной специализации, направления перевода и юрисдикции.

Verifier проверяет язык обычного ответа и финального текста агента и запрашивает `RETRY_WITH_FEEDBACK` с указанием требуемого языка при нарушении. Повтор выполняется с той же LoRA в существующем бюджете. Это политика inference, а не переобучение: сильное языковое смещение адаптера может привести к исчерпанию повторов.

Контракт:

```json
{
  "status": "ACCEPT",
  "feedback": null
}
```

или:

```json
{
  "status": "RETRY_WITH_FEEDBACK",
  "feedback": "Ответ обрывается до завершения функции validate_request()."
}
```

или:

```json
{
  "status": "FAIL",
  "feedback": "Эксперт не смог сформировать корректный результат в доступном бюджете."
}
```

### 10.1. ACCEPT

Экспертный output возвращается наружу **без повторной генерации**.

```text
Expert output
  ↓
Verifier ACCEPT
  ↓
return exact expert output
```

### 10.2. RETRY_WITH_FEEDBACK

Допускается ограниченный retry **того же эксперта**:

```text
original messages
+ previous expert output
+ verifier feedback
  ↓
same Expert / same LoRA
  ↓
new expert output
```

Рекомендуемый default: максимум **1 retry** сверх первой попытки.

Это не считается multi-agent orchestration, потому что:

- expert_id не меняется;
- новая специализация не выбирается;
- workflow не строится;
- Verifier только проверяет завершённость/приемлемость текущего inference.

### 10.3. FAIL

После исчерпания retry или критической ошибки GhostSwitch возвращает failure status внешнему harness.

Дальнейшее решение принимает harness.

Например он может:

- повторить вызов позже;
- изменить prompt;
- выбрать DIRECT expert;
- использовать другую модель;
- прекратить workflow.

### 10.4. Что удаляется из v5

Удаляется старый completion protocol, где модель должна продолжить/пересобрать ответ с `COMPLETE:` / `INCOMPLETE:` и может видеть только хвост незавершённого артефакта.

Verifier не должен создавать новый пользовательский артефакт.

---

## 11. Отсутствие Final Thinker

В V6 **нет внешнего Final Thinker** после экспертного результата.

V5:

```text
Expert
  ↓
Final Thinker
  ↓
новая генерация final answer
```

V6:

```text
Expert
  ↓
Verifier ACCEPT
  ↓
тот же output наружу
```

Причины:

- меньше latency;
- меньше токенов;
- нет двойной работы;
- не теряется код/HTML/JSON эксперта;
- нет риска, что базовая модель сократит большой artifact;
- harness получает именно результат выбранной специализации.

Если внешний harness хочет переформатировать или синтезировать ответ, он делает это своим следующим workflow-step.

---

## 12. Multi-expert behavior

GhostSwitch Core **не управляет цепочками экспертов**.

Если задаче нужны несколько специализаций, внешний harness делает несколько отдельных inference calls.

Пример:

```text
Harness
  ↓
"Реализуй backend endpoint"
  ↓
GhostSwitch AUTO
  ↓
Router → coding.backend
  ↓
response A

Harness
  ↓
"Проверь этот код на security issues: <response A>"
  ↓
GhostSwitch AUTO
  ↓
Router → security.code_review
  ↓
response B
```

GhostSwitch не хранит понятия:

```text
previous expert
next expert
producer
reviewer
expert chain
expert dependency
artifact lifecycle across agents
```

Передача результатов между экспертами — ответственность harness через обычные `messages/context` следующего вызова.

---

## 13. Base route

Не каждый запрос требует LoRA.

Router может выбрать:

```json
{
  "route": "base",
  "expert_id": null
}
```

Тогда запрос выполняется общей базовой моделью без адаптера:

```text
Request
  ↓
Router
  ↓
BASE
  ↓
Base Model
  ↓
Inference
  ↓
Verifier
  ↓
Response
```

Это полезно для:

- general questions;
- простых инструкций;
- задач вне узкой специализации;
- случаев, когда LoRA не даст ожидаемого улучшения.

---

## 14. Ошибки и unsupported routes

V6 должен явно различать model failure и отсутствие capability.

Пример unsupported:

```json
{
  "status": "unsupported",
  "answer": null,
  "routing": {
    "route": "unsupported",
    "missing_capability": "legal.tax.georgia"
  }
}
```

Пример runtime failure:

```json
{
  "status": "failed",
  "answer": null,
  "error": {
    "type": "adapter_load_error",
    "expert_id": "coding.python"
  }
}
```

GhostSwitch не должен пытаться строить новый multi-step plan для компенсации отсутствующего эксперта.

Это решение отдаётся внешнему harness.

---

## 15. Response schema

V6 должен уметь возвращать OpenAI-compatible response и GhostSwitch metadata.

Пример:

```json
{
  "id": "mb-req-123",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 840,
    "total_tokens": 2040
  },
  "ghostswitch": {
    "mode": "auto",
    "route": "expert",
    "expert_id": "coding.python",
    "adapter": "python-v4",
    "router_confidence": 0.96,
    "verification": {
      "status": "ACCEPT",
      "attempts": 1
    },
    "backend": "mlx",
    "latency_ms": 842
  }
}
```

Для DIRECT mode:

```json
{
  "ghostswitch": {
    "mode": "direct",
    "expert_id": "coding.python",
    "router_used": false
  }
}
```

---

## 16. Telemetry и observability

Из v5 стоит сохранить подробную observability.

Полезные поля:

```text
expert_id
adapter_id
backend
router latency
router raw output (debug only)
router confidence
adapter load latency
adapter activation latency
inference latency
tokens
stop reason
verifier result
retry count
cache hit/miss
runtime errors
```

Полные prompts/raw outputs лучше хранить только в debug/tracing режиме из-за:

- размера;
- приватности;
- чувствительного пользовательского контекста.

Для production response наружу должна выходить компактная telemetry.

---

## 17. Бюджеты генерации

Бюджеты v6 становятся проще, потому что больше нет multi-expert graph и отдельного final synthesis.

Рекомендуемая модель бюджета:

```text
router budget
+ expert generation budget
+ verifier budget
+ optional one retry budget
```

Пример профиля:

```text
Router:      32–128 tokens
Expert:      до configured max_tokens
Verifier:    32–128 tokens
Retry:       максимум 1 раз, тот же expert
```

Не нужен отдельный reserve под Final Thinker.

Конкретные значения должны задаваться execution profile и benchmark'ами.

---

## 18. Execution profiles

V6 сохраняет идею backend-specific execution profiles.

Например:

```text
mlx9b_v6.json
gguf9b_v6.json
```

Профиль определяет:

- base model;
- backend;
- context window;
- generation defaults;
- router budget;
- expert budget;
- verifier budget;
- retry policy;
- adapter cache;
- backend-specific настройки.

Но profile больше не содержит настройки DAG, multi-step decomposition и Final Thinker.

---

## 19. Предлагаемая структура кода

```text
core/
├── api.py
├── request_contract.py
├── main.py
│
├── routing/
│   ├── router.py
│   ├── router_contract.py
│   └── policy.json
│
├── experts/
│   ├── registry.py
│   ├── manifest.json
│   └── prompts/
│
├── runtime/
│   ├── lora_manager.py
│   ├── inference.py
│   ├── mlx_backend.py
│   └── gguf_backend.py
│
├── verification/
│   ├── verifier.py
│   └── verifier_contract.py
│
├── telemetry/
│   └── metrics.py
│
└── config/
    └── execution_profiles/
        ├── mlx9b_v6.json
        └── gguf9b_v6.json
```

Это целевая логическая структура; миграция может быть постепенной без немедленного переименования всех существующих файлов.

---

## 20. Что переносится из v5

### Оставить почти без изменений

```text
LoRA compatibility checks
MLX adapter loading/conversion
GGUF adapter preparation/activation
adapter cache
base-model reuse
backend abstraction
backend telemetry
_RUN_LOCK / serialization policy
expert prompts
runtime metrics
execution profiles concept
```

### Оставить, но упростить

```text
request_contract.py
capability_router.py
lora_expert_provider.py
main.py
result schema
budget management
```

### Удалить из Core

```text
jobs[]
depends_on
multi-step plan
DAG validation
coverage of full workflow
repair of dependencies
internal_decomposition
max_experts_per_task
blocked downstream steps
expert-to-expert orchestration
Final Thinker
submit_final
late handoff logic
final synthesis budget
COMPLETE / INCOMPLETE artifact recreation protocol
```

---

## 21. Миграция файлов v5 → v6

### `core/lora_runtime.py`

**Сохранить почти полностью.**

Целевая роль: `LoRAManager`.

### `core/llm.py`

**Сохранить.**

Целевая роль: backend-independent inference + backend implementations + telemetry.

### `base/qwen35-9b/adapters.json`

**Сохранить.**

Использовать как источник adapter metadata либо мигрировать в новый Expert Registry manifest.

### `base/qwen35-9b/prompts/*`

**Сохранить.**

Prompt является частью Expert definition.

### `core/config/routing/v2_capability_dispatch_v5.json`

**Сохранить semantic часть, удалить orchestration policy.**

Нужны descriptions/capabilities для выбора одного эксперта.

### `core/capability_router.py`

**Сильно упростить.**

Было:

```text
query → jobs[] → dependencies → coverage → repair
```

Станет:

```text
messages → expert_id/base/unsupported
```

### `core/lora_expert_provider.py`

**Сохранить inference и expert prompt assembly.**

Удалить:

- step dependencies;
- downstream metadata;
- старый completion protocol.

Добавить взаимодействие с Verifier.

### `core/request_contract.py`

**Упростить.**

Удалить:

```text
internal_decomposition
max_experts_per_task
```

Добавить:

```text
mode
direct expert_id
OpenAI-compatible messages
```

### `core/main.py`

**Переписать controller flow.**

Новый flow:

```text
validate
→ resolve mode
→ route or direct expert
→ resolve registry
→ inference
→ verifier
→ optional same-expert retry
→ response
```

### `core/thinker.py`

Текущая роль Final Thinker в V6 не нужна.

Файл можно удалить либо переиспользовать только для другой узкой runtime-функции, если она действительно нужна.

### `core/final_policy.py`

Текущий final synthesis policy удалить.

Если нужны verifier policies, создать отдельный `verification/` layer.

---

## 22. Что GhostSwitch V6 принципиально НЕ делает

Это важная граница архитектуры.

GhostSwitch V6 не должен постепенно снова превращаться в agent harness.

Запрещённая для Core логика:

```text
"после coding expert вызови testing expert"
"если tester нашёл ошибку, вернись к coder"
"построй план из пяти специалистов"
"запусти независимые ветки параллельно"
"разреши зависимости между задачами"
"вызови browser/tool"
"прочитай файл и продолжи agent loop"
"сделай итоговый synthesis всех экспертов"
```

Всё это выполняет внешний harness.

GhostSwitch должен видеть каждый текущий вызов как самостоятельный inference request.

---

## 23. Интеграция с OpenClaw / LangGraph / smolagents

Для внешней системы GhostSwitch выглядит как provider/model endpoint.

```text
External Harness
       ↓
OpenAI-compatible GhostSwitch API
       ↓
GhostSwitch Router
       ↓
Expert Registry
       ↓
LoRA Manager
       ↓
Base + LoRA
       ↓
Inference
       ↓
Verifier
       ↓
Response
```

Harness может сделать 1, 5 или 20 вызовов GhostSwitch в рамках своего workflow. GhostSwitch не должен знать об их взаимосвязи, кроме контекста, который harness сам передал в `messages`.

---

## 24. Главная продуктовая идея V6

GhostSwitch V6 — не новый OpenClaw и не новый LangGraph.

Это **specialized inference layer**, который позволяет любому harness использовать одну общую базовую модель как набор динамически выбираемых специализированных моделей через LoRA.

Для внешней системы:

```text
one model endpoint
```

Внутри GhostSwitch:

```text
Router
  ↓
Expert specialization
  ↓
Base + LoRA hot-swap
  ↓
Verified inference
```

Это позволяет независимо развивать:

- качество Router;
- количество экспертов;
- качество LoRA;
- скорость hot-swap;
- inference backend;
- verifier;

не создавая собственную полноценную agent platform.

---

## 25. Каноническая схема GhostSwitch v6

```text
                         External Harness
                  OpenClaw / LangGraph / custom
                                │
                                │ current inference step
                                ▼
┌────────────────────────────────────────────────────────────┐
│                     GhostSwitch v6 Core                      │
│                                                            │
│  Request/API                                               │
│      ↓                                                     │
│  ┌────────────── Mode ──────────────┐                      │
│  │                                  │                      │
│ AUTO                              DIRECT                   │
│  │                                  │                      │
│ Router                           expert_id                  │
│  │                                  │                      │
│  └───────────────┬──────────────────┘                      │
│                  ↓                                         │
│            Expert Registry                                 │
│                  ↓                                         │
│             LoRA Manager                                   │
│                  ↓                                         │
│          Base Model + Adapter                              │
│                  ↓                                         │
│              Inference                                     │
│                  ↓                                         │
│              Verifier                                      │
│          ┌───────┼────────┐                                │
│          │       │        │                                │
│       ACCEPT   RETRY     FAIL                              │
│          │       │        │                                │
│          │   same expert   │                                │
│          │   + feedback    │                                │
│          └───────┴────────┘                                │
│                  ↓                                         │
│               Response                                     │
└────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         External Harness
                  decides what happens next
```

---

## 26. Краткое отличие V5 и V6

```text
V5
request
→ router/planner
→ jobs + dependencies
→ multiple expert execution
→ dependency controller
→ completion/finalizer logic
→ Final Thinker
→ answer
```

```text
V6
request/current harness step
→ router OR direct expert
→ one selected specialization
→ base + LoRA
→ inference
→ verifier
→ answer
```

Главное изменение не в отказе от Router или LoRA-экспертов. Они остаются ядром GhostSwitch.

Удаляется именно **workflow orchestration между экспертами**.

---

## 27. Итог

GhostSwitch v6 должен быть быстрым, предсказуемым и подключаемым model intelligence layer.

Его задача:

```text
получить текущий запрос
→ выбрать лучшую специализацию
→ подключить нужную LoRA
→ выполнить inference
→ проверить результат
→ вернуть его без повторной генерации
```

Его задача **не**:

```text
планировать всю работу пользователя
→ создавать агентов
→ управлять их зависимостями
→ выполнять tools
→ собирать финальный ответ из цепочки агентов
```

Именно эта граница делает V6 совместимым с существующими и будущими harness, не уничтожая главное преимущество GhostSwitch — динамический выбор и hot-swap специализированных LoRA поверх общей базовой модели.

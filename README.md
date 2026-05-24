# ВКР МАИ на тему "Интеллектуальный агент для подготовки гражданско-правовых договоров"

Цирулев Н.В., студент группы М8О-408Б-22

## Требования
- Docker и Docker Compose
- NVIDIA GPU + драйвер + NVIDIA Container Toolkit
- Свободные порты: `3000`, `8000`, `2024`, `5432`, `6333`, `9000`, `9001`, `11435`

## Запуск

```bash
sudo docker compose up -d --build
sudo docker compose exec ollama ollama singin # там появится ссылка, ее открыть в браузере и авторизоваться в ollama
sudo docker compose exec ollama ollama pull gemma4:31b-cloud # или другую модель с ollama
```

## Интерфейсы

| Назначение | URL |
|---|---|
| Веб-интерфейс пользователя (чат с агентом) | `http://localhost:3000/` |
| Админ-панель (креды задаются в `.env` (`OWNER_USERNAME` / `OWNER_PASSWORD`)) | `http://localhost:3000/admin/login` |
| Swagger UI FastAPI (REST API бекенда) | `http://localhost:8000/docs` |
| ReDoc FastAPI | `http://localhost:8000/redoc` |
| LangGraph Studio (визуализация и пошаговое исполнение графа агента) | `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` |
| LangGraph dev API (Swagger) | `http://localhost:2024/docs` |
| Qdrant Web UI | `http://localhost:6333/dashboard` |
| Qdrant REST API | `http://localhost:6333/` |
| MinIO S3 API, доступ по кредам из `.env` : `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` | `http://localhost:9000/` |
| Ollama REST API | `http://localhost:11435/` |
| PostgreSQL (доступ: `POSTGRES_USER` / `POSTGRES_PASSWORD` в `.env`) | `localhost:5432`, БД `contracts_db` |

## Запуск тестов поиска на эталонном датасете

Скрипт `scripts/eval_retrieval.py` прогоняет каждое описание из `data/gold_sources.json` через граф агента до узла `retrieve_norms` (с прерыванием) и сравнивает возвращённые чанки с эталонными ссылками. Уточняющие вопросы агента симулируются отдельным LLM-вызовом, поэтому ручного участия не требуется.

Запускается **внутри контейнера** `app` или `langgraph_dev` (нужны доступ к Qdrant, Ollama и загруженным моделям):

```bash
sudo docker compose exec app python scripts/eval_retrieval.py
```

Аргументы:

```bash
sudo docker compose exec app python scripts/eval_retrieval.py \
    --input data/gold_sources.json \
    --output-dir data/eval_outputs \
    --max-clarifications 3 \
    --k-values 3,5,10,20
```

Результат — CSV в `data/eval_outputs/`. Имя файла кодирует конфигурацию (эмбеддер, реранкер, фильтр) и unix-timestamp, чтобы разные прогоны не перезаписывали друг друга. По каждой строке выводятся `precision@k`, `recall@k`, `hit@k`, латентность узла поиска и сравнение «эталон vs выдача» (`required_sources` / `actual_sources`).

Чтобы сравнить две конфигурации поиска (например, с реранкером и без него), меняются переменные в `.env` (`RERANKER_ENABLED`, `RETRIEVAL_FILTER_ENABLED`, `EMBED_MODEL`), пересобирается бекенд и запускается скрипт повторно — CSV сохранится под другим именем.

Сводный анализ полученных CSV — в `notebooks/eval_outputs_analysis_notebook.ipynb`.

## Структура репозитория

| Каталог | Что внутри |
|---|---|
| `app/` | FastAPI-бекенд, граф LangGraph, RAG, биллинг, админка, аутентификация |
| `frontend/` | Статика SPA на vanilla JS, nginx-конфиг, два экрана: пользовательский и админский |
| `scripts/` | CLI-инструменты для оценки качества и зондирования RAG |
| `data/` | `gold_sources.json` — эталонный датасет; `eval_outputs/` — результаты прогонов |
| `notebooks/` | Jupyter-разбор результатов оценки |
| `diploma-text/` | LaTeX-источники текста ВКР |
| `results/` | Артефакты экспертной апробации (CSV оценок, графики, скриншоты опроса) |

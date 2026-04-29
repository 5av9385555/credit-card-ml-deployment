# Credit Card Default Prediction (ML Deployment)

## 📌 Описание проекта

Данный проект реализует production-like сервис машинного обучения для прогнозирования дефолта по кредитным картам.

Модель обучена на датасете **Default of Credit Card Clients (UCI)** и развёрнута в виде REST API с использованием Flask и Docker.

---

## 🎯 Цель

Построить полный pipeline:

* обучение модели
* сохранение модели
* создание API
* контейнеризация
* тестирование

---

## 🧠 Модель

Используется модель:

* **GradientBoostingClassifier**

Метрика качества:

* ROC-AUC ≈ **0.78**

---

## 📓 Обучение модели

Ноутбук с обучением модели:

[Открыть ноутбук](notebooks/training_pipeline.ipynb)

---

## 🧪 A/B тестирование

Подробное описание эксперимента:

👉 [ab_test_plan.md](./ab_test_plan.md)

---

## 📂 Структура проекта

```
credit-card-ml-deployment/
│
├── app/                         # Основной код приложения
│   ├── api.py                   # Flask API
│   └── model_handler.py         # Загрузка модели
│
├── models/                      # Сохранённые модели
│   ├── model_v1.pkl             # Обученная ML модель
│   ├── features_v1.pkl          # Список признаков/фичей
│
├── notebooks/                   # Исследование и обучение модели
│   └── training_pipeline.ipynb  # Jupyter notebook с EDA и обучением
│
├── tests/                       # Автоматические тесты
│   └── test_api.py              # Тесты API (/health и /predict)
│
├── Dockerfile                   # Инструкция сборки Docker-образа
├── requirements.txt             # Зависимости Python
├── docker-compose.yml           # (опционально) запуск через Docker Compose
├── ab_test_plan.md              # План A/B тестирования моделей
└── README.md                    # Документация проекта
```

---

## 🚀 Запуск проекта

### 🔹 1. Локально (без Docker)

```bash
pip install -r requirements.txt
python -m app.api
```

Сервис будет доступен:

```
http://127.0.0.1:5000
```

---

### 🔹 2. Через Docker Hub

#### Скачать образ:

```bash
docker pull 5avdocker/credit-card-api
```

#### Запуск контейнера:

```bash
docker run -p 5000:5000 5avdocker/credit-card-api
```

---

## 🔍 API эндпоинты

### ✅ Проверка сервиса

```
GET /health
```

Ответ:

```json
{"status": "healthy"}
```

---

### 🤖 Предсказание

```
POST /predict
```

Пример запроса (PowerShell):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
-Method Post `
-Body (@{
LIMIT_BAL=20000
SEX=2
EDUCATION=2
MARRIAGE=1
AGE=24
PAY_0=2
PAY_2=2
PAY_3=-1
PAY_4=-1
PAY_5=-2
PAY_6=-2
BILL_AMT1=3913
BILL_AMT2=3102
BILL_AMT3=689
BILL_AMT4=0
BILL_AMT5=0
BILL_AMT6=0
PAY_AMT1=0
PAY_AMT2=689
PAY_AMT3=0
PAY_AMT4=0
PAY_AMT5=0
PAY_AMT6=0
avg_delay=-0.3333
max_delay=2
avg_bill=1284
avg_pay=114.83
utilization=0.19565
} | ConvertTo-Json) `
-ContentType "application/json"
```

---

## 📊 Ответ модели

```json
{
  "prediction": 1,
  "probability": 0.746,
  "model_version": "v1"
}
```

---

## 🧪 Тестирование

```bash
pytest
```

---

## ⚙️ Технологии

* Python 3.11+
* scikit-learn
* Flask
* Docker
* joblib

---

## 📈 Возможные улучшения

* A/B тестирование моделей
* логирование (ELK stack)
* масштабирование через Kubernetes
* использование Gunicorn + Nginx

---

## ⚠️ Замечания

* Возможны предупреждения sklearn о несовместимости версий (не критично)
* Flask используется как dev-сервер (в production нужен WSGI)

---

## 👨‍💻 Автор

Автор: Andrey Vasiluyk

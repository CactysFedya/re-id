# Установка и запуск проекта

Нужен Python `3.14` или новее.

Варианты установки:

- скачать архив проекта;
- склонировать репозиторий через `git clone`;
- скачать файл `.whl` из релиза.

## Установка из архива или из GitHub

Откройте терминал в папке проекта и выполните:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.gpu.txt
```

Если отсутствует видеокарта, вместо `requirements.gpu.txt`:

```powershell
python -m pip install -r requirements.cpu.txt
```

Установка зависимостей:

```powershell
python -m pip install .
```

Для разработки:

```powershell
python -m pip install -e .[dev]
```

## Установка из `.whl`

Необходимые файлы:

- файл `.whl`;
- `requirements.cpu.txt` или `requirements.gpu.txt` для CPU/GPU.

В терминале:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.cpu.txt
python -m pip install .\reid_pipeline-0.1.0-py3-none-any.whl
```

Для GPU вместо `requirements.cpu.txt` используйте `requirements.gpu.txt`.

## Запуск

Запуск на видеофайле:

```powershell
reid-run --video assets/test.mp4
```

Запуск с конфигом:

```powershell
reid-run --config configs/pipeline.toml
```

## Конфиг

Показать шаблон конфига:

```powershell
reid-dump-config
```

Сохранить шаблон в файл:

```powershell
reid-dump-config configs/pipeline.user.toml
```

Описание параметров: [config.md](config.md)  
Шаблон конфига: [configs/pipeline.template.toml](configs/pipeline.template.toml)

## Сборка

```powershell
python -m pip install build
python -m build
```

Проверка тестов:

```powershell
python -m unittest discover -s tests
```

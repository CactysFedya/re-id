# Программный модуль для повторной идентификации объектов в видеопотоке с камер видеонаблюдения

Проект подготовлен к упаковке в устанавливаемый Python-модуль с CLI и частично переопределяемым `TOML`-конфигом.

Требуемая версия Python: `3.14` или новее.

## Установка

Сначала установите PyTorch под вашу платформу:

### CPU

```powershell
python -m pip install -r requirements.cpu.txt
```

### GPU

```powershell
python -m pip install -r requirements.gpu.txt
```

Затем установите сам пакет и runtime-зависимости:

```powershell
python -m pip install .
```

Для разработки:

```powershell
python -m pip install -e .[dev]
```

## Запуск

Запуск со встроенными defaults и явным видеофайлом:

```powershell
reid-run --video assets/test.mp4
```

Запуск с пользовательским конфигом:

```powershell
reid-run --config configs/pipeline.toml
```

Запуск с RTSP-источником без редактирования конфига:

```powershell
reid-run --source-type http --source-uri http://192.168.0.1:8080/video
```

## Конфиг

- Полный шаблон конфига: `reid-dump-config`
- Запись шаблона в файл: `reid-dump-config configs/pipeline.user.toml`
- Документация по параметрам: [docs/config.md](docs/config.md)
- Конфиг по умолчанию: [configs/pipeline.template.toml](configs/pipeline.template.toml)

Пользовательский `TOML` может быть частичным: достаточно указать только те секции, которые нужно переопределить.

## Сборка

Собрать `sdist` и `wheel`:

```powershell
python -m pip install build
python -m build
```

Локальная проверка:

```powershell
python -m unittest discover -s tests
```

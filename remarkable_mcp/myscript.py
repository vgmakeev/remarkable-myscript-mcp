"""
MyScript OCR для reMarkable - работа с векторными данными напрямую.

Этот модуль интегрирует MyScript Cloud API для распознавания рукописного текста
из reMarkable, работая напрямую с векторными штрихами вместо конвертации в PNG.

Преимущества MyScript:
- Лучшее качество распознавания русского рукописного текста
- Работа с векторными данными (не нужно конвертировать в изображения)
- Поддержка множества языков

Настройка:
- MYSCRIPT_APP_KEY: Application Key от MyScript
- MYSCRIPT_HMAC_KEY: HMAC Key от MyScript
- REMARKABLE_OCR_BACKEND=myscript: Активация MyScript как бэкенда OCR

Получить ключи: https://developer.myscript.com/
"""

import hashlib
import hmac
import json
import os
import struct
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class LanguageCode(str, Enum):
    """Коды языков для MyScript API."""

    RU = "ru_RU"
    EN = "en_US"
    DE = "de_DE"
    FR = "fr_FR"
    ES = "es_ES"
    IT = "it_IT"
    PT = "pt_PT"
    ZH = "zh_CN"
    JA = "ja_JP"
    KO = "ko_KR"


class PointerType(str, Enum):
    """Типы указателей."""

    PEN = "PEN"
    TOUCH = "TOUCH"
    ERASER = "ERASER"


@dataclass
class Stroke:
    """Один штрих рукописного текста."""

    x: List[int]
    y: List[int]
    t: List[int]  # timestamps в миллисекундах
    p: List[float]  # pressure (0.0-1.0)
    pointer_type: PointerType = PointerType.PEN
    pointer_id: int = -1
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в формат MyScript API."""
        result = {
            "x": self.x,
            "y": self.y,
            "t": self.t,
            "p": self.p,
        }
        if self.pointer_type != PointerType.PEN:
            result["pointerType"] = self.pointer_type.value
        if self.pointer_id != -1:
            result["pointerId"] = self.pointer_id
        if self.id:
            result["id"] = self.id
        return result


@dataclass
class StrokeGroup:
    """Группа штрихов."""

    strokes: List[Stroke]
    pen_style: str = "color: #000000; -myscript-pen-width: ;"

    def to_dict(self) -> Dict[str, Any]:
        return {"penStyle": self.pen_style, "strokes": [s.to_dict() for s in self.strokes]}


@dataclass
class MyScriptRequest:
    """Запрос к MyScript API."""

    width: int = 1872  # reMarkable 2 width
    height: int = 1404  # reMarkable 2 height
    content_type: str = "Text"
    conversion_state: str = "DIGITAL_EDIT"
    x_dpi: int = 96
    y_dpi: int = 96
    stroke_groups: List[StrokeGroup] = None
    language: LanguageCode = LanguageCode.RU

    def __post_init__(self):
        if self.stroke_groups is None:
            self.stroke_groups = []

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в формат MyScript API."""
        return {
            "width": self.width,
            "height": self.height,
            "contentType": self.content_type,
            "conversionState": self.conversion_state,
            "xDPI": self.x_dpi,
            "yDPI": self.y_dpi,
            "strokeGroups": [sg.to_dict() for sg in self.stroke_groups],
            "configuration": {
                "lang": self.language.value,
                "text": {
                    "guides": {"enable": False},
                    "margin": {"top": 0, "left": 0, "right": 0, "bottom": 0},
                    "configuration": {"addLKText": True},
                },
                "export": {
                    "jiix": {
                        "strokes": True,
                        "bounding-box": True,
                        "style": True,
                        "text": {"chars": False, "words": True},
                    },
                    "image-resolution": 300,
                },
                "raw-content": {
                    "recognition": {"text": True, "shape": True},
                    "text": {"addLKText": True},
                },
            },
        }


class MyScriptOCR:
    """Клиент для работы с MyScript Cloud API."""

    BATCH_ENDPOINT = "/api/v4.0/iink/batch"
    BASE_URL = "https://cloud.myscript.com"
    FREE_TIER_LIMIT = 2000  # Бесплатный лимит: 2000 запросов в месяц

    def __init__(
        self,
        app_key: Optional[str] = None,
        hmac_key: Optional[str] = None,
    ):
        """
        Инициализация клиента MyScript.

        Args:
            app_key: Application Key от MyScript (или MYSCRIPT_APP_KEY env var)
            hmac_key: HMAC Key от MyScript (или MYSCRIPT_HMAC_KEY env var)
        """
        self.app_key = app_key or os.environ.get("MYSCRIPT_APP_KEY")
        self.hmac_key = hmac_key or os.environ.get("MYSCRIPT_HMAC_KEY")

        if not self.app_key or not self.hmac_key:
            raise ValueError(
                "MyScript API keys not configured. "
                "Set MYSCRIPT_APP_KEY and MYSCRIPT_HMAC_KEY environment variables."
            )

        self.session = requests.Session()

    def _compute_hmac(self, data: bytes) -> str:
        """
        Вычисление HMAC подписи для запроса.

        Args:
            data: JSON payload в виде bytes

        Returns:
            HMAC подпись в hex формате
        """
        # User key = app_key + hmac_key
        key = (self.app_key + self.hmac_key).encode("utf-8")
        # SHA-512 HMAC
        mac = hmac.new(key, data, hashlib.sha512)
        return mac.hexdigest()

    def recognize(self, request: MyScriptRequest) -> Dict[str, Any]:
        """
        Распознавание рукописного текста через MyScript API.

        Args:
            request: Запрос с векторными данными

        Returns:
            Результат распознавания с полем "label" (распознанный текст)
        """
        # Конвертируем запрос в JSON
        payload_dict = request.to_dict()
        payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")

        # Вычисляем HMAC
        hmac_signature = self._compute_hmac(payload_bytes)

        # Формируем URL
        url = f"{self.BASE_URL}{self.BATCH_ENDPOINT}"

        # Формируем заголовки
        headers = {
            "applicationKey": self.app_key,
            "hmac": hmac_signature,
            "Content-Type": "application/json",
            "Accept": "application/json, application/vnd.myscript.jiix",
        }

        # Отправляем запрос (увеличенный таймаут для больших документов)
        response = self.session.post(url, data=payload_bytes, headers=headers, timeout=180)
        response.raise_for_status()

        return response.json()

    def recognize_text(self, request: MyScriptRequest) -> str:
        """
        Распознавание текста с возвратом только текста.

        Args:
            request: Запрос с векторными данными

        Returns:
            Распознанный текст
        """
        result = self.recognize(request)
        return result.get("label", "")


def convert_rm_strokes_to_myscript(
    rm_strokes: List[Dict],
    time_offset: int = 0,
) -> List[StrokeGroup]:
    """
    Конвертация штрихов из формата reMarkable в формат MyScript.

    Args:
        rm_strokes: Список штрихов из reMarkable (формат rmscene)
        time_offset: Начальное смещение времени в миллисекундах

    Returns:
        Список групп штрихов для MyScript
    """
    stroke_groups = []
    current_group = StrokeGroup(strokes=[])

    speed_factor = 20000  # 200 * 100
    stroke_gap = 50000  # 500 * 100
    min_speed = 0.01

    t = time_offset

    for rm_stroke in rm_strokes:
        # Пропускаем ластики и маркеры
        brush_type = rm_stroke.get("brushType", 0)
        if brush_type in [1, 2, 3, 4, 5, 6]:  # Eraser, EraseArea, Highlighter
            continue

        # Конвертируем точки
        dots = rm_stroke.get("dots", [])
        if not dots:
            continue

        x_points = []
        y_points = []
        t_points = []
        p_points = []

        x0, y0 = -1, -1

        for dot in dots:
            x1 = int(round(dot.get("x", 0)))
            y1 = int(round(dot.get("y", 0)))

            # Избегаем дубликатов точек
            if x0 == x1 and y0 == y1:
                continue

            speed = max(min_speed, dot.get("speed", min_speed))
            time_offset_ms = int(round(speed_factor / speed))
            t += time_offset_ms

            x_points.append(x1)
            y_points.append(y1)
            t_points.append(t)
            p_points.append(max(0.0, min(1.0, dot.get("pressure", 0.5))))

            x0, y0 = x1, y1

        if x_points:
            stroke = Stroke(
                x=x_points,
                y=y_points,
                t=t_points,
                p=p_points,
                pointer_type=PointerType.PEN,
            )
            current_group.strokes.append(stroke)
            t += stroke_gap

    if current_group.strokes:
        stroke_groups.append(current_group)

    return stroke_groups


def parse_rm_file_v6(rm_data: bytes) -> Dict[str, Any]:
    """
    Парсит .rm файл версии 6 используя rmscene.

    Args:
        rm_data: Содержимое .rm файла

    Returns:
        Словарь с данными штрихов
    """
    import rmscene

    all_strokes = []
    width = 1404
    height = 1872

    tree = rmscene.read_tree(BytesIO(rm_data))

    for node in tree.walk():
        # Проверяем, что это узел типа Line с точками
        if hasattr(node, "points") and node.points:
            points = node.points
            dots = []
            for p in points:
                # Проверяем на NaN и извлекаем координаты
                x_val = float(p.x) if hasattr(p, "x") and not (p.x != p.x) else 0.0
                y_val = float(p.y) if hasattr(p, "y") and not (p.y != p.y) else 0.0

                # Pressure в rmscene обычно от 0 до 255, нормализуем до 0.0-1.0
                pressure_val = 0.5
                if hasattr(p, "pressure"):
                    pressure_raw = float(p.pressure)
                    if pressure_raw > 0:
                        pressure_val = min(1.0, max(0.0, pressure_raw / 255.0))

                # Speed может быть 0 или больше
                speed_val = float(p.speed) if hasattr(p, "speed") else 1.0

                dots.append(
                    {
                        "x": int(round(x_val)),
                        "y": int(round(y_val)),
                        "speed": max(0.01, speed_val),
                        "pressure": pressure_val,
                        "timestamp": 0,
                    }
                )

            if dots:
                # Определяем тип кисти по tool
                tool = getattr(node, "tool", None)
                brush_type = 0  # По умолчанию обычная ручка

                if tool is not None:
                    tool_val = int(tool) if isinstance(tool, (int, float)) else 0
                    if tool_val >= 5:
                        brush_type = 6  # Eraser
                    elif tool_val >= 3:
                        brush_type = 5  # Highlighter
                    else:
                        brush_type = 0  # Pen

                all_strokes.append({"dots": dots, "brushType": brush_type})

    return {
        "strokes": all_strokes,
        "layers": [{"strokes": all_strokes}] if all_strokes else [],
        "width": width,
        "height": height,
    }


def parse_rm_file(rm_data: bytes) -> Dict[str, Any]:
    """
    Парсит .rm файл и извлекает векторные данные.

    Поддерживает форматы v3, v5 и v6.

    Args:
        rm_data: Содержимое .rm файла

    Returns:
        Словарь с векторными данными для конвертации в MyScript
    """
    # Заголовки для разных версий
    HEADER_V3 = b"reMarkable .lines file, version=3          "
    HEADER_V5 = b"reMarkable .lines file, version=5          "
    HEADER_V6 = b"reMarkable .lines file, version=6          "
    HEADER_LEN = 43

    all_strokes = []
    width = 1404  # reMarkable default
    height = 1872  # reMarkable default

    pos = 0

    # Читаем заголовок
    header = rm_data[pos : pos + HEADER_LEN]
    pos += HEADER_LEN

    if header == HEADER_V3:
        version = 3
    elif header == HEADER_V5:
        version = 5
    elif header == HEADER_V6:
        # Для v6 используем rmscene
        return parse_rm_file_v6(rm_data)
    else:
        raise ValueError(f"Unsupported .rm file format: {header[:30]}")

    # Читаем количество слоёв
    n_layers = struct.unpack("<I", rm_data[pos : pos + 4])[0]
    pos += 4

    # Читаем слои
    for layer_idx in range(n_layers):
        # Количество штрихов в слое
        n_strokes = struct.unpack("<I", rm_data[pos : pos + 4])[0]
        pos += 4

        # Читаем штрихи
        for stroke_idx in range(n_strokes):
            # Читаем параметры штриха
            brush_type = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            brush_color = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            padding = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            brush_size = struct.unpack("<f", rm_data[pos : pos + 4])[0]
            pos += 4

            # Дополнительное поле для v5
            if version == 5:
                unknown = struct.unpack("<I", rm_data[pos : pos + 4])[0]
                pos += 4

            # Количество точек
            n_dots = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4

            # Читаем точки
            dots = []
            for dot_idx in range(n_dots):
                x = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4
                y = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4
                speed = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4
                tilt = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4
                width_val = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4
                pressure = struct.unpack("<f", rm_data[pos : pos + 4])[0]
                pos += 4

                dots.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "speed": float(speed),
                        "pressure": float(pressure),
                        "timestamp": 0,
                    }
                )

            if dots:  # Добавляем только если есть точки
                stroke_data = {"dots": dots, "brushType": int(brush_type)}
                all_strokes.append(stroke_data)

    return {
        "strokes": all_strokes,
        "layers": [{"strokes": all_strokes}] if all_strokes else [],
        "width": width,
        "height": height,
    }


def get_language_from_env() -> LanguageCode:
    """
    Получает язык OCR из переменной окружения MYSCRIPT_LANGUAGE.

    По умолчанию возвращает русский (ru_RU).
    """
    lang_env = os.environ.get("MYSCRIPT_LANGUAGE", "ru").lower()

    lang_map = {
        "ru": LanguageCode.RU,
        "en": LanguageCode.EN,
        "de": LanguageCode.DE,
        "fr": LanguageCode.FR,
        "es": LanguageCode.ES,
        "it": LanguageCode.IT,
        "pt": LanguageCode.PT,
        "zh": LanguageCode.ZH,
        "ja": LanguageCode.JA,
        "ko": LanguageCode.KO,
    }

    return lang_map.get(lang_env, LanguageCode.RU)


def ocr_rm_file_with_myscript(rm_data: bytes) -> Optional[str]:
    """
    Распознает текст из .rm файла через MyScript.

    Args:
        rm_data: Содержимое .rm файла

    Returns:
        Распознанный текст или None при ошибке
    """
    try:
        # Парсим .rm файл
        parsed_data = parse_rm_file(rm_data)

        strokes = parsed_data.get("strokes", [])
        if not strokes:
            return None

        # Конвертируем в формат MyScript
        stroke_groups = convert_rm_strokes_to_myscript(strokes, time_offset=0)

        if not stroke_groups or not any(sg.strokes for sg in stroke_groups):
            return None

        # Создаем запрос
        request = MyScriptRequest(
            width=parsed_data.get("width", 1872),
            height=parsed_data.get("height", 1404),
            language=get_language_from_env(),
            stroke_groups=stroke_groups,
        )

        # Отправляем на распознавание
        client = MyScriptOCR()
        result = client.recognize(request)
        return result.get("label", "")

    except Exception:
        # Ошибка MyScript - вернём None, чтобы можно было использовать fallback
        return None

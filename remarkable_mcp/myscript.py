"""
MyScript OCR for reMarkable - working with vector data directly.

This module integrates MyScript Cloud API for handwriting recognition
from reMarkable, working directly with vector strokes instead of PNG conversion.

Advantages of MyScript:
- Excellent handwriting recognition quality (especially for non-Latin scripts)
- Works with vector data (no image conversion needed)
- Multi-language support

Configuration:
- MYSCRIPT_APP_KEY: Application Key from MyScript
- MYSCRIPT_HMAC_KEY: HMAC Key from MyScript
- REMARKABLE_OCR_BACKEND=myscript: Enable MyScript as OCR backend
- MYSCRIPT_LANGUAGE: Language code (ru, en, de, fr, es, it, pt, zh, ja, ko)

Get API keys: https://developer.myscript.com/
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
    """Language codes for MyScript API."""

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
    """Pointer types for MyScript API."""

    PEN = "PEN"
    TOUCH = "TOUCH"
    ERASER = "ERASER"


@dataclass
class Stroke:
    """A single handwriting stroke."""

    x: List[int]
    y: List[int]
    t: List[int]  # timestamps in milliseconds
    p: List[float]  # pressure (0.0-1.0)
    pointer_type: PointerType = PointerType.PEN
    pointer_id: int = -1
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MyScript API format."""
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
    """A group of strokes with shared pen style."""

    strokes: List[Stroke]
    pen_style: str = "color: #000000; -myscript-pen-width: ;"

    def to_dict(self) -> Dict[str, Any]:
        return {"penStyle": self.pen_style, "strokes": [s.to_dict() for s in self.strokes]}


@dataclass
class MyScriptRequest:
    """Request payload for MyScript API."""

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
        """Convert to MyScript API format."""
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
    """Client for MyScript Cloud API."""

    BATCH_ENDPOINT = "/api/v4.0/iink/batch"
    BASE_URL = "https://cloud.myscript.com"
    FREE_TIER_LIMIT = 2000  # Free tier limit: 2000 requests per month

    def __init__(
        self,
        app_key: Optional[str] = None,
        hmac_key: Optional[str] = None,
    ):
        """
        Initialize MyScript client.

        Args:
            app_key: Application Key from MyScript (or MYSCRIPT_APP_KEY env var)
            hmac_key: HMAC Key from MyScript (or MYSCRIPT_HMAC_KEY env var)
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
        Compute HMAC signature for request.

        Args:
            data: JSON payload as bytes

        Returns:
            HMAC signature in hex format
        """
        # User key = app_key + hmac_key
        key = (self.app_key + self.hmac_key).encode("utf-8")
        # SHA-512 HMAC
        mac = hmac.new(key, data, hashlib.sha512)
        return mac.hexdigest()

    def recognize(self, request: MyScriptRequest) -> Dict[str, Any]:
        """
        Recognize handwriting via MyScript API.

        Args:
            request: Request with vector stroke data

        Returns:
            Recognition result with "label" field (recognized text)
        """
        # Convert request to JSON
        payload_dict = request.to_dict()
        payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")

        # Compute HMAC signature
        hmac_signature = self._compute_hmac(payload_bytes)

        # Build URL
        url = f"{self.BASE_URL}{self.BATCH_ENDPOINT}"

        # Build headers
        headers = {
            "applicationKey": self.app_key,
            "hmac": hmac_signature,
            "Content-Type": "application/json",
            "Accept": "application/json, application/vnd.myscript.jiix",
        }

        # Send request (increased timeout for large documents)
        response = self.session.post(url, data=payload_bytes, headers=headers, timeout=180)
        response.raise_for_status()

        return response.json()

    def recognize_text(self, request: MyScriptRequest) -> str:
        """
        Recognize handwriting and return text only.

        Args:
            request: Request with vector stroke data

        Returns:
            Recognized text
        """
        result = self.recognize(request)
        return result.get("label", "")


def convert_rm_strokes_to_myscript(
    rm_strokes: List[Dict],
    time_offset: int = 0,
) -> List[StrokeGroup]:
    """
    Convert strokes from reMarkable format to MyScript format.

    Args:
        rm_strokes: List of strokes from reMarkable (rmscene format)
        time_offset: Initial time offset in milliseconds

    Returns:
        List of stroke groups for MyScript API
    """
    stroke_groups = []
    current_group = StrokeGroup(strokes=[])

    speed_factor = 20000  # 200 * 100
    stroke_gap = 50000  # 500 * 100
    min_speed = 0.01

    t = time_offset

    for rm_stroke in rm_strokes:
        # Skip erasers and highlighters - they don't contain handwriting
        brush_type = rm_stroke.get("brushType", 0)
        if brush_type in [1, 2, 3, 4]:  # Eraser, EraseArea, Highlighter, HighlighterV5
            continue

        # Convert points
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

            # Skip duplicate points
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
    Parse .rm file version 6 using rmscene library.

    Args:
        rm_data: Contents of .rm file

    Returns:
        Dictionary with stroke data
    """
    import rmscene

    all_strokes = []
    width = 1404
    height = 1872

    tree = rmscene.read_tree(BytesIO(rm_data))

    for node in tree.walk():
        # Check if this is a Line node with points
        if hasattr(node, "points") and node.points:
            points = node.points
            dots = []
            for p in points:
                # Check for NaN and extract coordinates
                x_val = float(p.x) if hasattr(p, "x") and not (p.x != p.x) else 0.0
                y_val = float(p.y) if hasattr(p, "y") and not (p.y != p.y) else 0.0

                # Pressure in rmscene is typically 0-255, normalize to 0.0-1.0
                pressure_val = 0.5
                if hasattr(p, "pressure"):
                    pressure_raw = float(p.pressure)
                    if pressure_raw > 0:
                        pressure_val = min(1.0, max(0.0, pressure_raw / 255.0))

                # Speed can be 0 or greater
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
                # Determine brush type from rmscene tool enum
                # rmscene Pen values:
                #   0: PAINTBRUSH_1, 1: PENCIL_1, 2: BALLPOINT_1, 3: MARKER_1
                #   4: FINELINER_1, 5: HIGHLIGHTER_1, 6: ERASER, 7: MECHANICAL_PENCIL_1
                #   8: ERASER_AREA, 12: PAINTBRUSH_2, 13: MECHANICAL_PENCIL_2
                #   14: PENCIL_2, 15: BALLPOINT_2, 16: MARKER_2, 17: FINELINER_2
                #   18: HIGHLIGHTER_2, 21: CALIGRAPHY, 23: SHADER
                tool = getattr(node, "tool", None)
                brush_type = 0  # Default: regular pen (will be recognized)

                if tool is not None:
                    tool_val = int(tool) if isinstance(tool, (int, float)) else 0
                    # Mark erasers and highlighters for filtering
                    if tool_val in (6, 8):  # ERASER, ERASER_AREA
                        brush_type = 1  # Will be filtered out
                    elif tool_val in (5, 18):  # HIGHLIGHTER_1, HIGHLIGHTER_2
                        brush_type = 3  # Will be filtered out
                    else:
                        brush_type = 0  # Pen/Pencil/Marker - recognize

                all_strokes.append({"dots": dots, "brushType": brush_type})

    return {
        "strokes": all_strokes,
        "layers": [{"strokes": all_strokes}] if all_strokes else [],
        "width": width,
        "height": height,
    }


def parse_rm_file(rm_data: bytes) -> Dict[str, Any]:
    """
    Parse .rm file and extract vector data.

    Supports formats v3, v5, and v6.

    Args:
        rm_data: Contents of .rm file

    Returns:
        Dictionary with vector data for MyScript conversion
    """
    # Headers for different versions
    HEADER_V3 = b"reMarkable .lines file, version=3          "
    HEADER_V5 = b"reMarkable .lines file, version=5          "
    HEADER_V6 = b"reMarkable .lines file, version=6          "
    HEADER_LEN = 43

    all_strokes = []
    width = 1404  # reMarkable default
    height = 1872  # reMarkable default

    pos = 0

    # Read header
    header = rm_data[pos : pos + HEADER_LEN]
    pos += HEADER_LEN

    if header == HEADER_V3:
        version = 3
    elif header == HEADER_V5:
        version = 5
    elif header == HEADER_V6:
        # For v6 use rmscene library
        return parse_rm_file_v6(rm_data)
    else:
        raise ValueError(f"Unsupported .rm file format: {header[:30]}")

    # Read number of layers
    n_layers = struct.unpack("<I", rm_data[pos : pos + 4])[0]
    pos += 4

    # Read layers
    for layer_idx in range(n_layers):
        # Number of strokes in layer
        n_strokes = struct.unpack("<I", rm_data[pos : pos + 4])[0]
        pos += 4

        # Read strokes
        for stroke_idx in range(n_strokes):
            # Read stroke parameters
            brush_type = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            brush_color = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            padding = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4
            brush_size = struct.unpack("<f", rm_data[pos : pos + 4])[0]
            pos += 4

            # Additional field for v5
            if version == 5:
                unknown = struct.unpack("<I", rm_data[pos : pos + 4])[0]
                pos += 4

            # Number of points
            n_dots = struct.unpack("<I", rm_data[pos : pos + 4])[0]
            pos += 4

            # Read points
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

            if dots:  # Only add if there are points
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
    Get OCR language from MYSCRIPT_LANGUAGE environment variable.

    Defaults to Russian (ru_RU) if not set.
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


# MyScript API has a payload size limit (~1MB).
# Large pages with many strokes (e.g., 3000+) will get 413 error.
# Batch strokes to avoid this limit.
MYSCRIPT_BATCH_SIZE = 500
MYSCRIPT_LINE_THRESHOLD = 30  # Y-distance threshold to consider strokes on same line (pixels)


def _get_stroke_y_center(stroke: Dict) -> float:
    """Get average Y coordinate of a stroke for spatial sorting."""
    dots = stroke.get("dots", [])
    if not dots:
        return 0.0
    y_values = [d.get("y", 0) for d in dots]
    return sum(y_values) / len(y_values)


def _get_stroke_x_center(stroke: Dict) -> float:
    """Get average X coordinate of a stroke."""
    dots = stroke.get("dots", [])
    if not dots:
        return 0.0
    x_values = [d.get("x", 0) for d in dots]
    return sum(x_values) / len(x_values)


def _group_strokes_by_lines(strokes: List[Dict], threshold: float = MYSCRIPT_LINE_THRESHOLD) -> List[List[Dict]]:
    """
    Group strokes into lines based on Y-coordinate proximity.
    
    Strokes with Y-centers within 'threshold' pixels are considered on the same line.
    Within each line, strokes are sorted left-to-right by X.
    Lines are sorted top-to-bottom by Y.
    
    Args:
        strokes: List of stroke dictionaries
        threshold: Max Y-distance to consider strokes on same line (default: 30px)
    
    Returns:
        List of lines, where each line is a list of strokes sorted by X
    """
    if not strokes:
        return []
    
    # Calculate Y-center for each stroke
    strokes_with_y = [(s, _get_stroke_y_center(s), _get_stroke_x_center(s)) for s in strokes]
    
    # Sort by Y first
    strokes_with_y.sort(key=lambda x: x[1])
    
    lines = []
    current_line = []
    current_line_y = None
    
    for stroke, y_center, x_center in strokes_with_y:
        if current_line_y is None:
            # First stroke
            current_line = [(stroke, x_center)]
            current_line_y = y_center
        elif abs(y_center - current_line_y) <= threshold:
            # Same line - add to current
            current_line.append((stroke, x_center))
            # Update line Y as average
            current_line_y = (current_line_y * (len(current_line) - 1) + y_center) / len(current_line)
        else:
            # New line - save current and start new
            # Sort current line by X (left to right)
            current_line.sort(key=lambda x: x[1])
            lines.append([s for s, _ in current_line])
            current_line = [(stroke, x_center)]
            current_line_y = y_center
    
    # Don't forget last line
    if current_line:
        current_line.sort(key=lambda x: x[1])
        lines.append([s for s, _ in current_line])
    
    return lines


def _create_line_batches(lines: List[List[Dict]], batch_size: int = MYSCRIPT_BATCH_SIZE) -> List[List[Dict]]:
    """
    Create batches of strokes, keeping lines together.
    
    Each batch contains up to batch_size strokes, but we try not to split lines.
    If a single line has more than batch_size strokes, it goes into its own batch.
    
    Args:
        lines: List of lines (each line is a list of strokes)
        batch_size: Maximum strokes per batch
    
    Returns:
        List of batches (each batch is a flat list of strokes)
    """
    batches = []
    current_batch = []
    current_count = 0
    
    for line in lines:
        line_count = len(line)
        
        if line_count > batch_size:
            # Line is too big - save current batch and put line in its own batch
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_count = 0
            batches.append(line)
        elif current_count + line_count > batch_size:
            # Adding this line would exceed batch size - start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = line.copy()
            current_count = line_count
        else:
            # Add line to current batch
            current_batch.extend(line)
            current_count += line_count
    
    # Don't forget last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def _recognize_batch(
    client: MyScriptOCR,
    batch: List[Dict],
    batch_index: int,
    width: int,
    height: int,
    language: LanguageCode,
) -> tuple[int, Optional[str]]:
    """
    Recognize a single batch of strokes.
    
    Args:
        client: MyScript client instance
        batch: List of strokes to recognize
        batch_index: Index of batch for ordering results
        width, height: Document dimensions
        language: OCR language
    
    Returns:
        Tuple of (batch_index, recognized_text or None)
    """
    try:
        stroke_groups = convert_rm_strokes_to_myscript(batch, time_offset=0)
        
        if not stroke_groups or not any(sg.strokes for sg in stroke_groups):
            return (batch_index, None)
        
        request = MyScriptRequest(
            width=width,
            height=height,
            language=language,
            stroke_groups=stroke_groups,
        )
        
        result = client.recognize(request)
        text = result.get("label", "")
        return (batch_index, text.strip() if text else None)
    except Exception:
        return (batch_index, None)


# Number of parallel threads for MyScript API calls
MYSCRIPT_PARALLEL_THREADS = 7


def ocr_rm_file_with_myscript(
    rm_data: bytes, 
    batch_size: int = MYSCRIPT_BATCH_SIZE,
    line_threshold: float = MYSCRIPT_LINE_THRESHOLD,
    parallel_threads: int = MYSCRIPT_PARALLEL_THREADS,
) -> Optional[str]:
    """
    Recognize text from .rm file using MyScript.

    For large pages with many strokes, automatically batches requests
    to avoid MyScript API payload size limits (413 error).
    
    Strokes are grouped by lines (based on Y-coordinate proximity) and
    batches are created to keep lines together for better OCR accuracy.
    
    Batches are processed in parallel (default: 7 threads) for speed,
    then results are merged in the correct order.

    Args:
        rm_data: Contents of .rm file
        batch_size: Maximum strokes per API request (default: 500)
        line_threshold: Y-distance threshold to group strokes into lines (default: 30px)
        parallel_threads: Number of parallel threads for API calls (default: 7)

    Returns:
        Recognized text or None on error
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        # Parse .rm file
        parsed_data = parse_rm_file(rm_data)

        strokes = parsed_data.get("strokes", [])
        if not strokes:
            return None

        width = parsed_data.get("width", 1872)
        height = parsed_data.get("height", 1404)
        language = get_language_from_env()

        # Group strokes by lines (top-to-bottom, left-to-right within each line)
        lines = _group_strokes_by_lines(strokes, threshold=line_threshold)
        
        # Flatten for single batch if small enough
        all_strokes_sorted = [s for line in lines for s in line]

        # If strokes fit in one batch, use simple path
        if len(all_strokes_sorted) <= batch_size:
            stroke_groups = convert_rm_strokes_to_myscript(all_strokes_sorted, time_offset=0)
            
            if not stroke_groups or not any(sg.strokes for sg in stroke_groups):
                return None

            request = MyScriptRequest(
                width=width,
                height=height,
                language=language,
                stroke_groups=stroke_groups,
            )

            client = MyScriptOCR()
            result = client.recognize(request)
            return result.get("label", "")

        # Large page: create batches keeping lines together
        batches = _create_line_batches(lines, batch_size=batch_size)
        
        if not batches:
            return None
        
        # Process batches in parallel
        client = MyScriptOCR()
        results_dict: Dict[int, Optional[str]] = {}
        
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            # Submit all batches
            futures = {
                executor.submit(
                    _recognize_batch, client, batch, idx, width, height, language
                ): idx
                for idx, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                batch_idx, text = future.result()
                results_dict[batch_idx] = text
        
        # Merge results in correct order
        ordered_results = []
        for idx in range(len(batches)):
            text = results_dict.get(idx)
            if text:
                ordered_results.append(text)
        
        return "\n".join(ordered_results) if ordered_results else None

    except Exception:
        # MyScript error - return None to allow fallback
        return None

#!/usr/bin/env python3
"""
Generate animated GIF from meeting scheduling logs.
Shows two blackboards with agents coordinating meetings through chat.
"""

import json
import os
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from fontTools.ttLib import TTFont, TTLibError

RESAMPLING_NAMESPACE = getattr(Image, "Resampling", Image)
RESAMPLING_LANCZOS = getattr(RESAMPLING_NAMESPACE, "LANCZOS", None)
if RESAMPLING_LANCZOS is None:
    RESAMPLING_LANCZOS = getattr(Image, "LANCZOS", None)
if RESAMPLING_LANCZOS is None:
    RESAMPLING_LANCZOS = getattr(RESAMPLING_NAMESPACE, "BICUBIC", None)
if RESAMPLING_LANCZOS is None:
    RESAMPLING_LANCZOS = getattr(Image, "BICUBIC", None)
if RESAMPLING_LANCZOS is None:
    RESAMPLING_LANCZOS = getattr(Image, "NEAREST", 0)

# Configuration
WIDTH = 1280
HEIGHT = 1080
FPS = 30
TOTAL_DURATION = 36  # seconds (base timeline before playback speed multiplier)
PLAYBACK_SPEED = 1.5  # 50% faster playback
CHARS_PER_FRAME = 3  # Typing speed
MIN_CHARS_PER_FRAME = 1.0
TYPING_SPEED_BOOST = 1.05


# Configuration
# WIDTH = 1000
# HEIGHT = 1000
# FPS = 25
# TOTAL_DURATION = 10  # seconds (base timeline before playback speed multiplier)
# PLAYBACK_SPEED = 0.85  # slow the playback slightly for clarity
# CHARS_PER_FRAME = 45  # Typing speed
# MIN_CHARS_PER_FRAME = 15
# TYPING_SPEED_BOOST = 1.05
# FINAL_HOLD_SECONDS = 0.75
# BLACKBOARD_SPEED_MULTIPLIERS = {
#     0: 1.0,
#     1: 1.3,
# }
# GIF_MAX_COLORS = 160
# GIF_SIZE_LIMIT_BYTES = 5 * 1024 * 1024
# FRAME_STEP_CANDIDATES = [1, 2, 3, 4, 5, 6, 8, 10]

# Professional color scheme
COLORS = {
    'Jordan': '#2E5090',    # Deep Blue
    'Riley': '#2D7A5E',     # Forest Green
    'Avery': '#6B5B93',     # Muted Purple
    'Taylor': '#CC7A3C',    # Warm Orange
    'background': '#F5F5F5',  # Light gray
    'text': '#333333',      # Dark gray
    'system': '#999999',    # Light gray for system
    'divider': '#CCCCCC',   # Divider line
    'header_bg': '#FFFFFF', # White header background
    'action': '#D32F2F',    # Red for actions
}

# Meeting participant mapping
MEETING_PARTICIPANTS = {
    0: ['Jordan', 'Avery'],
    1: ['Jordan', 'Riley', 'Taylor']
}

MEETING_INFO = {
    0: 'M001 - Isenberg',
    1: 'M002 - CS Building'
}

MEETING_OWNERS = {
    0: 'Jordan',
    1: 'Riley',
}

WATERMARK_TEXT = "^*Generated using a real trajectory"
WATERMARK_COLOR = '#777777'

EMOJI_ICON_MAP = {
    'get_blackboard_events': '📖',
    'post_message': '✍️',
    'schedule_meeting': '📅'
}
DEFAULT_EMOJI_ICON = '🔧'
EMOJI_CHAR_SET = set(EMOJI_ICON_MAP.values()) | {DEFAULT_EMOJI_ICON}


def is_variation_selector(ch: str) -> bool:
    code = ord(ch)
    return (0xFE00 <= code <= 0xFE0F) or (0xE0100 <= code <= 0xE01EF)


def is_skin_tone_modifier(ch: str) -> bool:
    code = ord(ch)
    return 0x1F3FB <= code <= 0x1F3FF


def split_graphemes(text: str) -> list[str]:
    clusters: list[str] = []
    cluster = ''
    for ch in text:
        if not cluster:
            cluster = ch
            continue

        if is_variation_selector(ch) or is_skin_tone_modifier(ch):
            cluster += ch
            continue

        if ch == '\u200d':  # zero-width joiner keeps cluster open
            cluster += ch
            continue

        if cluster.endswith('\u200d'):
            cluster += ch
            continue

        clusters.append(cluster)
        cluster = ch

    if cluster:
        clusters.append(cluster)

    return clusters


def is_emoji_cluster(cluster: str) -> bool:
    if cluster in EMOJI_CHAR_SET:
        return True
    for ch in cluster:
        if ch == '\u200d' or is_variation_selector(ch) or is_skin_tone_modifier(ch):
            continue
        code = ord(ch)
        if (
            0x1F300 <= code <= 0x1FAFF
            or 0x1F900 <= code <= 0x1F9FF
            or 0x2700 <= code <= 0x27BF
        ):
            return True
    return False


def cluster_width(cluster: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox(cluster)
    return max(0, bbox[2] - bbox[0])


def get_bitmap_emoji(cluster: str, fonts: dict[str, ImageFont.FreeTypeFont], target_height: int) -> Image.Image | None:
    cache = fonts.get('emoji_bitmap_cache')
    tt_font = fonts.get('emoji_bitmap')
    strike = fonts.get('emoji_bitmap_strike')
    cmap = fonts.get('emoji_bitmap_cmap')

    if cache is None or tt_font is None or strike is None or cmap is None:
        return None

    key = (cluster, target_height)
    if key in cache:
        return cache[key]

    cleaned = ''.join(ch for ch in cluster if ch != '\ufe0f')
    if not cleaned or '\u200d' in cleaned or len(cleaned) > 2:
        cache[key] = None
        return None

    codepoints = [ord(ch) for ch in cleaned]
    if len(codepoints) != 1:
        cache[key] = None
        return None

    glyph_name = cmap.get(codepoints[0])
    if not glyph_name:
        cache[key] = None
        return None

    bitmap = strike.get(glyph_name)
    if not bitmap:
        cache[key] = None
        return None

    try:
        img = Image.open(BytesIO(bitmap.imageData)).convert('RGBA')
    except (OSError, UnidentifiedImageError, ValueError):
        cache[key] = None
        return None

    if target_height and img.height:
        scale = target_height / img.height
        new_width = max(1, int(img.width * scale))
        new_height = max(1, int(img.height * scale))
        if (new_width, new_height) != img.size:
            img = img.resize((new_width, new_height), RESAMPLING_LANCZOS)

    cache[key] = img
    return img


def measure_text(text: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> int:
    base_font = fonts['small']
    emoji_font = fonts.get('emoji')
    target_height = base_font.getbbox('Hg')[3] - base_font.getbbox('Hg')[1]

    total = 0
    for cluster in split_graphemes(text):
        if emoji_font and is_emoji_cluster(cluster):
            total += cluster_width(cluster, emoji_font)
        elif is_emoji_cluster(cluster):
            img = get_bitmap_emoji(cluster, fonts, target_height)
            if img:
                total += img.width + 2
            else:
                total += cluster_width(cluster, base_font)
        else:
            total += cluster_width(cluster, base_font)
    return total


def wrap_text(text: str, fonts: dict[str, ImageFont.FreeTypeFont], max_width: int) -> list[str]:
    if not text:
        return ['']

    words = text.split(' ')
    lines: list[str] = []
    current = ''

    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if measure_text(candidate, fonts) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
        current = word

        while measure_text(current, fonts) > max_width and len(split_graphemes(current)) > 1:
            partial = ''
            remainder = ''
            for cluster in split_graphemes(current):
                tentative = partial + cluster
                if not partial or measure_text(tentative, fonts) <= max_width:
                    partial = tentative
                else:
                    remainder = current[len(partial):]
                    break
            if partial:
                lines.append(partial.rstrip())
            current = remainder.strip()
            if not current:
                break

    if current:
        lines.append(current)

    return lines if lines else ['']


def draw_text_with_fallback(draw: ImageDraw.Draw, image: Image.Image, position: tuple[int, int], text: str,
                            fonts: dict[str, ImageFont.FreeTypeFont], fill: tuple[int, int, int]):
    base_font = fonts['small']
    emoji_font = fonts.get('emoji')
    emoji_embedded = fonts.get('emoji_embedded', False)
    target_height = base_font.getbbox('Hg')[3] - base_font.getbbox('Hg')[1]

    x, y = position
    for cluster in split_graphemes(text):
        font = base_font
        kwargs = {}
        advance = 0

        if emoji_font and is_emoji_cluster(cluster):
            font = emoji_font
            if emoji_embedded:
                kwargs['embedded_color'] = True
            draw.text((x, y), cluster, font=font, **kwargs)
            advance = cluster_width(cluster, font)
        elif is_emoji_cluster(cluster):
            img_bitmap = get_bitmap_emoji(cluster, fonts, target_height)
            if img_bitmap:
                icon_y = y + max(0, (target_height - img_bitmap.height) // 2)
                image.paste(img_bitmap, (int(x), int(icon_y)), img_bitmap)
                advance = img_bitmap.width + 2
            else:
                draw.text((x, y), cluster, font=font, fill=fill)
                advance = cluster_width(cluster, font)
        else:
            draw.text((x, y), cluster, font=font, fill=fill)
            advance = cluster_width(cluster, font)

        x += advance


class MeetingEvent:
    """Represents a single event in the meeting coordination."""
    def __init__(self, timestamp: str | float, agent: str, blackboard: int,
                 message: str, event_type: str, round_num: str = None):
        self.timestamp = timestamp
        self.agent = agent
        self.blackboard = blackboard
        self.message = message
        self.event_type = event_type  # 'communication' or 'action'
        self.round_num = round_num
        self.is_tool_call = False
        self.has_tool_indicator = False  # Whether to show tool indicator before this message
        self.tool_name = None  # Name of tool used (if any)
        # Initialized here to avoid attribute-defined-outside-init
        self.time_value: float = 0.0
        self.sequence: int = 0


class ToolCallEvent:
    """Represents a tool call made by an agent."""
    def __init__(self, timestamp: str | float, agent: str, blackboard: int,
                 tool_name: str, round_num: str = None, params: dict = None):
        self.timestamp = timestamp
        self.agent = agent
        self.blackboard = blackboard
        self.tool_name = tool_name
        self.round_num = round_num
        self.params = params or {}
        self.is_tool_call = True
        self.emoji = EMOJI_ICON_MAP.get(tool_name, DEFAULT_EMOJI_ICON)
        # Initialized here to avoid attribute-defined-outside-init
        self.time_value: float = 0.0
        self.sequence: int = 0
        # Create display message based on tool type
        if tool_name == 'get_blackboard_events':
            base_msg = f"checking blackboard {blackboard}..."
        elif tool_name == 'post_message':
            base_msg = "writing message..."
        elif tool_name == 'schedule_meeting':
            meeting_id = params.get('meeting_id', '?')
            base_msg = f"scheduling {meeting_id}..."
        else:
            base_msg = f"using {tool_name}..."

        self.message = f"{self.emoji} {base_msg} -> {tool_name}()"


SF_PRO_FONT_PATTERNS = {
    'title': [
        'SF-Pro-Display-Semibold.otf',
        'SF-Pro-Display-Semibold.ttf',
        'SFProDisplay-Semibold.otf',
        'SFProDisplay-Semibold.ttf',
        'SF-Pro-Display-Bold.otf',
        'SFProDisplay-Bold.ttf',
    ],
    'header': [
        'SF-Pro-Display-Medium.otf',
        'SF-Pro-Display-Medium.ttf',
        'SFProDisplay-Medium.otf',
        'SFProDisplay-Medium.ttf',
        'SF-Pro-Display-Semibold.otf',
    ],
    'small_bold': [
        'SF-Pro-Text-Semibold.otf',
        'SF-Pro-Text-Semibold.ttf',
        'SFProText-Semibold.otf',
        'SFProText-Semibold.ttf',
        'SF-Pro-Text-Bold.otf',
        'SFProText-Bold.ttf',
    ],
    'small': [
        'SF-Pro-Text-Regular.otf',
        'SF-Pro-Text-Regular.ttf',
        'SFProText-Regular.otf',
        'SFProText-Regular.ttf',
    ],
}

FALLBACK_FONT_PATHS = {
    'title': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ],
    'header': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
    ],
    'small_bold': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    ],
    'small': [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ],
}

EMOJI_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf',
    '/System/Library/Fonts/Apple Color Emoji.ttc',
    '/System/Library/Fonts/AppleColorEmoji.ttf',
]


def _get_sf_pro_search_dirs() -> list[Path]:
    """Collect directories to search for SF Pro fonts."""
    env_dirs = []
    for env_var in ("SF_PRO_FONT_DIR", "SF_PRO_DIR", "SFPRO_FONT_DIR"):
        value = os.environ.get(env_var)
        if value:
            env_dirs.append(Path(value))

    candidate_dirs = [
        Path('/System/Library/Fonts'),
        Path('/Library/Fonts'),
        Path.home() / 'Library' / 'Fonts',
        Path('/System/Library/AssetsV2/com_apple_MobileAsset_Font6'),
    ]

    dirs: list[Path] = []
    for directory in env_dirs + candidate_dirs:
        if directory and directory.exists():
            resolved = directory.resolve()
            if resolved not in dirs:
                dirs.append(resolved)
    return dirs


def _find_sf_pro_font(patterns: list[str]) -> str | None:
    """Return the first SF Pro font path matching the provided patterns."""
    for directory in _get_sf_pro_search_dirs():
        for pattern in patterns:
            # Search relative to directory without descending into massive trees unnecessarily
            matches = list(directory.glob(pattern))
            if not matches and ('**/' not in pattern):
                matches = list(directory.glob(f'**/{pattern}'))
            for match in matches:
                if match.exists():
                    return str(match.resolve())
    return None


def _try_load_font(path: str, size: int) -> ImageFont.FreeTypeFont | None:
    """Attempt to load a font with multiple layout engines."""
    layout_engines: list[int] = []
    basic = getattr(ImageFont, 'LAYOUT_BASIC', None)
    if basic is not None:
        layout_engines.append(basic)
    raqm = getattr(ImageFont, 'LAYOUT_RAQM', None)
    if raqm is not None and raqm not in layout_engines:
        layout_engines.append(raqm)
    if not layout_engines:
        layout_engines = [None]

    for engine in layout_engines:
        try:
            if engine is None:
                return ImageFont.truetype(path, size)
            return ImageFont.truetype(path, size, layout_engine=engine)
        except OSError:
            continue
    return None


def _load_font_from_candidates(candidates: list[str], size: int) -> tuple[ImageFont.FreeTypeFont | None, str | None]:
    """Attempt to load the first working font from candidates."""
    for path in candidates:
        font = _try_load_font(path, size)
        if font is not None:
            return font, path
    return None, None


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Load fonts, preferring SF Pro when available."""
    fonts: dict[str, ImageFont.FreeTypeFont] = {}

    font_sizes = {
        'title': 32,
        'header': 28,
        'small_bold': 18,
        'small': 18,
    }

    for key, size in font_sizes.items():
        candidates: list[str] = []
        sf_candidate = _find_sf_pro_font(SF_PRO_FONT_PATTERNS.get(key, []))
        if sf_candidate:
            candidates.append(sf_candidate)
        candidates.extend(FALLBACK_FONT_PATHS.get(key, []))

        font, font_path = _load_font_from_candidates(candidates, size)
        if font is None:
            print(f"Warning: Could not load font for '{key}', using default.")
            font = ImageFont.load_default()
            font_path = None
        fonts[key] = font
        if font_path and sf_candidate and 'SF' in Path(font_path).name:
            print(f"Loaded SF Pro font for '{key}' from {font_path}")

    emoji_candidates: list[str] = []
    env_emoji = os.environ.get('EMOJI_FONT_PATH')
    if env_emoji:
        emoji_candidates.append(env_emoji)
    emoji_candidates.extend(EMOJI_FONT_CANDIDATES)
    emoji_font = None
    emoji_font_path = None
    for size in (48, 40, 36, 32, 28, 24):
        emoji_font, emoji_font_path = _load_font_from_candidates(emoji_candidates, size)
        if emoji_font:
            break
    if emoji_font is None:
        print("Info: Emoji font not found; falling back to bitmap extraction.")
    fonts['emoji'] = emoji_font
    fonts['emoji_embedded'] = bool(emoji_font and emoji_font_path and ('emoji' in Path(emoji_font_path).name.lower()))

    emoji_bitmap_font = None
    emoji_bitmap_cmap = None
    emoji_bitmap_strike = None
    if emoji_font_path is None:
        for candidate in emoji_candidates:
            if Path(candidate).exists():
                emoji_font_path = candidate
                break
    if emoji_font_path and Path(emoji_font_path).exists():
        try:
            emoji_bitmap_font = TTFont(emoji_font_path)
            emoji_bitmap_cmap = emoji_bitmap_font.getBestCmap()
            strikes = emoji_bitmap_font['CBDT'].strikeData
            emoji_bitmap_strike = strikes[0] if strikes else None
        except (TTLibError, KeyError, OSError, AttributeError):
            emoji_bitmap_font = None
            emoji_bitmap_cmap = None
            emoji_bitmap_strike = None

    fonts['emoji_bitmap'] = emoji_bitmap_font
    fonts['emoji_bitmap_cmap'] = emoji_bitmap_cmap
    fonts['emoji_bitmap_strike'] = emoji_bitmap_strike
    fonts['emoji_bitmap_cache'] = {}

    return fonts


def parse_logs(log_dir: str) -> list[MeetingEvent | ToolCallEvent]:
    """Parse log files and extract events chronologically."""
    log_path = Path(log_dir) / "seed_42"

    # First, parse tool calls and create a lookup dict
    tool_calls_by_agent_time = {}  # Key: (agent, time, blackboard) -> tool info
    tool_events: list[ToolCallEvent] = []
    sequence_counter = 0

    def time_to_float(dt_obj: datetime) -> float:
        """Convert datetime to seconds since midnight."""
        return (
            dt_obj.hour * 3600
            + dt_obj.minute * 60
            + dt_obj.second
            + dt_obj.microsecond / 1_000_000
        )

    try:
        with open(log_path / "tool_calls.json", 'r', encoding='utf-8') as tool_calls_file:
            tool_calls_data = json.load(tool_calls_file)

        for call in tool_calls_data:
            tool_name = call.get('tool_name')
            agent = call.get('agent_name')
            timestamp_raw = call.get('timestamp', 0)
            phase = call.get('phase', 'planning')
            round_num = call.get('round')
            params = call.get('parameters', {})

            # Convert timestamp to time string for consistency (HH:MM:SS format)
            time_value = None
            try:
                if isinstance(timestamp_raw, str):
                    # Parse ISO format timestamp
                    if 'T' in timestamp_raw:
                        dt = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M:%S')
                        time_value = time_to_float(dt)
                    else:
                        # Already a time string, use it directly
                        time_str = timestamp_raw
                        dt = datetime.strptime(time_str, '%H:%M:%S')
                        time_value = time_to_float(dt)
                else:
                    # Unix timestamp, convert it
                    dt = datetime.fromtimestamp(timestamp_raw)
                    time_str = dt.strftime('%H:%M:%S')
                    time_value = time_to_float(dt)
            except (ValueError, TypeError) as e:
                # Fallback
                print(f"Warning: Could not parse timestamp {timestamp_raw}: {e}")
                time_str = "00:00:00"
                time_value = 0.0

            blackboard = params.get('blackboard_id')

            if tool_name == 'schedule_meeting':
                meeting_id = params.get('meeting_id')
                if meeting_id == 'M001':
                    blackboard = 0
                elif meeting_id == 'M002':
                    blackboard = 1

                owner = MEETING_OWNERS.get(blackboard)
                if owner and agent != owner:
                    continue

            if blackboard is None or agent is None:
                continue

            round_str = 'Execution' if phase == 'execution' else f"Round {round_num}" if round_num else 'Planning'

            if tool_name == 'post_message':
                key = (agent, time_str, blackboard)
                tool_calls_by_agent_time[key] = {
                    'tool_name': tool_name,
                    'timestamp': time_str,
                    'phase': phase,
                    'round': round_num,
                    'time_value': time_value if time_value is not None else 0.0
                }

            tool_event = ToolCallEvent(
                timestamp=time_str,
                agent=agent,
                blackboard=blackboard,
                tool_name=tool_name,
                round_num=round_str,
                params=params
            )
            tool_event.time_value = time_value if time_value is not None else 0.0
            tool_event.sequence = sequence_counter
            sequence_counter += 1
            tool_events.append(tool_event)
    except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load tool_calls.json: {e}")

    # Parse blackboard files for message events
    events = []
    for blackboard_id in [0, 1]:
        blackboard_file = log_path / f"blackboard_{blackboard_id}.txt"
        with open(blackboard_file, 'r', encoding='utf-8') as blackboard_file_handle:
            content = blackboard_file_handle.read()

        # Parse events from blackboard - new format
        # [Event #X, Iteration: Y] [HH:MM:SS] [Phase] Agent (event_type)  Content/Message: ...
        lines = content.strip().split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for event lines starting with [Event
            if line.startswith('[Event'):
                try:
                    # Extract metadata
                    # [Event #1, Iteration: 1] [16:15:50] [Planning] Jordan (communication)  Content: ...

                    # Find the second bracket pair for timestamp
                    parts = line.split(']')
                    if len(parts) < 4:
                        i += 1
                        continue

                    # Extract time (HH:MM:SS format in second bracket)
                    raw_time = parts[1].strip()
                    time_str = raw_time.strip('[]')
                    try:
                        message_time_value = time_to_float(datetime.strptime(time_str, "%H:%M:%S"))
                    except ValueError:
                        message_time_value = 0.0

                    # Extract phase (Planning/Execution)
                    phase_str = parts[2].strip().strip('[]')

                    # Extract agent and event type from remaining part
                    rest = ']'.join(parts[3:]).strip()

                    # Find agent name and event type
                    if '(' not in rest or ')' not in rest:
                        i += 1
                        continue

                    paren_start = rest.index('(')
                    paren_end = rest.index(')')

                    agent = rest[:paren_start].strip()
                    event_type = rest[paren_start+1:paren_end].strip()

                    # Extract message content
                    message_part = rest[paren_end+1:].strip()

                    # Skip system context messages
                    if event_type == 'context' or agent == 'SYSTEM':
                        i += 1
                        continue

                    message = ""

                    # Handle different content formats
                    if 'Content:' in message_part:
                        message = message_part.split('Content:', 1)[1].strip()
                    elif 'Message:' in message_part:
                        message = message_part.split('Message:', 1)[1].strip()
                        message = message.strip('"')
                    elif 'Action_Type:' in message_part:
                        # For action events, collect all lines until blank line
                        action_type = message_part.split('Action_Type:', 1)[1].strip()
                        message = f"Scheduled {action_type}"

                        # Look ahead for action params
                        j = i + 1
                        while j < len(lines) and lines[j].strip():
                            if 'Action_Params:' in lines[j]:
                                # Extract meeting info from params
                                params_line = lines[j]
                                if "'meeting_id':" in params_line:
                                    meeting_match = re.search(r"'meeting_id':\s*'(\w+)'", params_line)
                                    slot_match = re.search(r"'slot':\s*(\d+)", params_line)
                                    if meeting_match:
                                        meeting_id = meeting_match.group(1)
                                        slot_num = slot_match.group(1) if slot_match else '?'
                                        message = f"✓ Scheduled {meeting_id} for slot {slot_num}"
                            j += 1

                    # Determine round number and phase from the log
                    if phase_str == 'Execution':
                        round_num = 'Execution'
                        phase = 'Execution'
                    elif 'Planning' in phase_str:
                        # Extract round number if present in log
                        round_num = phase_str  # Will be updated with actual round later
                        phase = 'Planning'
                    else:
                        round_num = 'Planning'
                        phase = 'Planning'

                    if message:
                        if (
                            blackboard_id == 1
                            and agent == 'Jordan'
                            and message.startswith('✓ Scheduled M001')
                        ):
                            i += 1
                            continue

                        if (
                            event_type == 'action_executed'
                            and 'Scheduled' in message
                            and MEETING_OWNERS.get(blackboard_id) not in (None, agent)
                        ):
                            i += 1
                            continue

                        if event_type == 'action_executed':
                            whole_seconds = int(message_time_value)
                            message_time_value = whole_seconds + 0.999

                        event = MeetingEvent(
                            timestamp=time_str,
                            agent=agent,
                            blackboard=blackboard_id,
                            message=message,
                            event_type=event_type,
                            round_num=round_num
                        )

                        # Check if there's a matching tool call for this message
                        tool_key = (agent, time_str, blackboard_id)
                        if tool_key in tool_calls_by_agent_time:
                            tool_info = tool_calls_by_agent_time[tool_key]
                            event.tool_name = tool_info['tool_name']
                            # Use round info from tool call
                            round_from_tool = tool_info.get('round')
                            phase_from_tool = tool_info.get('phase')
                            if phase_from_tool == 'execution':
                                event.round_num = 'Execution'
                            elif round_from_tool:
                                event.round_num = f"Round {round_from_tool}"

                            tool_time_value = tool_info.get('time_value')
                            if tool_time_value is not None:
                                message_time_value = max(message_time_value, tool_time_value + 0.001)

                        event.time_value = message_time_value
                        event.sequence = sequence_counter
                        sequence_counter += 1
                        events.append(event)

                except (ValueError, IndexError) as e:
                    pass

            i += 1

    # Add tool call events
    events.extend(tool_events)

    # Sort events by timestamp
    events.sort(key=lambda e: (
        getattr(e, 'time_value', 0.0),
        getattr(e, 'sequence', 0)
    ))

    # Clean up round numbers - ensure they're properly formatted
    for event in events:
        if hasattr(event, 'round_num'):
            if not event.round_num or event.round_num == 'Planning':
                # Default to Round 1 for planning events without specific round
                event.round_num = 'Round 1'
            elif hasattr(event, 'event_type') and event.event_type == 'action_executed':
                event.round_num = 'Execution'

    # Debug logging
    print("\n=== EVENT SEQUENCE DEBUG ===")
    for i, event in enumerate(events):
        event_type_str = "TOOL" if hasattr(event, 'is_tool_call') and event.is_tool_call else "MSG"
        tool_indicator = " [has_tool]" if hasattr(event, 'has_tool_indicator') and event.has_tool_indicator else ""
        msg_preview = event.message[:50] + "..." if len(event.message) > 50 else event.message
        print(f"{i+1}. [{event.timestamp}] Board {event.blackboard} | {event.agent} | {event_type_str}{tool_indicator} | {event.round_num} | {msg_preview}")
    print("=== END DEBUG ===\n")

    return events


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def calculate_message_height(event: MeetingEvent | ToolCallEvent, fonts: dict,
                            message_width: int, visible_chars: int | None = None) -> int:
    """Calculate the height a message will take when rendered."""
    message = event.message if visible_chars is None else event.message[:visible_chars]

    # Determine label based on event type
    if hasattr(event, 'is_tool_call') and event.is_tool_call:
        label = f"[TOOL] {event.agent}:"
    elif hasattr(event, 'event_type') and event.event_type == 'action_executed':
        label = f"[ACTION] {event.agent}:"
    else:
        label = f"{event.agent}:"

    bbox = fonts['small_bold'].getbbox(label + "  ")
    label_width = bbox[2] - bbox[0]

    icon_indent = 0

    available_width = message_width - label_width - icon_indent
    available_width = max(50, available_width)

    wrapped_lines = wrap_text(message, fonts, available_width)
    num_lines = len(wrapped_lines) if wrapped_lines else 1

    return num_lines * 30 + 10  # 30 per line + 10 spacing


def create_frame(completed_events: list[MeetingEvent | ToolCallEvent],
                typing_events: dict[int, tuple[MeetingEvent | ToolCallEvent | None, int]],
                scroll_offsets: dict[int, int],
                current_round: str,
                fonts: dict) -> Image.Image:
    """Create a single frame of the animation.

    Args:
        completed_events: List of fully typed events (including tool calls)
        typing_events: Dict mapping blackboard_id to (event, chars_visible) tuple
        scroll_offsets: Dict mapping blackboard_id to scroll offset
        current_round: Current round name
        fonts: Font dictionary
    """
    # Create base image
    img = Image.new('RGB', (WIDTH, HEIGHT), hex_to_rgb(COLORS['background']))
    draw = ImageDraw.Draw(img)

    # Two-line title at top
    title_line1 = "Meeting Scheduling Environment"
    bbox1 = fonts['title'].getbbox(title_line1)
    title1_width = bbox1[2] - bbox1[0]
    draw.text(
        ((WIDTH - title1_width) // 2, 18),
        title_line1,
        fill=hex_to_rgb(COLORS['text']),
        font=fonts['title']
    )

    # Extract phase and round from current_round
    if current_round == 'Execution':
        phase = 'Execution'
        round_num = '1'
    elif 'Round' in current_round:
        phase = 'Planning'
        round_num = current_round.split('Round')[1].strip()
    else:
        phase = 'Planning'
        round_num = '1'

    title_line2 = f"Phase {phase} • Round {round_num}"
    bbox2 = fonts['header'].getbbox(title_line2)
    title2_width = bbox2[2] - bbox2[0]
    draw.text(
        ((WIDTH - title2_width) // 2, 58),
        title_line2,
        fill=hex_to_rgb(COLORS['text']),
        font=fonts['header']
    )

    accent_line_y = 94
    draw.line([(WIDTH // 2 - 150, accent_line_y), (WIDTH // 2 + 150, accent_line_y)],
              fill=hex_to_rgb(COLORS['divider']), width=2)

    # Divider line
    draw.line([(0, 105), (WIDTH, 105)], fill=hex_to_rgb(COLORS['divider']), width=2)

    # Vertical divider
    draw.line([(WIDTH // 2, 105), (WIDTH // 2, HEIGHT)],
              fill=hex_to_rgb(COLORS['divider']), width=2)

    # Draw both blackboards
    for board_id in [0, 1]:
        x_offset = 0 if board_id == 0 else WIDTH // 2
        board_width = WIDTH // 2

        # Header
        header_y = 125

        # Meeting title and subtitle styling
        meeting_title_raw = MEETING_INFO[board_id]
        if ' - ' in meeting_title_raw:
            meeting_code, meeting_location = meeting_title_raw.split(' - ', 1)
        else:
            meeting_code, meeting_location = meeting_title_raw, ''

        board_title = f"Blackboard {board_id} for {meeting_code}"
        location_text = f"Location: {meeting_location}" if meeting_location else ''

        title_bbox = fonts['header'].getbbox(board_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]

        if location_text:
            subtitle_bbox = fonts['small'].getbbox(location_text)
            subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
            subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]
        else:
            subtitle_width = 0
            subtitle_height = 0

        bubble_width = max(title_width, subtitle_width)
        bubble_padding_x = 28
        bubble_padding_y = 12
        bubble_inner_spacing = 6 if location_text else 0
        bubble_height = title_height + subtitle_height + bubble_inner_spacing

        bubble_left = x_offset + (board_width - bubble_width) // 2 - bubble_padding_x
        bubble_top = header_y - bubble_padding_y
        bubble_right = bubble_left + bubble_width + bubble_padding_x * 2
        bubble_bottom = bubble_top + bubble_height + bubble_padding_y * 2

        draw.rounded_rectangle(
            [bubble_left, bubble_top, bubble_right, bubble_bottom],
            radius=16,
            fill=hex_to_rgb(COLORS['header_bg']),
            outline=hex_to_rgb(COLORS['divider'])
        )

        title_x = x_offset + (board_width - title_width) // 2
        title_y = bubble_top + bubble_padding_y
        draw.text(
            (title_x, title_y),
            board_title,
            fill=hex_to_rgb(COLORS['text']),
            font=fonts['header']
        )

        if location_text:
            subtitle_x = x_offset + (board_width - subtitle_width) // 2
            subtitle_y = title_y + title_height + bubble_inner_spacing
            draw.text(
                (subtitle_x, subtitle_y),
                location_text,
                fill=hex_to_rgb(COLORS['system']),
                font=fonts['small']
            )

        agent_y = bubble_bottom + 18

        # Agent names
        agents = MEETING_PARTICIPANTS[board_id]

        # Draw each agent name in their color
        x_pos = x_offset + 40
        draw.text((x_pos, agent_y), "Participants: ",
                 fill=hex_to_rgb(COLORS['text']), font=fonts['small'])

        bbox = fonts['small'].getbbox("Participants: ")
        x_pos += bbox[2] - bbox[0]

        for i, agent in enumerate(agents):
            if i > 0:
                draw.text((x_pos, agent_y), " • ",
                         fill=hex_to_rgb(COLORS['text']), font=fonts['small'])
                bbox = fonts['small'].getbbox(" • ")
                x_pos += bbox[2] - bbox[0]

            draw.text((x_pos, agent_y), agent,
                     fill=hex_to_rgb(COLORS[agent]), font=fonts['small_bold'])
            bbox = fonts['small_bold'].getbbox(agent)
            x_pos += bbox[2] - bbox[0]

        # Messages area
        message_start_y = 225
        message_x = x_offset + 40
        message_width = board_width - 80
        viewport_height = HEIGHT - message_start_y - 50  # Leave some bottom margin

        # Get scroll offset for this blackboard
        scroll_offset = scroll_offsets.get(board_id, 0)

        # Collect all events for this blackboard
        board_completed = [e for e in completed_events if e.blackboard == board_id]

        # Check if we're typing on this board
        board_typing = None
        board_typing_chars = 0
        if board_id in typing_events:
            board_typing, board_typing_chars = typing_events[board_id]

        # Combine all events to display
        all_board_events = board_completed + ([board_typing] if board_typing and board_typing_chars > 0 else [])

        # Render messages with scrolling
        message_y = message_start_y - scroll_offset

        for i, event in enumerate(all_board_events):
            # Determine if this is the typing event
            is_typing = event == board_typing
            visible_text = event.message[:board_typing_chars] if is_typing else event.message

            # Skip if completely above viewport
            message_height = calculate_message_height(event, fonts, message_width,
                                                     board_typing_chars if is_typing else None)
            if message_y + message_height < message_start_y:
                message_y += message_height
                continue

            # Stop if below viewport
            if message_y >= message_start_y + viewport_height:
                break

            # Agent name label and styling
            agent_color = COLORS.get(event.agent, COLORS['text'])
            label = f"{event.agent}:"
            text_color = COLORS['text']

            # Style based on event type
            if hasattr(event, 'is_tool_call') and event.is_tool_call:
                label = f"[TOOL] {event.agent}:"
                agent_color = COLORS['system']  # Gray color for tool calls
                text_color = COLORS['system']   # Make tool text also gray
            elif hasattr(event, 'event_type') and event.event_type == 'action_executed':
                label = f"[ACTION] {event.agent}:"
                agent_color = COLORS['action']

            # Only draw if within viewport
            if message_y >= message_start_y:
                draw.text((message_x, message_y), label,
                         fill=hex_to_rgb(agent_color), font=fonts['small_bold'])

            # Message text (wrapped)
            bbox = fonts['small_bold'].getbbox(label + "  ")
            label_width = bbox[2] - bbox[0]

            icon_indent = 0

            available_width = message_width - label_width - icon_indent
            available_width = max(50, available_width)

            wrapped_lines = wrap_text(visible_text, fonts, available_width)

            # First line
            if wrapped_lines and message_y >= message_start_y:
                text_start_x = message_x + label_width

                draw_text_with_fallback(
                    draw,
                    img,
                    (text_start_x, message_y),
                    wrapped_lines[0],
                    fonts,
                    hex_to_rgb(text_color)
                )

            message_y += 30

            # Remaining lines
            text_indent = message_x + label_width + icon_indent
            if text_indent < message_x + 20:
                text_indent = message_x + 20

            for line in wrapped_lines[1:]:
                if message_y >= message_start_y and message_y <= message_start_y + viewport_height:
                    draw_text_with_fallback(
                        draw,
                        img,
                        (text_indent, message_y),
                        line,
                        fonts,
                        hex_to_rgb(text_color)
                    )
                message_y += 30

            message_y += 10  # Space between messages

    # Watermark (bottom right)
    watermark_font = fonts.get('small', ImageFont.load_default())
    wm_bbox = watermark_font.getbbox(WATERMARK_TEXT)
    wm_width = wm_bbox[2] - wm_bbox[0]
    wm_height = wm_bbox[3] - wm_bbox[1]
    margin = 20
    wm_x = WIDTH - wm_width - margin
    wm_y = HEIGHT - wm_height - margin
    draw.text((wm_x, wm_y), WATERMARK_TEXT, fill=hex_to_rgb(WATERMARK_COLOR), font=watermark_font)

    return img


def generate_gif(log_dir: str, output_path: str):
    """Generate the animated GIF."""
    print("Parsing logs...")
    events = parse_logs(log_dir)

    if not events:
        print("No events found in logs!")
        return

    print(f"Found {len(events)} events")

    fonts = load_fonts()

    # Calculate timing - 60% slower means multiply by 0.4
    total_frames = int(TOTAL_DURATION * FPS)
    playback_fps = FPS * PLAYBACK_SPEED

    # Separate events by blackboard for parallel processing
    events_by_board = {0: [], 1: []}
    for event in events:
        events_by_board[event.blackboard].append(event)

    # Calculate typing speed per blackboard
    chars_per_frame_board = {}
    for board_id in [0, 1]:
        total_chars_board = sum(len(e.message) for e in events_by_board[board_id])
        if total_chars_board == 0:
            chars_per_frame_board[board_id] = 0.0
        else:
            avg_required = (total_chars_board / total_frames) * TYPING_SPEED_BOOST
            speed_multiplier = BLACKBOARD_SPEED_MULTIPLIERS.get(board_id, 1.0)
            target_chars = avg_required * speed_multiplier
            chars_per_frame_board[board_id] = max(MIN_CHARS_PER_FRAME, target_chars)

    print(f"Generating {total_frames} frames at {FPS} FPS (playback {playback_fps:.2f} FPS)...")
    print(
        "Typing speed:" 
        f" Board 0: ~{chars_per_frame_board[0]:.2f},"
        f" Board 1: ~{chars_per_frame_board[1]:.2f} chars/frame"
    )

    frames = []

    # Track state independently for each blackboard
    board_state = {}
    for board_id in [0, 1]:
        board_state[board_id] = {
            'completed_events': [],
            'current_event_idx': 0,
            'current_event_chars': 0,
            'char_budget': 0.0,
        }

    current_round = "Round 1"

    # Message dimensions for scroll calculation
    message_width = (WIDTH // 2) - 80
    viewport_height = HEIGHT - 225 - 50

    for frame_num in range(total_frames):
        # Update typing for each blackboard independently
        for board_id in [0, 1]:
            state = board_state[board_id]
            board_events = events_by_board[board_id]

            if state['current_event_idx'] >= len(board_events):
                continue  # This board is done

            state['char_budget'] += chars_per_frame_board[board_id]

            while state['char_budget'] >= 1 and state['current_event_idx'] < len(board_events):
                event = board_events[state['current_event_idx']]
                event_length = len(event.message)
                remaining_in_event = event_length - state['current_event_chars']

                chars_available = min(int(state['char_budget']), remaining_in_event)
                if chars_available <= 0:
                    break

                state['current_event_chars'] += chars_available
                state['char_budget'] -= chars_available

                if state['current_event_chars'] >= event_length:
                    state['completed_events'].append(event)
                    current_round = event.round_num
                    state['current_event_idx'] += 1
                    state['current_event_chars'] = 0
                else:
                    current_round = event.round_num
                    break

        # Collect all completed events and current typing events
        all_completed = []
        typing_events = {}

        for board_id in [0, 1]:
            state = board_state[board_id]
            board_events = events_by_board[board_id]

            # Add completed events
            all_completed.extend(state['completed_events'])

            # Add typing event if active
            if state['current_event_idx'] < len(board_events) and state['current_event_chars'] > 0:
                typing_events[board_id] = (
                    board_events[state['current_event_idx']],
                    state['current_event_chars']
                )

        # Calculate scroll offsets for each blackboard
        scroll_offsets = {}
        for board_id in [0, 1]:
            state = board_state[board_id]

            # Get all events for this board (completed + typing)
            board_completed = state['completed_events']
            board_events_display = board_completed[:]

            if board_id in typing_events:
                typing_event, typing_chars = typing_events[board_id]
                if typing_chars > 0:
                    board_events_display.append(typing_event)

            # Calculate total height of messages (including partially typed entries)
            total_height = 0
            for e in board_events_display:
                is_typing = board_id in typing_events and e == typing_events[board_id][0]
                typing_chars_for_calc = typing_events[board_id][1] if is_typing else None

                msg_height = calculate_message_height(e, fonts, message_width, typing_chars_for_calc)
                total_height += msg_height

            # Calculate scroll offset to keep latest messages visible
            if total_height > viewport_height:
                scroll_offsets[board_id] = total_height - viewport_height
            else:
                scroll_offsets[board_id] = 0

        # Create frame
        frame = create_frame(all_completed, typing_events, scroll_offsets, current_round, fonts)
        frames.append(frame)

        if (frame_num + 1) % 50 == 0:
            print(f"  Generated {frame_num + 1}/{total_frames} frames...")

    frame_duration = 1.0 / playback_fps if playback_fps else 0.0
    durations = [frame_duration] * len(frames)
    if durations:
        durations[-1] = FINAL_HOLD_SECONDS

    durations_ms = [max(1, int(round(d * 1000))) for d in durations]

    frame_step_options = FRAME_STEP_CANDIDATES if FRAME_STEP_CANDIDATES else [1]
    file_size_bytes = 0
    total_generated_frames = len(frames)

    for step in frame_step_options:
        if step <= 0:
            continue

        export_frames: list[Image.Image] = []
        export_durations: list[int] = []

        for start_idx in range(0, total_generated_frames, step):
            end_idx = min(start_idx + step, total_generated_frames)
            frame = frames[start_idx]
            if GIF_MAX_COLORS:
                frame = frame.convert("P", palette=Image.ADAPTIVE, colors=GIF_MAX_COLORS)
            export_frames.append(frame)
            chunk_duration = sum(durations_ms[start_idx:end_idx])
            export_durations.append(max(1, chunk_duration))

        if not export_frames:
            continue

        first_frame, *rest_frames = export_frames
        reduced_count = len(export_frames)
        print(
            f"Saving GIF ({GIF_MAX_COLORS} colors, frame step {step}, "
            f"{reduced_count} frames) to {output_path}..."
        )
        first_frame.save(
            output_path,
            save_all=True,
            append_images=rest_frames,
            loop=0,
            duration=export_durations,
            disposal=2,
            optimize=True,
        )

        file_size_bytes = os.path.getsize(output_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        print(f"Done! GIF saved ({file_size_mb:.2f} MB)")

        if file_size_bytes <= GIF_SIZE_LIMIT_BYTES:
            break

    file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0
    if file_size_bytes > GIF_SIZE_LIMIT_BYTES:
        limit_mb = GIF_SIZE_LIMIT_BYTES / (1024 * 1024)
        print(
            f"Warning: GIF size {file_size_mb:.2f} MB exceeds limit of {limit_mb:.2f} MB "
            "after applying frame stepping."
        )


if __name__ == '__main__':
    DEFAULT_LOG_DIR = 'logs/MeetingScheduling/baseline_gpt-4.1-mini-2025-04-14/20251028-165118'
    DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_LOG_DIR, 'meeting_animation_small.gif')

    generate_gif(DEFAULT_LOG_DIR, DEFAULT_OUTPUT_PATH)

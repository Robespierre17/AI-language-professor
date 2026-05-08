"""Voice conversation loop: Whisper STT -> Claude language tutor -> ElevenLabs TTS.

The tutor switches target language mid-session when the learner asks
("let's switch to Hebrew", "practice Japanese now"). Language state is held in a
Session object and exposed to Claude via a set_target_language tool.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from openai import OpenAI

load_dotenv()

SAMPLE_RATE = 16_000
CHANNELS = 1
SILENCE_RMS = 350           # int16 RMS below this counts as silence
SILENCE_DURATION = 0.8      # seconds of silence after speech that ends a turn
MIN_SPEECH_DURATION = 0.4   # ignore silence until this much speech is heard
MAX_TURN_SECONDS = 60

CLAUDE_MODEL = "claude-sonnet-4-5"
EXTRACTOR_MODEL = "claude-haiku-4-5-20251001"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah

MEMORY_PATH = Path(__file__).parent / "memory.json"
VOCAB_SUMMARY_LIMIT = 30
MISTAKE_SUMMARY_LIMIT = 10

SUPPORTED_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "ru": "Russian",
    "ja": "Japanese",
    "fa": "Farsi",
    "ar": "Arabic",
    "sv": "Swedish",
}

NAME_TO_CODE = {name.lower(): code for code, name in SUPPORTED_LANGUAGES.items()}
NAME_TO_CODE["persian"] = "fa"  # common alias for Farsi

SUPPORTED_LIST = ", ".join(SUPPORTED_LANGUAGES.values())

# Languages Whisper is allowed to report. English is permitted as a transcription
# language (the learner often speaks English between turns) but not as a target.
ALLOWED_DETECTED = (
    set(SUPPORTED_LANGUAGES.keys())
    | {name.lower() for name in SUPPORTED_LANGUAGES.values()}
    | {"en", "english"}
)

# Navigation commands are spoken in English. If the first-pass transcription
# (biased toward the target language) contains any of these, we retry with
# Whisper hinted to English so the command isn't garbled.
SWITCH_PHRASE_HINTS = (
    "switch",
    "change to",
    "practice",
    "let me try",
    "let's try",
    "lets try",
    "let's do",
    "lets do",
    "speak ",
)


def looks_like_switch_request(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in SWITCH_PHRASE_HINTS)


SENT_END_CHARS = set(".!?。！？؟")


def _split_sentence(buf: str) -> tuple[str | None, str]:
    """Pull the first complete sentence out of `buf`. Returns (sentence, rest)
    or (None, buf) if no boundary yet. Newlines split. Other terminators only
    split when followed by whitespace, so '3.14' and 'wait...' aren't broken
    up while they're still streaming in.
    """
    i = 0
    while i < len(buf):
        ch = buf[i]
        if ch == "\n":
            sentence = buf[:i].strip()
            rest = buf[i + 1:].lstrip()
            if sentence:
                return sentence, rest
            i += 1
        elif ch in SENT_END_CHARS:
            j = i
            while j + 1 < len(buf) and buf[j + 1] in SENT_END_CHARS:
                j += 1
            if j + 1 >= len(buf):
                return None, buf
            if buf[j + 1].isspace():
                sentence = buf[:j + 1].strip()
                rest = buf[j + 1:].lstrip()
                if sentence:
                    return sentence, rest
            i = j + 1
        else:
            i += 1
    return None, buf


# Hardcoded voice IDs per language. Languages left out fall back to DEFAULT_VOICE_ID.
VOICE_MAP: dict[str, str] = {
    "he": "FpYM69yCAZCp21WhYw4m",  # Grossman audiobook clone
}

def load_memory() -> dict:
    """Read memory.json. Empty dict on first run or unreadable file (preserves
    the existing file on parse error so corrupted state isn't silently overwritten
    until the next save)."""
    if not MEMORY_PATH.exists():
        return {}
    try:
        return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  could not read {MEMORY_PATH.name}: {e}", file=sys.stderr)
        return {}


def save_memory(memory: dict) -> None:
    MEMORY_PATH.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def memory_slot(memory: dict, code: str) -> dict:
    return memory.setdefault(code, {"vocab": [], "mistakes": [], "preferences": []})


def format_memory_summary(memory: dict) -> str:
    """Render stored memory as a compact section to append to the system prompt."""
    if not memory:
        return ""
    lines = [
        "Prior-session memory for this learner. Honor preferences strictly; "
        "reinforce past vocab and corrections naturally without re-teaching what "
        "the learner already knows."
    ]
    for code in sorted(memory.keys()):
        slot = memory[code]
        prefs = slot.get("preferences", [])
        vocab = slot.get("vocab", [])
        mistakes = slot.get("mistakes", [])
        if not (prefs or vocab or mistakes):
            continue
        name = SUPPORTED_LANGUAGES.get(code, code)
        lines.append(f"\n[{name}]")
        if prefs:
            lines.append("Preferences (always honor):")
            for p in prefs:
                lines.append(f"- {p}")
        if vocab:
            recent = vocab[-VOCAB_SUMMARY_LIMIT:]
            words = ", ".join(v.get("word", "") for v in recent if v.get("word"))
            if words:
                lines.append(f"Vocab already taught (recent): {words}")
        if mistakes:
            recent = mistakes[-MISTAKE_SUMMARY_LIMIT:]
            lines.append("Past corrections to reinforce:")
            for m in recent:
                d, c = m.get("description", ""), m.get("correction", "")
                if d and c:
                    lines.append(f"- {d} → {c}")
    return "\n".join(lines)


SYSTEM_PROMPT_BASE = f"""You are an adaptive language tutor in a real-time spoken conversation with a learner.

The only supported target languages are: {SUPPORTED_LIST}. Never switch to anything else.

Each user turn is prefixed with a [Mode] note giving the current target language. Speak primarily in that language. Drop briefly into English only to clarify a tricky grammatical point or correct a serious error, then return.

Adapt the teaching tradition to the language:
- Hebrew/Arabic: emphasize triliteral roots, point out cognates within the Semitic family, note vowel patterns
- Japanese: highlight politeness levels and particle usage; give romaji for new vocabulary in early turns
- Spanish/French: connect via shared Latin roots, surface gender patterns
- Russian: explain case logic, verbal aspect, stress placement
- Farsi: note ezafe construction, Indo-European cognates despite Arabic script
- Swedish: pitch accent, en/ett gender, Germanic cognates with English

Switching rules — read carefully:
- Only call set_target_language when the learner explicitly names one of the supported languages ({SUPPORTED_LIST}) and clearly asks to switch, change, or practice that language.
- Speech-to-text occasionally mishears and produces nonsense or unrelated languages (e.g. Latin, Esperanto, garbled text). Treat anything that is not a clear request for one of the eight supported languages as a mishearing: stay in the current target language and gently ask the learner to repeat.
- Never switch to English, German, Italian, Portuguese, Korean, Mandarin, Hindi, Latin, or any other language. If the learner asks for one of those, briefly say it isn't supported and offer the eight that are.

After a successful switch, continue naturally in the new language with a short greeting that confirms the switch and invites a first exchange.

This is spoken conversation. Keep replies short — usually 2 to 4 sentences. No bullet points, no code blocks, no markdown. Speak like a friend tutoring you over coffee."""


def build_system_prompt(memory: dict) -> str:
    summary = format_memory_summary(memory)
    return f"{SYSTEM_PROMPT_BASE}\n\n{summary}" if summary else SYSTEM_PROMPT_BASE

SET_LANGUAGE_TOOL = {
    "name": "set_target_language",
    "description": (
        "Switch the active target language for the lesson. "
        f"Only the following are supported: {SUPPORTED_LIST}. "
        "Call only when the learner explicitly asks to switch, change, or practice one of these. "
        "Do not call for any other language — treat unrecognized requests as mishearings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "language_code": {
                "type": "string",
                "enum": sorted(SUPPORTED_LANGUAGES.keys()),
                "description": "ISO 639-1 code. Must be one of: es, fr, he, ru, ja, fa, ar, sv.",
            },
            "language_name": {
                "type": "string",
                "enum": sorted(SUPPORTED_LANGUAGES.values()),
                "description": "Human-readable English name. Must be one of: Spanish, French, Hebrew, Russian, Japanese, Farsi, Arabic, Swedish.",
            },
        },
        "required": ["language_code", "language_name"],
    },
}

REMEMBER_VOCAB_TOOL = {
    "name": "remember_vocabulary",
    "description": (
        "Record new vocabulary the tutor introduced this turn that the learner "
        "should retain. Stored under the current target language. Skip words "
        "used only in passing or already in prior-session memory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "description": "One or more new words/phrases just introduced.",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string", "description": "Word or phrase in the target language."},
                        "translation": {"type": "string", "description": "Brief English gloss."},
                        "note": {"type": "string", "description": "Optional usage note or example."},
                    },
                    "required": ["word", "translation"],
                },
            },
        },
        "required": ["words"],
    },
}

REMEMBER_PREFERENCE_TOOL = {
    "name": "remember_preference",
    "description": (
        "Record a teaching-style preference the learner explicitly stated this "
        "turn (e.g. 'shorter responses', 'don't translate every word', 'speak "
        "slower', 'always quiz me', 'stop doing X'). Require an explicit "
        "statement — never infer from neutral remarks."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "preference": {
                "type": "string",
                "description": "One short imperative sentence capturing the rule.",
            },
        },
        "required": ["preference"],
    },
}

REMEMBER_MISTAKE_TOOL = {
    "name": "remember_mistake",
    "description": (
        "Record a notable learner error along with the correction so it can be "
        "reinforced next session. Substantive grammar/usage issues only — skip "
        "minor slips."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "What the learner said or got wrong."},
            "correction": {"type": "string", "description": "The right form plus a brief why."},
        },
        "required": ["description", "correction"],
    },
}

CHAT_TOOLS = [SET_LANGUAGE_TOOL]
EXTRACTION_TOOLS = [
    REMEMBER_VOCAB_TOOL,
    REMEMBER_PREFERENCE_TOOL,
    REMEMBER_MISTAKE_TOOL,
]

EXTRACTOR_SYSTEM_PROMPT = """You analyze a single just-completed turn of a language-tutor conversation and extract memory updates by calling tools.

Call zero or more of these tools, each at most once per turn:
- remember_vocabulary: when the tutor introduced new words/phrases worth retaining. Skip words used only in passing.
- remember_preference: when the learner explicitly stated a teaching-style preference. Require an explicit statement — never infer.
- remember_mistake: when the tutor corrected a substantive learner error. Skip minor slips.

If nothing notable happened, call no tools. Do not produce any conversational text — your output is not shown to the learner."""


@dataclass
class Session:
    target_language_code: str = "es"
    target_language_name: str = "Spanish"
    messages: list = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    system_prompt: str = SYSTEM_PROMPT_BASE


def record_until_silence() -> np.ndarray:
    """Block on the mic, return int16 mono audio when the speaker stops."""
    block_seconds = 0.1
    block_size = int(SAMPLE_RATE * block_seconds)
    silence_target = int(SILENCE_DURATION / block_seconds)
    speech_min = int(MIN_SPEECH_DURATION / block_seconds)
    max_blocks = int(MAX_TURN_SECONDS / block_seconds)

    blocks: list[np.ndarray] = []
    speech_blocks = 0
    silence_blocks = 0

    print("\U0001f3a4  listening...", flush=True)
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", blocksize=block_size
    ) as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_size)
            blocks.append(block.copy())
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            if rms > SILENCE_RMS:
                speech_blocks += 1
                silence_blocks = 0
            elif speech_blocks >= speech_min:
                silence_blocks += 1
                if silence_blocks >= silence_target:
                    break

    return np.concatenate(blocks).flatten()


def _whisper_call(
    openai_client: OpenAI, path: str, language: str | None
) -> tuple[str, str]:
    kwargs: dict = {"model": WHISPER_MODEL, "response_format": "verbose_json"}
    if language:
        kwargs["language"] = language
    with open(path, "rb") as f:
        kwargs["file"] = f
        result = openai_client.audio.transcriptions.create(**kwargs)
    return (result.text or "").strip(), (result.language or "").strip()


def transcribe(
    openai_client: OpenAI, audio: np.ndarray, target_code: str
) -> tuple[str, str]:
    """Return (text, detected_language). Whisper auto-detects.

    Biases Whisper with the current target language unless the target is English.
    If the first-pass transcription contains an English navigation phrase
    ("switch", "practice", etc.), retry with language="en" so the command is
    captured cleanly regardless of the current target language.
    If Whisper reports a language outside our 9 allowed (8 supported + English),
    we treat it as a mishearing and force the detected language to target_code.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(audio.tobytes())

        first_hint = target_code if target_code and target_code != "en" else None
        text, detected = _whisper_call(openai_client, path, first_hint)

        if first_hint and looks_like_switch_request(text):
            en_text, en_detected = _whisper_call(openai_client, path, "en")
            if en_text:
                text, detected = en_text, en_detected

        if detected.lower() not in ALLOWED_DETECTED:
            detected = target_code
        return text, detected
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def speak(eleven_client: ElevenLabs, text: str, voice_id: str) -> None:
    audio_iter = eleven_client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=TTS_MODEL,
        text=text,
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75, style=0.0),
        output_format="pcm_24000",
    )
    pcm = b"".join(audio_iter)
    samples = np.frombuffer(pcm, dtype=np.int16)
    sd.play(samples, samplerate=24_000)
    sd.wait()


def chat(
    claude: anthropic.Anthropic, session: Session, user_text: str
) -> Iterator[str]:
    """Stream one user turn, yielding each completed sentence as it arrives.

    Handles set_target_language tool calls between yields. The caller reads
    `session.target_language_code` after each yield to pick the right TTS voice
    — that value reflects the language the just-yielded sentence was generated
    in, since tool-use processing only happens after a stream's final flush.
    """
    mode_note = f"[Mode: target language is {session.target_language_name}.] "
    session.messages.append({"role": "user", "content": mode_note + user_text})

    while True:
        buffer = ""
        with claude.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=session.system_prompt,
            tools=CHAT_TOOLS,
            messages=session.messages,
        ) as stream:
            for chunk in stream.text_stream:
                buffer += chunk
                while True:
                    sentence, buffer = _split_sentence(buffer)
                    if sentence is None:
                        break
                    yield sentence
            if buffer.strip():
                yield buffer.strip()
            final_message = stream.get_final_message()

        session.messages.append({"role": "assistant", "content": final_message.content})

        if final_message.stop_reason != "tool_use":
            return

        tool_results = []
        for block in final_message.content:
            if block.type != "tool_use":
                continue
            if block.name == "set_target_language":
                tool_results.append(_handle_set_language(session, block))
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": f"Unknown tool: {block.name}",
                })
        session.messages.append({"role": "user", "content": tool_results})


def _handle_set_language(session: Session, block) -> dict:
    code = (block.input.get("language_code") or "").lower()
    requested_name = block.input.get("language_name") or ""
    if code not in SUPPORTED_LANGUAGES:
        code = NAME_TO_CODE.get(requested_name.strip().lower(), "")
    if code in SUPPORTED_LANGUAGES:
        name = SUPPORTED_LANGUAGES[code]
        session.target_language_code = code
        session.target_language_name = name
        print(f"\U0001f310  switched to {name} ({code})", flush=True)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Target language is now {name}. Continue the conversation in {name}.",
        }
    print(
        f"⚠️  ignored switch to '{requested_name or code}' "
        f"(not supported); staying in {session.target_language_name}",
        flush=True,
    )
    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "is_error": True,
        "content": (
            f"'{requested_name or code}' is not a supported language. "
            f"Supported languages are: {SUPPORTED_LIST}. "
            f"This is likely a speech-to-text mishearing. "
            f"Stay in {session.target_language_name} and ask the learner to repeat."
        ),
    }


def extract_memory(
    claude: anthropic.Anthropic,
    session: Session,
    user_text: str,
    assistant_text: str,
) -> None:
    """Run a separate Haiku call to pull vocab/preference/mistake updates from
    the just-completed turn and persist them. Failures are non-fatal — memory
    is best-effort, not load-bearing for the conversation."""
    turn = (
        f"Target language: {session.target_language_name}\n"
        f"Learner: {user_text}\n"
        f"Tutor: {assistant_text}"
    )
    try:
        response = claude.messages.create(
            model=EXTRACTOR_MODEL,
            max_tokens=512,
            system=EXTRACTOR_SYSTEM_PROMPT,
            tools=EXTRACTION_TOOLS,
            messages=[{"role": "user", "content": turn}],
        )
    except Exception as e:  # noqa: BLE001 — boundary call, log and skip
        print(f"⚠️  memory extraction failed: {e}", file=sys.stderr)
        return

    code = session.target_language_code
    changed = False
    for block in response.content:
        if block.type != "tool_use":
            continue
        if block.name == "remember_vocabulary":
            slot = memory_slot(session.memory, code)
            added = []
            for w in (block.input.get("words") or []):
                word = (w.get("word") or "").strip()
                if not word:
                    continue
                entry = {"word": word, "translation": (w.get("translation") or "").strip()}
                note = (w.get("note") or "").strip()
                if note:
                    entry["note"] = note
                slot["vocab"].append(entry)
                added.append(word)
            if added:
                print(f"[memory] vocab [{code}]: {', '.join(added)}", flush=True)
                changed = True
        elif block.name == "remember_preference":
            pref = (block.input.get("preference") or "").strip()
            if pref:
                slot = memory_slot(session.memory, code)
                slot["preferences"].append(pref)
                print(f"[memory] preference [{code}]: {pref}", flush=True)
                changed = True
        elif block.name == "remember_mistake":
            desc = (block.input.get("description") or "").strip()
            corr = (block.input.get("correction") or "").strip()
            if desc and corr:
                slot = memory_slot(session.memory, code)
                slot["mistakes"].append({"description": desc, "correction": corr})
                print(f"[memory] correction [{code}]: {desc} → {corr}", flush=True)
                changed = True
    if changed:
        save_memory(session.memory)


def main() -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.environ.get(var):
            print(f"missing env var: {var}", file=sys.stderr)
            sys.exit(1)

    claude = anthropic.Anthropic()
    openai_client = OpenAI()
    eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    memory = load_memory()
    session = Session(memory=memory, system_prompt=build_system_prompt(memory))

    print("AI Language Professor. Ctrl-C to quit.", flush=True)
    if memory:
        langs = ", ".join(SUPPORTED_LANGUAGES.get(c, c) for c in sorted(memory.keys()))
        print(f"loaded memory for: {langs}", flush=True)
    print(f"\nStarting in {session.target_language_name}. "
          f"Supported: {SUPPORTED_LIST}. "
          "Say 'let's switch to <language>' anytime.\n", flush=True)

    try:
        while True:
            audio = record_until_silence()
            if len(audio) < int(SAMPLE_RATE * 0.3):
                print("(too short)\n", flush=True)
                continue

            text, detected = transcribe(
                openai_client, audio, session.target_language_code
            )
            if not text:
                print("(silence)\n", flush=True)
                continue
            print(f"you [{detected}]: {text}", flush=True)

            assistant_sentences: list[str] = []
            for sentence in chat(claude, session, text):
                lang = session.target_language_code
                print(f"tutor [{lang}]: {sentence}", flush=True)
                voice_id = VOICE_MAP.get(lang, DEFAULT_VOICE_ID)
                speak(eleven, sentence, voice_id)
                assistant_sentences.append(sentence)
            if not assistant_sentences:
                print("(no reply)", flush=True)
            else:
                extract_memory(claude, session, text, " ".join(assistant_sentences))
            print(flush=True)

    except KeyboardInterrupt:
        print("\ngoodbye.", flush=True)


if __name__ == "__main__":
    main()
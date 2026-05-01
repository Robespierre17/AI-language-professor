"""Voice conversation loop: Whisper STT -> Claude language tutor -> ElevenLabs TTS.

The tutor switches target language mid-session when the learner asks
("let's switch to Hebrew", "practice Japanese now"). Language state is held in a
Session object and exposed to Claude via a set_target_language tool.
"""

from __future__ import annotations

import os
import sys
import tempfile
import wave
from dataclasses import dataclass, field

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
SILENCE_DURATION = 1.4      # seconds of silence after speech that ends a turn
MIN_SPEECH_DURATION = 0.4   # ignore silence until this much speech is heard
MAX_TURN_SECONDS = 60

CLAUDE_MODEL = "claude-opus-4-7"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah

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


# Search terms used to score voices per language. Includes the English name,
# the endonym, and accent labels likely to appear in ElevenLabs voice metadata.
LANGUAGE_VOICE_ALIASES = {
    "es": ("spanish", "español", "castilian", "castellano"),
    "fr": ("french", "français", "francais", "parisian"),
    "he": ("hebrew", "ivrit", "israeli"),
    "ru": ("russian", "русский"),
    "ja": ("japanese", "日本", "nihongo"),
    "fa": ("persian", "farsi", "iranian"),
    "ar": ("arabic", "عربي", "egyptian", "levantine", "gulf"),
    "sv": ("swedish", "svenska", "svensk"),
}


def _voice_haystack(voice) -> str:
    labels = getattr(voice, "labels", None) or {}
    parts = [
        getattr(voice, "name", "") or "",
        getattr(voice, "description", "") or "",
        labels.get("language", "") or "",
        labels.get("accent", "") or "",
        labels.get("description", "") or "",
        labels.get("descriptive", "") or "",
        labels.get("use_case", "") or "",
    ]
    return " ".join(str(p) for p in parts).lower()


def build_voice_map(eleven_client: ElevenLabs) -> dict[str, str]:
    """Pick the best native-accent voice in the user's library per language.

    Returns {code: voice_id}. Languages with no clear match are omitted, and the
    caller falls back to DEFAULT_VOICE_ID (the multilingual generic).
    """
    try:
        voices = list(eleven_client.voices.get_all().voices)
    except Exception as e:  # noqa: BLE001 — boundary call, surface and fall back
        print(f"⚠️  could not list ElevenLabs voices: {e}", file=sys.stderr)
        return {}

    voice_map: dict[str, str] = {}
    for code, terms in LANGUAGE_VOICE_ALIASES.items():
        best = None
        best_score = 0
        for v in voices:
            if v.voice_id == DEFAULT_VOICE_ID:
                continue  # the generic multilingual fallback — never preferred
            haystack = _voice_haystack(v)
            score = sum(10 for term in terms if term in haystack)
            if score == 0:
                continue
            if "multilingual" in haystack:
                score -= 8
            category = getattr(v, "category", "") or ""
            if category == "premade":
                score += 2
            elif category == "professional":
                score += 1
            if score > best_score:
                best_score = score
                best = v
        if best and best_score >= 10:
            voice_map[code] = best.voice_id
            print(
                f"\U0001f50a  {SUPPORTED_LANGUAGES[code]}: {best.name} "
                f"({best.voice_id})",
                flush=True,
            )
        else:
            print(
                f"\U0001f50a  {SUPPORTED_LANGUAGES[code]}: no native voice in "
                f"library, falling back to default",
                flush=True,
            )
    return voice_map


VOICE_MAP: dict[str, str] = {}

SYSTEM_PROMPT = f"""You are an adaptive language tutor in a real-time spoken conversation with a learner.

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


@dataclass
class Session:
    target_language_code: str = "es"
    target_language_name: str = "Spanish"
    messages: list = field(default_factory=list)


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


def chat(claude: anthropic.Anthropic, session: Session, user_text: str) -> str:
    """Run one user turn. Handle set_target_language tool calls. Return spoken reply."""
    mode_note = f"[Mode: target language is {session.target_language_name}.] "
    session.messages.append({"role": "user", "content": mode_note + user_text})

    while True:
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SET_LANGUAGE_TOOL],
            messages=session.messages,
        )
        session.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text").strip()

        tool_results = []
        for block in response.content:
            if block.type != "tool_use" or block.name != "set_target_language":
                continue
            code = (block.input.get("language_code") or "").lower()
            requested_name = block.input.get("language_name") or ""
            if code not in SUPPORTED_LANGUAGES:
                code = NAME_TO_CODE.get(requested_name.strip().lower(), "")
            if code in SUPPORTED_LANGUAGES:
                name = SUPPORTED_LANGUAGES[code]
                session.target_language_code = code
                session.target_language_name = name
                print(f"\U0001f310  switched to {name} ({code})", flush=True)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Target language is now {name}. Continue the conversation in {name}.",
                })
            else:
                print(
                    f"⚠️  ignored switch to '{requested_name or code}' "
                    f"(not supported); staying in {session.target_language_name}",
                    flush=True,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "is_error": True,
                    "content": (
                        f"'{requested_name or code}' is not a supported language. "
                        f"Supported languages are: {SUPPORTED_LIST}. "
                        f"This is likely a speech-to-text mishearing. "
                        f"Stay in {session.target_language_name} and ask the learner to repeat."
                    ),
                })
        session.messages.append({"role": "user", "content": tool_results})


def main() -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY"):
        if not os.environ.get(var):
            print(f"missing env var: {var}", file=sys.stderr)
            sys.exit(1)

    claude = anthropic.Anthropic()
    openai_client = OpenAI()
    eleven = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    session = Session()

    print("AI Language Professor. Ctrl-C to quit.", flush=True)
    print("Discovering ElevenLabs voices...", flush=True)
    VOICE_MAP.update(build_voice_map(eleven))
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

            reply = chat(claude, session, text)
            if not reply:
                print("(no reply)\n", flush=True)
                continue
            voice_id = VOICE_MAP.get(session.target_language_code, DEFAULT_VOICE_ID)
            print(f"tutor [{session.target_language_code}]: {reply}\n", flush=True)
            speak(eleven, reply, voice_id)

    except KeyboardInterrupt:
        print("\ngoodbye.", flush=True)


if __name__ == "__main__":
    main()

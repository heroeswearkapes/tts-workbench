"""Declarative catalog for the voice assets published with Kokoro 82M."""

from tts_workbench.domain import VoiceDescriptor


def _voice(
    voice_id: str,
    name: str,
    language: str,
    gender: str,
    category: str,
    *,
    accent: str | None = None,
) -> VoiceDescriptor:
    return VoiceDescriptor(
        id=voice_id,
        name=name,
        language=language,
        accent=accent,
        gender=gender,
        category=category,
    )


KOKORO_VOICES: tuple[VoiceDescriptor, ...] = (
    # American English
    _voice("af_heart", "Heart", "en-US", "female", "American English", accent="American"),
    _voice("af_alloy", "Alloy", "en-US", "female", "American English", accent="American"),
    _voice("af_aoede", "Aoede", "en-US", "female", "American English", accent="American"),
    _voice("af_bella", "Bella", "en-US", "female", "American English", accent="American"),
    _voice("af_jessica", "Jessica", "en-US", "female", "American English", accent="American"),
    _voice("af_kore", "Kore", "en-US", "female", "American English", accent="American"),
    _voice("af_nicole", "Nicole", "en-US", "female", "American English", accent="American"),
    _voice("af_nova", "Nova", "en-US", "female", "American English", accent="American"),
    _voice("af_river", "River", "en-US", "female", "American English", accent="American"),
    _voice("af_sarah", "Sarah", "en-US", "female", "American English", accent="American"),
    _voice("af_sky", "Sky", "en-US", "female", "American English", accent="American"),
    _voice("am_adam", "Adam", "en-US", "male", "American English", accent="American"),
    _voice("am_echo", "Echo", "en-US", "male", "American English", accent="American"),
    _voice("am_eric", "Eric", "en-US", "male", "American English", accent="American"),
    _voice("am_fenrir", "Fenrir", "en-US", "male", "American English", accent="American"),
    _voice("am_liam", "Liam", "en-US", "male", "American English", accent="American"),
    _voice("am_michael", "Michael", "en-US", "male", "American English", accent="American"),
    _voice("am_onyx", "Onyx", "en-US", "male", "American English", accent="American"),
    _voice("am_puck", "Puck", "en-US", "male", "American English", accent="American"),
    _voice("am_santa", "Santa (American)", "en-US", "male", "American English", accent="American"),
    # British English
    _voice("bf_alice", "Alice", "en-GB", "female", "British English", accent="British"),
    _voice("bf_emma", "Emma", "en-GB", "female", "British English", accent="British"),
    _voice("bf_isabella", "Isabella", "en-GB", "female", "British English", accent="British"),
    _voice("bf_lily", "Lily", "en-GB", "female", "British English", accent="British"),
    _voice("bm_daniel", "Daniel", "en-GB", "male", "British English", accent="British"),
    _voice("bm_fable", "Fable", "en-GB", "male", "British English", accent="British"),
    _voice("bm_george", "George", "en-GB", "male", "British English", accent="British"),
    _voice("bm_lewis", "Lewis", "en-GB", "male", "British English", accent="British"),
    # Japanese
    _voice("jf_alpha", "Alpha (Japanese)", "ja-JP", "female", "Japanese"),
    _voice("jf_gongitsune", "Gongitsune", "ja-JP", "female", "Japanese"),
    _voice("jf_nezumi", "Nezumi", "ja-JP", "female", "Japanese"),
    _voice("jf_tebukuro", "Tebukuro", "ja-JP", "female", "Japanese"),
    _voice("jm_kumo", "Kumo", "ja-JP", "male", "Japanese"),
    # Mandarin Chinese
    _voice("zf_xiaobei", "Xiaobei", "zh-CN", "female", "Mandarin Chinese"),
    _voice("zf_xiaoni", "Xiaoni", "zh-CN", "female", "Mandarin Chinese"),
    _voice("zf_xiaoxiao", "Xiaoxiao", "zh-CN", "female", "Mandarin Chinese"),
    _voice("zf_xiaoyi", "Xiaoyi", "zh-CN", "female", "Mandarin Chinese"),
    _voice("zm_yunjian", "Yunjian", "zh-CN", "male", "Mandarin Chinese"),
    _voice("zm_yunxi", "Yunxi", "zh-CN", "male", "Mandarin Chinese"),
    _voice("zm_yunxia", "Yunxia", "zh-CN", "male", "Mandarin Chinese"),
    _voice("zm_yunyang", "Yunyang", "zh-CN", "male", "Mandarin Chinese"),
    # Spanish
    _voice("ef_dora", "Dora (Spanish)", "es", "female", "Spanish"),
    _voice("em_alex", "Alex (Spanish)", "es", "male", "Spanish"),
    _voice("em_santa", "Santa (Spanish)", "es", "male", "Spanish"),
    # French
    _voice("ff_siwis", "Siwis", "fr-FR", "female", "French"),
    # Hindi
    _voice("hf_alpha", "Alpha (Hindi)", "hi-IN", "female", "Hindi"),
    _voice("hf_beta", "Beta", "hi-IN", "female", "Hindi"),
    _voice("hm_omega", "Omega", "hi-IN", "male", "Hindi"),
    _voice("hm_psi", "Psi", "hi-IN", "male", "Hindi"),
    # Italian
    _voice("if_sara", "Sara", "it-IT", "female", "Italian"),
    _voice("im_nicola", "Nicola", "it-IT", "male", "Italian"),
    # Brazilian Portuguese
    _voice(
        "pf_dora",
        "Dora (Brazilian Portuguese)",
        "pt-BR",
        "female",
        "Brazilian Portuguese",
        accent="Brazilian",
    ),
    _voice(
        "pm_alex",
        "Alex (Brazilian Portuguese)",
        "pt-BR",
        "male",
        "Brazilian Portuguese",
        accent="Brazilian",
    ),
    _voice(
        "pm_santa",
        "Santa (Brazilian Portuguese)",
        "pt-BR",
        "male",
        "Brazilian Portuguese",
        accent="Brazilian",
    ),
)

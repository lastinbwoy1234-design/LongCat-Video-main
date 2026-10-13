#!/usr/bin/env python3
"""video_scaffold.py - topic in, scene-structured script + production folders out.

Content types (channel profiles):
  car       specs, reviews, cinematic car clips
  food      quick recipe breakdowns, step-by-step cooking, food facts
  timelapse progressive build / growth time-lapses (e.g. bunker build)

Usage:
  python video_scaffold.py "2025 Porsche 911 GT3" --type car --minutes 5
  python video_scaffold.py "Garlic butter shrimp" --type food --minutes 3
  python video_scaffold.py "Underground bunker build" --type timelapse --minutes 8
  python video_scaffold.py "Test topic" --type car --dry-run      # no API call, placeholder script
  python video_scaffold.py --status output/2025-porsche-911-gt3   # which scenes still need assets
  python video_scaffold.py --list-types

Needs ANTHROPIC_API_KEY (and `pip install anthropic`) unless --dry-run is used.

To add a new content type, add one ChannelProfile to PROFILES. Nothing else changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"
WORDS_PER_SECOND = 150 / 60  # narration pace used for duration estimates
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

# Fields every scene has, whatever the channel type.
COMMON_SCENE_FIELDS = {
    "narration": "exact words the narrator speaks (plain text, no stage directions)",
    "visual_prompt": "one detailed shot description for a text-to-video model or stock-footage search: "
    "subject, action, setting, lighting, camera movement. No on-screen text, no real named people.",
}
# Fields every video has.
COMMON_VIDEO_FIELDS = {
    "title": "YouTube title, under 90 characters",
    "description": "YouTube description, 2-4 sentences",
    "tags": "list of 8-12 tags",
    "fact_check": "list of factual claims in the script (figures, dates, specs) a human should verify before publishing",
}


@dataclass(frozen=True)
class ChannelProfile:
    key: str
    label: str
    brief: str  # writing guidance given to the model
    video_fields: dict  # extra top-level fields: name -> description
    scene_fields: dict  # extra per-scene fields: name -> description
    seconds_per_scene: tuple  # (min, max) target narration length per scene
    placeholder_scenes: tuple  # scene labels used by --dry-run


PROFILES: dict[str, ChannelProfile] = {
    "car": ChannelProfile(
        key="car",
        label="Car channel",
        brief=(
            "Cinematic car video: open with a hook, then walk through design, performance, interior, "
            "driving impressions and a verdict. Highlight real specs and honest pros/cons. "
            "Visual prompts should read like cinematic car footage (rolling shots, detail close-ups, "
            "golden-hour or night city lighting). Never invent specs; if unsure, leave the figure out "
            "and list it in fact_check."
        ),
        video_fields={
            "specs": "object of spec label -> value (engine, power, 0-60, price...). Only values you are confident in.",
            "verdict": "one-sentence overall verdict",
        },
        scene_fields={
            "shot_type": "e.g. tracking shot, drone, interior detail, rolling shot, close-up",
            "on_screen_text": "short spec callout shown on screen, or empty string",
        },
        seconds_per_scene=(12, 25),
        placeholder_scenes=("hook", "design", "performance", "interior", "driving", "verdict"),
    ),
    "food": ChannelProfile(
        key="food",
        label="Food channel",
        brief=(
            "Quick recipe breakdown: hook, ingredients, clear numbered cooking steps, a short food fact "
            "(origin, science or tip), and a plating/serving outro. Keep steps short and actionable. "
            "Visual prompts should be overhead or close-up kitchen shots with appetizing lighting. "
            "Use scene_type 'fact' for food-fact scenes. Never invent nutrition numbers."
        ),
        video_fields={
            "servings": "number of servings",
            "total_time_minutes": "total prep + cook time in minutes",
            "ingredients": "list of {item, amount}",
        },
        scene_fields={
            "scene_type": "one of: intro, ingredients, step, fact, outro",
            "step_number": "integer for step scenes, otherwise null",
            "on_screen_text": "short caption (e.g. '2 min, medium heat') or empty string",
            "timer_seconds": "cook/rest time in seconds if the step has one, otherwise null",
        },
        seconds_per_scene=(8, 20),
        placeholder_scenes=("intro", "ingredients", "step 1", "step 2", "step 3", "fact", "outro"),
    ),
    "timelapse": ChannelProfile(
        key="timelapse",
        label="AI time-lapse / bunker build",
        brief=(
            "Progressive-build time-lapse: the SAME locked camera shows a subject going from empty site "
            "to finished result across stages. Define fixed_camera once, and begin EVERY visual_prompt "
            "with that camera description, keeping location, lighting direction and framing identical so "
            "scenes cut together as one time-lapse. Each scene shows a later stage and describes only what "
            "has changed since the previous one. Narration is short and atmospheric."
        ),
        video_fields={
            "build_subject": "what is being built or grown",
            "fixed_camera": "one sentence describing the locked camera position, lens and lighting used in every scene",
            "total_stages": "number of stages (equals number of scenes)",
        },
        scene_fields={
            "stage_label": "short name of this stage, e.g. 'Excavation'",
            "progress_percent": "integer 0-100, strictly increasing across scenes, last scene is 100",
            "time_label": "elapsed time shown on screen, e.g. 'Day 3'",
            "change_from_previous": "what is new compared with the previous scene",
        },
        seconds_per_scene=(10, 20),
        placeholder_scenes=("empty site", "groundwork", "structure", "interior", "finishing", "final reveal"),
    ),
}


# --------------------------------------------------------------------------- prompt + LLM


def build_prompt(profile: ChannelProfile, topic: str, minutes: float) -> str:
    lo, hi = profile.seconds_per_scene
    words = int(minutes * 60 * WORDS_PER_SECOND)
    n_lo, n_hi = max(3, int(minutes * 60 / hi)), max(4, int(minutes * 60 / lo))

    video_schema = {**COMMON_VIDEO_FIELDS, **profile.video_fields}
    scene_schema = {**COMMON_SCENE_FIELDS, **profile.scene_fields}
    schema = {**video_schema, "scenes": [scene_schema]}

    return (
        f"You are a scriptwriter for a YouTube channel. Channel type: {profile.label}.\n\n"
        f"Topic: {topic}\n"
        f"Target length: {minutes:g} minutes (about {words} words of narration at 150 words per minute).\n"
        f"Use {n_lo}-{n_hi} scenes of roughly {lo}-{hi} seconds of narration each.\n\n"
        f"Channel guidance: {profile.brief}\n\n"
        "Return ONLY valid JSON (no markdown fences) in this shape, where each value below is a "
        "description of what to put there:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model reply contained no JSON object")
    return json.loads(text[start : end + 1])


def validate(script: dict, profile: ChannelProfile) -> None:
    for key in list(COMMON_VIDEO_FIELDS) + list(profile.video_fields) + ["scenes"]:
        if key not in script:
            raise ValueError(f"script missing top-level field '{key}'")
    if not script["scenes"]:
        raise ValueError("script has no scenes")
    required = list(COMMON_SCENE_FIELDS) + list(profile.scene_fields)
    for i, scene in enumerate(script["scenes"], 1):
        for key in required:
            if key not in scene:
                raise ValueError(f"scene {i} missing field '{key}'")
        if not str(scene["narration"]).strip() or not str(scene["visual_prompt"]).strip():
            raise ValueError(f"scene {i} has empty narration or visual_prompt")


def generate_script(profile: ChannelProfile, topic: str, minutes: float, model: str) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Export it, or use --dry-run to test the folder framework.")
    try:
        import anthropic
    except ImportError:
        raise SystemExit("Missing dependency: pip install anthropic")

    client = anthropic.Anthropic()
    prompt = build_prompt(profile, topic, minutes)
    last_error = None
    for attempt in range(2):
        content = prompt if attempt == 0 else (
            prompt + f"\n\nYour previous reply was invalid ({last_error}). Return only the corrected JSON."
        )
        msg = client.messages.create(
            model=model, max_tokens=16000, messages=[{"role": "user", "content": content}]
        )
        try:
            script = _extract_json(msg.content[0].text)
            validate(script, profile)
            return script
        except (ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
    raise SystemExit(f"Model did not return a valid script after 2 attempts: {last_error}")


def dry_run_script(profile: ChannelProfile, topic: str, minutes: float) -> dict:
    """Placeholder script so the folder framework can be tested without an API call."""
    labels = list(profile.placeholder_scenes)
    script: dict = {
        "title": f"[DRY RUN] {topic}",
        "description": f"Placeholder description for {topic}.",
        "tags": [profile.key, topic],
        "fact_check": [],
        "scenes": [],
    }
    for k, desc in profile.video_fields.items():
        script[k] = [] if "list" in desc else ({} if "object" in desc else "")
    if profile.key == "timelapse":
        script["total_stages"] = len(labels)
    for i, label in enumerate(labels, 1):
        scene = {
            "narration": f"Placeholder narration for {label} of {topic}. " * 3,
            "visual_prompt": f"Placeholder shot for {label}.",
        }
        for k in profile.scene_fields:
            scene[k] = ""
        if profile.key == "timelapse":
            scene["stage_label"] = label
            scene["progress_percent"] = round(100 * i / len(labels))
        if profile.key == "food":
            scene["scene_type"] = "step" if label.startswith("step") else label
            scene["step_number"] = None
            scene["timer_seconds"] = None
        script["scenes"].append(scene)
    return script


# --------------------------------------------------------------------------- folder framework


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "untitled"


def scene_id(i: int) -> str:
    return f"scene_{i:03d}"


def estimate_seconds(narration: str) -> float:
    return round(len(narration.split()) / WORDS_PER_SECOND, 1)


def build_framework(script: dict, profile: ChannelProfile, topic: str, out_root: Path) -> Path:
    root = out_root / slugify(topic)
    for sub in ("00_shared/music", "00_shared/sfx", "00_shared/thumbnails", "00_shared/exports"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    manifest_scenes = []
    md = [f"# {script['title']}", "", f"*{profile.label} · topic: {topic}*", "", script["description"], ""]

    for i, scene in enumerate(script["scenes"], 1):
        sid = scene_id(i)
        sdir = root / "scenes" / sid
        (sdir / "voiceover").mkdir(parents=True, exist_ok=True)
        (sdir / "footage").mkdir(parents=True, exist_ok=True)
        (sdir / "narration.txt").write_text(scene["narration"].strip() + "\n", encoding="utf-8")
        (sdir / "visual_prompt.txt").write_text(scene["visual_prompt"].strip() + "\n", encoding="utf-8")
        (sdir / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")

        est = estimate_seconds(scene["narration"])
        manifest_scenes.append(
            {
                "id": sid,
                "est_seconds": est,
                "voiceover_drop": f"scenes/{sid}/voiceover/",
                "footage_drop": f"scenes/{sid}/footage/",
                "suggested_voiceover_name": f"{sid}.mp3",
                "suggested_footage_name": f"{sid}.mp4",
            }
        )
        md += [f"## {i}. {sid} (~{est}s)", "", f"**Narration:** {scene['narration']}", "",
               f"**Visual:** {scene['visual_prompt']}", ""]
        extras = {k: scene[k] for k in profile.scene_fields if scene.get(k) not in ("", None)}
        if extras:
            md += [f"**Details:** {json.dumps(extras, ensure_ascii=False)}", ""]

    if script.get("fact_check"):
        md += ["## Fact-check before publishing", ""] + [f"- {c}" for c in script["fact_check"]] + [""]

    (root / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "script.md").write_text("\n".join(md), encoding="utf-8")
    manifest = {
        "topic": topic,
        "type": profile.key,
        "title": script["title"],
        "total_est_seconds": round(sum(s["est_seconds"] for s in manifest_scenes), 1),
        "scenes": manifest_scenes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "README.txt").write_text(
        "Drop one voiceover and one footage file in each scene folder:\n"
        "  scenes/scene_001/voiceover/  <- narration audio (mp3/wav/m4a)\n"
        "  scenes/scene_001/footage/    <- background footage (mp4/mov/webm)\n"
        "Read narration.txt / visual_prompt.txt in each scene folder for what to record or generate.\n"
        "00_shared/ holds music, sfx, thumbnails and finished exports.\n"
        "Check progress with: python video_scaffold.py --status <this folder>\n",
        encoding="utf-8",
    )
    return root


def show_status(root: Path) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found. Pass a folder created by this script.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_vo, missing_fo = [], []
    print(f"{manifest['title']}  ({manifest['type']}, ~{manifest['total_est_seconds']}s)")
    for s in manifest["scenes"]:
        has_vo = any(p.suffix.lower() in AUDIO_EXTS for p in (root / s["voiceover_drop"]).glob("*"))
        has_fo = any(p.suffix.lower() in VIDEO_EXTS for p in (root / s["footage_drop"]).glob("*"))
        print(f"  {s['id']}  voiceover {'OK     ' if has_vo else 'MISSING'}  footage {'OK' if has_fo else 'MISSING'}")
        if not has_vo:
            missing_vo.append(s["id"])
        if not has_fo:
            missing_fo.append(s["id"])
    total = len(manifest["scenes"])
    print(f"Voiceovers: {total - len(missing_vo)}/{total}   Footage: {total - len(missing_fo)}/{total}")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description="Topic -> scene script + production folder framework")
    p.add_argument("topic", nargs="?", help="video topic, e.g. 'Garlic butter shrimp'")
    p.add_argument("--type", choices=list(PROFILES), help="content type")
    p.add_argument("--minutes", type=float, default=5.0, help="target video length (default 5)")
    p.add_argument("--out", default="output", help="output root folder (default ./output)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dry-run", action="store_true", help="skip the AI call; write a placeholder script")
    p.add_argument("--status", metavar="FOLDER", help="report which scenes still need voiceover/footage")
    p.add_argument("--list-types", action="store_true")
    args = p.parse_args()

    if args.list_types:
        for prof in PROFILES.values():
            print(f"{prof.key:10s} {prof.label}")
        return
    if args.status:
        show_status(Path(args.status))
        return
    if not args.topic or not args.type:
        p.error("topic and --type are required (or use --status / --list-types)")

    profile = PROFILES[args.type]
    if args.dry_run:
        script = dry_run_script(profile, args.topic, args.minutes)
    else:
        print(f"Generating {profile.label} script for '{args.topic}' ...")
        script = generate_script(profile, args.topic, args.minutes, args.model)

    root = build_framework(script, profile, args.topic, Path(args.out))
    total = sum(estimate_seconds(s["narration"]) for s in script["scenes"])
    print(f"Created {root}  ({len(script['scenes'])} scenes, ~{total / 60:.1f} min of narration)")
    print("Next: add voiceovers and footage per scene, then run with --status to check progress.")


if __name__ == "__main__":
    main()

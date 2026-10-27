"""
action_map.py

Maps recognized sign classes -> urgency, output modalities, and a clear
driver instruction (spoken + on-screen).

Supports:
  - signs_general 52-class folder / human names (this project's VGG16)
  - legacy short keys used by the YOLO scaffold (stop, no_entry, ...)

Driver profile:
  - hearing: prefer AUDIO (+ screen for high urgency)
  - deaf: SCREEN flash only (no audio dependency)
  - both: screen + audio (+ vibration on high urgency)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Urgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Modality(Enum):
    SCREEN = "screen_flash"
    VIBRATION = "vibration"
    AUDIO = "audio"


@dataclass
class AlertDecision:
    sign_class: str
    sign_name: str
    action: str
    urgency: Urgency
    modalities: list[Modality]
    message: str


# Human-readable action for each signs_general class name (from classes.txt).
SIGN_ACTIONS: dict[str, tuple[str, Urgency]] = {
    "Speed limit (5km/h)": ("Slow down to 5 km/h or below.", Urgency.MEDIUM),
    "Speed limit (15km/h)": ("Slow down to 15 km/h or below.", Urgency.MEDIUM),
    "Speed limit (30km/h)": ("Slow down to 30 km/h or below.", Urgency.MEDIUM),
    "Speed limit (40km/h)": ("Keep speed at or below 40 km/h.", Urgency.LOW),
    "Speed limit (50km/h)": ("Keep speed at or below 50 km/h.", Urgency.LOW),
    "Speed limit (60km/h)": ("Keep speed at or below 60 km/h.", Urgency.LOW),
    "Speed limit (70km/h)": ("Keep speed at or below 70 km/h.", Urgency.LOW),
    "speed limit (80km/h)": ("Keep speed at or below 80 km/h.", Urgency.LOW),
    "Dont Go straight or left": ("Do not go straight or left. Follow the allowed direction.", Urgency.HIGH),
    "Unknown7": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Dont Go straight": ("Do not go straight. Choose an allowed turn.", Urgency.HIGH),
    "Dont Go Left": ("Do not turn left.", Urgency.HIGH),
    "Dont Go Left or Right": ("Do not turn left or right. Go straight only if allowed.", Urgency.HIGH),
    "Dont Go Right": ("Do not turn right.", Urgency.HIGH),
    "Dont overtake from Left": ("Do not overtake from the left.", Urgency.MEDIUM),
    "No Uturn": ("Do not make a U-turn.", Urgency.HIGH),
    "No Car": ("Cars not allowed. Do not enter with a car.", Urgency.HIGH),
    "No horn": ("Do not use the horn.", Urgency.MEDIUM),
    "No entry": ("No entry. Do not proceed into this road.", Urgency.HIGH),
    "No stopping": ("No stopping. Keep moving; do not stop here.", Urgency.MEDIUM),
    "Go straight or right": ("Go straight or turn right.", Urgency.MEDIUM),
    "Go straight": ("Go straight ahead.", Urgency.MEDIUM),
    "Go Left": ("Turn left.", Urgency.MEDIUM),
    "Go Left or right": ("Turn left or right.", Urgency.MEDIUM),
    "Go Right": ("Turn right.", Urgency.MEDIUM),
    "keep Left": ("Keep left.", Urgency.MEDIUM),
    "keep Right": ("Keep right.", Urgency.MEDIUM),
    "Roundabout mandatory": ("Enter the roundabout and follow the circle.", Urgency.MEDIUM),
    "watch out for cars": ("Watch out for cars. Slow down and scan surroundings.", Urgency.HIGH),
    "Horn": ("Horn allowed / horn zone. Use only if needed.", Urgency.LOW),
    "Bicycles crossing": ("Bicycles may cross. Slow down and yield if needed.", Urgency.HIGH),
    "Uturn": ("U-turn ahead is allowed. Check traffic before turning.", Urgency.MEDIUM),
    "Road Divider": ("Road divider ahead. Stay in your lane.", Urgency.MEDIUM),
    "Unknown6": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Danger Ahead": ("Danger ahead. Slow down immediately.", Urgency.HIGH),
    "Zebra Crossing": ("Zebra crossing ahead. Slow down and be ready to stop for pedestrians.", Urgency.HIGH),
    "Children crossing": ("Children crossing. Slow down and watch both sides.", Urgency.HIGH),
    "Dangerous curve to the left": ("Dangerous left curve. Slow down and stay in lane.", Urgency.HIGH),
    "Dangerous curve to the right": ("Dangerous right curve. Slow down and stay in lane.", Urgency.HIGH),
    "Unknown1": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Unknown2": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Unknown3": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Go right or straight": ("Go right or straight.", Urgency.MEDIUM),
    "Go left or straight": ("Go left or straight.", Urgency.MEDIUM),
    "Unknown4": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "ZigZag Curve": ("Zig-zag curve ahead. Reduce speed.", Urgency.HIGH),
    "Train Crossing": ("Train crossing ahead. Slow down and be ready to stop.", Urgency.HIGH),
    "Under Construction": ("Road under construction. Slow down and follow detours.", Urgency.HIGH),
    "Unknown5": ("Unfamiliar sign ahead. Proceed with caution.", Urgency.MEDIUM),
    "Fences": ("Fence / barrier area. Drive carefully.", Urgency.MEDIUM),
    "Heavy Vehicle Accidents": ("Heavy-vehicle hazard area. Increase following distance.", Urgency.HIGH),
}

# Legacy keys used by older YOLO scaffold demos
LEGACY_ACTIONS: dict[str, tuple[str, Urgency]] = {
    "stop": ("Stop completely before the line.", Urgency.HIGH),
    "yield": ("Yield to traffic / pedestrians.", Urgency.MEDIUM),
    "give_way": ("Give way to other traffic.", Urgency.MEDIUM),
    "pedestrian_crossing": ("Pedestrian crossing. Be ready to stop.", Urgency.HIGH),
    "school_zone": ("School zone. Slow down.", Urgency.HIGH),
    "no_entry": ("No entry. Do not proceed.", Urgency.HIGH),
    "speed_limit_30": ("Keep speed at or below 30 km/h.", Urgency.MEDIUM),
    "speed_limit_50": ("Keep speed at or below 50 km/h.", Urgency.LOW),
    "speed_limit_60": ("Keep speed at or below 60 km/h.", Urgency.LOW),
    "speed_limit_80": ("Keep speed at or below 80 km/h.", Urgency.LOW),
    "no_overtaking": ("Do not overtake.", Urgency.MEDIUM),
    "traffic_light_ahead": ("Traffic light ahead. Prepare to stop.", Urgency.MEDIUM),
    "one_way": ("One-way road. Follow the allowed direction.", Urgency.MEDIUM),
    "no_parking": ("No parking here.", Urgency.LOW),
    "roundabout": ("Roundabout ahead. Yield and circulate.", Urgency.MEDIUM),
}

_URGENCY_ORDER = {Urgency.LOW: 0, Urgency.MEDIUM: 1, Urgency.HIGH: 2}


def estimate_time_to_action(bbox_height_px: float, frame_height_px: int) -> float:
    relative_size = bbox_height_px / max(frame_height_px, 1)
    seconds = max(0.5, 6.0 * (1 - relative_size) ** 2)
    return round(seconds, 2)


def _lookup(sign_class: str) -> tuple[str, str, Urgency]:
    """Return (canonical_name, action, base_urgency)."""
    if sign_class in SIGN_ACTIONS:
        action, urg = SIGN_ACTIONS[sign_class]
        return sign_class, action, urg
    if sign_class in LEGACY_ACTIONS:
        action, urg = LEGACY_ACTIONS[sign_class]
        return sign_class.replace("_", " ").title(), action, urg
    # Folder-style names: 18_No_entry -> try fuzzy match on human names
    cleaned = sign_class
    if "_" in sign_class and sign_class[:2].isdigit():
        cleaned = sign_class.split("_", 1)[1].replace("_", " ")
    for name, (action, urg) in SIGN_ACTIONS.items():
        if name.lower() == cleaned.lower() or cleaned.lower() in name.lower():
            return name, action, urg
    return sign_class, "Unfamiliar sign. Proceed with caution.", Urgency.MEDIUM


def modalities_for(urgency: Urgency, driver_profile: str) -> list[Modality]:
    profile = (driver_profile or "both").lower()
    if profile == "deaf":
        # Visual-first: always flash; vibrate on medium/high
        if urgency == Urgency.HIGH:
            return [Modality.SCREEN, Modality.VIBRATION]
        return [Modality.SCREEN]
    if profile == "hearing":
        if urgency == Urgency.HIGH:
            return [Modality.AUDIO, Modality.SCREEN]
        if urgency == Urgency.MEDIUM:
            return [Modality.AUDIO, Modality.SCREEN]
        return [Modality.AUDIO]
    # both / default
    if urgency == Urgency.HIGH:
        return [Modality.SCREEN, Modality.VIBRATION, Modality.AUDIO]
    if urgency == Urgency.MEDIUM:
        return [Modality.SCREEN, Modality.AUDIO]
    return [Modality.SCREEN]


def decide_action(
    sign_class: str,
    seconds_to_action: float = 3.0,
    driver_profile: str = "both",
) -> AlertDecision:
    sign_name, action, base_urgency = _lookup(sign_class)

    if seconds_to_action < 1.5:
        urgency = Urgency.HIGH
    elif seconds_to_action < 3.0:
        urgency = max(base_urgency, Urgency.MEDIUM, key=_URGENCY_ORDER.get)
    else:
        urgency = base_urgency

    modalities = modalities_for(urgency, driver_profile)
    message = f"Sign: {sign_name}. Action: {action}"
    return AlertDecision(
        sign_class=sign_class,
        sign_name=sign_name,
        action=action,
        urgency=urgency,
        modalities=modalities,
        message=message,
    )


if __name__ == "__main__":
    for sign in ["No entry", "Speed limit (70km/h)", "Zebra Crossing", "stop"]:
        for profile in ["hearing", "deaf", "both"]:
            d = decide_action(sign, seconds_to_action=1.2, driver_profile=profile)
            print(f"{profile:8s} | {d.sign_name:28s} | {d.urgency.value:6s} | "
                  f"{[m.value for m in d.modalities]} | {d.action}")

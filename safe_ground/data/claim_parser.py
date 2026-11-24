"""Robust claim parser with strict JSON parsing and heuristic regex fallback."""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from safe_ground.data.schema import Claim
from safe_ground.utils.constants import (
    CLASS_NAME_TO_ID,
    BEARING_SECTORS,
    DISTANCE_BINS,
)


class ClaimParser:
    """Extracts structured claims from VLM text outputs with multiple fallback strategies."""

    def __init__(self):
        self.json_codeblock_pattern = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)
        self.tag_pattern = re.compile(r"<claim\s+([^>]+)/>", re.IGNORECASE)
        self.attr_pattern = re.compile(r'(\w+)=["\']([^"\']+)["\']')

        # Known object keywords for heuristic fallback
        self.known_classes = list(CLASS_NAME_TO_ID.keys())

    def parse(self, text: str) -> List[Claim]:
        """Parses claims from text, trying JSON, XML tags, and heuristic regex in order."""
        # 1. Try JSON block extraction
        claims = self._try_parse_json(text)
        if claims:
            return claims

        # 2. Try XML/HTML-like tags: <claim class="pedestrian" sector="front_left" dist="near"/>
        claims = self._try_parse_tags(text)
        if claims:
            return claims

        # 3. Fallback to heuristic rule-based extraction
        return self._heuristic_fallback(text)

    def _try_parse_json(self, text: str) -> List[Claim]:
        """Attempts to extract and parse JSON array of claim objects."""
        # Check inside codeblocks first
        matches = self.json_codeblock_pattern.findall(text)
        candidates = matches if matches else []

        # Also search for standalone JSON array [...]
        if not candidates:
            start_idx = text.find("[")
            end_idx = text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidates.append(text[start_idx : end_idx + 1])

        for cand in candidates:
            try:
                # Clean up any trailing commas that break standard json.loads
                cleaned = re.sub(r",\s*([\]\}])", r"\1", cand)
                data = json.loads(cleaned)
                if isinstance(data, dict) and "claims" in data:
                    data = data["claims"]
                if isinstance(data, list):
                    claims = []
                    for item in data:
                        if isinstance(item, dict):
                            claims.append(self._standardize_claim_dict(item))
                    if claims:
                        return claims
            except Exception:
                continue
        return []

    def _try_parse_tags(self, text: str) -> List[Claim]:
        """Extracts claims formatted as <claim class="..." bearing="..." dist="..."/>."""
        claims = []
        for match in self.tag_pattern.finditer(text):
            attr_str = match.group(1)
            attrs = dict(self.attr_pattern.findall(attr_str))
            if "class" in attrs or "object_class" in attrs:
                obj_cls = attrs.get("class", attrs.get("object_class", "other"))
                bearing = attrs.get("bearing", attrs.get("bearing_sector", "front"))
                dist = attrs.get("dist", attrs.get("distance_bin", "medium"))
                attr = attrs.get("attr", attrs.get("attribute", "normal"))
                cam = attrs.get("cam", attrs.get("camera_view", "CAM_FRONT"))
                claims.append(
                    Claim(
                        object_class=obj_cls.lower(),
                        bearing_sector=bearing.lower(),
                        distance_bin=dist.lower(),
                        attribute=attr.lower(),
                        camera_view=cam,
                    )
                )
        return claims

    def _heuristic_fallback(self, text: str) -> List[Claim]:
        """Last-resort extractor for unstructured sentences."""
        claims = []
        sentences = re.split(r"[.!?\n]+", text)

        for sent in sentences:
            sent_lower = sent.lower().strip()
            if not sent_lower:
                continue

            found_class = None
            for cls_name in self.known_classes:
                # word boundary match
                if re.search(r"\b" + re.escape(cls_name) + r"s?\b", sent_lower):
                    found_class = cls_name
                    break

            if not found_class:
                continue

            # Detect bearing sector (check multi-word sectors first)
            bearing = "front"
            sorted_sectors = sorted(BEARING_SECTORS, key=lambda s: len(s), reverse=True)
            for sector in sorted_sectors:
                sector_words = sector.replace("_", " ")
                if sector_words in sent_lower or sector in sent_lower:
                    bearing = sector
                    break

            # Detect distance bin
            dist_bin = "medium"
            if "close" in sent_lower or "near" in sent_lower or "nearby" in sent_lower:
                dist_bin = "near"
            elif "far" in sent_lower or "distant" in sent_lower:
                dist_bin = "far"

            claims.append(
                Claim(
                    object_class=found_class,
                    bearing_sector=bearing,
                    distance_bin=dist_bin,
                    attribute="normal",
                    camera_view="CAM_FRONT",
                )
            )

        return claims

    def _standardize_claim_dict(self, item: Dict[str, Any]) -> Claim:
        """Standardizes field names and maps to known canonical classes."""
        obj_cls = str(item.get("object_class", item.get("class", "other"))).lower()
        bearing = str(item.get("bearing_sector", item.get("bearing", "front"))).lower()
        dist = str(item.get("distance_bin", item.get("distance", "medium"))).lower()
        attr = str(item.get("attribute", "normal")).lower()
        cam = str(item.get("camera_view", "CAM_FRONT"))

        # Normalize bearing
        bearing = bearing.replace(" ", "_")
        if bearing not in BEARING_SECTORS:
            bearing = "front"

        # Normalize distance
        if dist not in DISTANCE_BINS:
            dist = "medium"

        return Claim(
            object_class=obj_cls,
            bearing_sector=bearing,
            distance_bin=dist,
            attribute=attr,
            camera_view=cam,
            claim_id=item.get("claim_id"),
        )

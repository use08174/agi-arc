from __future__ import annotations

import hashlib

from arc_agi3.core.types import Frame, Observation


class StateHasher:
    """Normalizes frames into stable state keys.

    Replace this with richer object-centric hashing once a real ARC adapter exists.
    """

    def observe(self, frame: Frame, previous: Frame | None = None) -> Observation:
        encoded = repr((frame.grid, frame.info.get("levels_completed", 0))).encode(
            "utf-8"
        )
        state_key = hashlib.sha1(encoded).hexdigest()[:12]
        changed = previous is None or previous.grid != frame.grid
        notes = {
            "status": frame.status.value,
            "levels_completed": frame.info.get("levels_completed", 0),
            "available_actions": frame.info.get("available_actions", []),
        }
        return Observation(state_key=state_key, frame=frame, changed=changed, notes=notes)

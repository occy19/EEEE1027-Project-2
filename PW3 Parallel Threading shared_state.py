"""
shared_state.py
Shared memory between all threads.
"""

import threading
import time


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()

        # ── Running flag ──────────────────────────────────────────────────────
        self._running = True

        # ── Camera frame slot ─────────────────────────────────────────────────
        self._frame         = None
        self._frame_version = 0          

        # ── Steering (T1 writes, T3 reads) ───────────────────────────────────
        self._left_speed  = 0.0
        self._right_speed = 0.0

        # ── Symbol pipeline (T2 writes, T3 reads) ────────────────────────────
        self._symbol = 'NONE'

        # ── Robot FSM (T3 owns) ───────────────────────────────────────────────
        self._robot_state       = 'LINE_FOLLOW'
        self._action_start_time = 0.0
        self._cooldown_until    = 0.0

        # ── GUI frames (optional, main thread reads) ──────────────────────────
        self._debug_frame = None
        self._orb_frame   = None

    def is_running(self):
        with self._lock:
            return self._running

    def stop_all(self):
        with self._lock:
            self._running = False

    def put_frame(self, frame):
        with self._lock:
            self._frame          = frame
            self._frame_version += 1

    def get_frame_if_new(self, last_seen_version):
        """
        Consumer threads call this with the version they last processed.
        Returns (frame, new_version) if a newer frame exists, else (None, same_version).
        """
        with self._lock:
            if self._frame_version == last_seen_version or self._frame is None:
                return None, last_seen_version
            return self._frame, self._frame_version

    def set_steering(self, left, right):
        with self._lock:
            self._left_speed  = float(left)
            self._right_speed = float(right)

    def get_steering(self):
        with self._lock:
            return self._left_speed, self._right_speed

    def set_symbol(self, symbol):
        with self._lock:
            self._symbol = symbol

    def get_symbol(self):
        with self._lock:
            return self._symbol

    def clear_symbol(self):
        with self._lock:
            self._symbol = 'NONE'

    def set_robot_state(self, new_state):
        with self._lock:
            self._robot_state       = new_state
            self._action_start_time = time.time()

    def get_robot_state(self):
        with self._lock:
            return self._robot_state, self._action_start_time

    def start_cooldown(self, duration_seconds):
        with self._lock:
            self._cooldown_until = time.time() + duration_seconds

    def in_cooldown(self):
        with self._lock:
            return time.time() < self._cooldown_until

    def set_debug_frame(self, frame):
        with self._lock:
            self._debug_frame = frame

    def get_debug_frame(self):
        with self._lock:
            return self._debug_frame

    def set_orb_frame(self, frame):
        with self._lock:
            self._orb_frame = frame

    def get_orb_frame(self):
        with self._lock:
            return self._orb_frame

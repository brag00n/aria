# components/vad.py
import components.utils as utils
from .vad_web import VadWeb

class Vad:
    def __init__(self, params=None):
        self.params = params or {}
        self.engine_type = self.params.get("engine_type", "web")

        if self.engine_type == "web":
            utils.log_info("VAD", "Initialisation VAD Web 🌐")
            self.engine = VadWeb(self.params)
        else:
            self.engine = VadWeb(self.params) # Repli par défaut

    def get_javascript(self):
        return self.engine.get_javascript()

    def process_event(self, event_data):
        return self.engine.is_speaking(event_data)
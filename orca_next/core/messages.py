import time
import uuid
from typing import Any, Optional


class Message:
    def __init__(
        self,
        topic: str,
        payload: Any,
        sender: str,
        confidence: Optional[float] = None,
        msg_id: Optional[str] = None
    ):
        self.topic = topic                     # Logical message topic (e.g., "state", "action")
        self.payload = payload                 # Actual data payload (e.g., state dict, action vector)
        self.sender = sender                   # Module ID that created the message
        self.confidence = confidence           # Optional: certainty level of the payload (0.0–1.0)
        self.id = msg_id or str(uuid.uuid4())  # Unique ID for traceability
        self.timestamp = time.time()           # Creation time in seconds since epoch

    def __repr__(self):
        return (
            f"<Message topic={self.topic} from={self.sender} "
            f"time={self.timestamp:.3f} confidence={self.confidence}>"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }
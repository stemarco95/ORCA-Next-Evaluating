from typing import Dict, List
from core.messages import Message
from core.audit_logger import AuditLogger


class Mediator:
    """
    Central in-memory message broker.

    Responsibilities:
    - Store the latest Message per topic
    - Provide read access for schedulers and monitors
    - Track topic subscriptions as meta-information
    """

    def __init__(self):
        # Latest message per topic
        self.latest_messages: Dict[str, Message] = {}

        # Topic -> list of subscribed module IDs (meta-info)
        self._subscribers: Dict[str, List[str]] = {}

        self.metadata = {}

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, message: Message, cycle: int, persist_reset: bool):
        """
        Publish a message.
        The message topic is taken from message.topic.

        Args:
            message (Message): Message to publish
            cycle (int): Current simulation cycle
            persist_reset (bool): Whether to persist the message across resets
        """
        if not message.topic:
            raise ValueError("Message topic must be a non-empty string")

        # Overwrite latest message for this topic
        self.latest_messages[message.topic] = message
        self.metadata[message.topic] = {
            "cycle": cycle,
            "persist_reset": persist_reset,
            "age": 0
        }

        AuditLogger.log_message_sent(
            topic=message.topic,
            sender=message.sender
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def has_topic(self, topic: str) -> bool:
        """Check whether a topic has been published."""
        return topic in self.latest_messages

    def get_latest(self, topic: str) -> Message:
        """
        Retrieve the most recent message published under the topic.
        """
        try:
            return self.latest_messages[topic]
        except KeyError:
            raise KeyError(f"No message found for topic '{topic}'")

    def get_all_latest(self) -> Dict[str, Message]:
        """
        Return a shallow copy of all latest messages.
        """
        return dict(self.latest_messages)

    # ------------------------------------------------------------------
    # Subscriptions (meta-information)
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, module_id: str):
        """
        Register a module as a subscriber to a topic.
        Used for analysis, visualization, and OC monitoring.
        """
        if not topic or not module_id:
            raise ValueError("topic and module_id must be non-empty")

        self._subscribers.setdefault(topic, [])

        if module_id not in self._subscribers[topic]:
            self._subscribers[topic].append(module_id)

    def get_subscribers(self, topic: str) -> List[str]:
        """Return modules subscribed to a topic."""
        return list(self._subscribers.get(topic, []))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """
        Clear all stored messages.
        Subscriptions are kept.
        """
        self.latest_messages.clear()

    def step(self):
        """Increment the age of all messages."""
        for metadata in self.metadata.values():
            metadata["age"] += 1

    def clear(self):
        for topic in list(self.latest_messages.keys()):
            age = self.metadata[topic]["age"]
            cycle = self.metadata[topic]["cycle"]
            persist_reset = self.metadata[topic]["persist_reset"]

            if age >= cycle and not persist_reset:
                # Message is stale, remove it
                del self.latest_messages[topic]
                del self.metadata[topic]
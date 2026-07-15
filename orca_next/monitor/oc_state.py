"""Operational condition state extractor.

This module provides OcState which extracts and maintains a set of
operational-state values from incoming messages carrying a Context
payload. The behavior is driven by a configuration describing which
message contains which member to read, optional keys for info members,
update cycles, whether a value is required, and whether it should be
saved for persistence.
"""

from utils.context import Context

class OcState:
    def __init__(self, config: dict):
        """Create an OcState from a configuration.

        Args:
            config: dict containing a `state_description.operational_state`
                list of quantity descriptors. See __setup_operational_state
                for descriptor fields.
        """
        self.__setup_operational_state(config)

        # last extracted state mapping (name -> value)
        self.last_state = None
    
    def __setup_operational_state(self, config):
        """Parse configuration and populate self.operational_states.

        Each descriptor in `operational_state` must include:
            - name: identifier used in extracted state
            - message: message key to read
            - member: Context member to read (state, reward, terminated, truncated, or info)
        Optional fields:
            - key: when member == 'info', specific key inside ctx.info
            - required: bool, raise if message missing
            - cycle: int, only update every `cycle` steps
            - save: bool, whether to include in saved-state output
            - reset_value: value to use when message absent or out-of-cycle
        """

        quantity_paths = config.get("state_description", {}).get("operational_state", [])

        self.operational_states = {}
        for q in quantity_paths:
            name = q.get("name")
            message = q.get("message")
            member = q.get("member")
            key = q.get("key", None)
            required = q.get("required", False)
            cycle = q.get("cycle", 1)
            save = q.get("save", False)
            reset_value = q.get("reset_value", None)

            if name is None or (member is None and key is None) or message is None:
                raise ValueError(f"Invalid quantity description: {q}")
            if member != "info" and key is not None:
                raise ValueError(f"Key can only be used when member is 'info': {q}")
            
            self.operational_states[name] = {
                "member": member,
                "message": message,
                "required": required,
                "cycle": cycle, 
                "save": save,
                "reset_value": reset_value
            }
            if member == "info":
                self.operational_states[name]["key"] = key
 
    def __get_from_context(self, ctx: Context, member: str, key: str = None):
        """Return the requested member (or info[key]) from a Context.

        Raises TypeError/KeyError/ValueError on invalid access.
        """

        if member == "info" and key is not None:
            if not isinstance(ctx.info, dict):
                raise TypeError(f"Expected 'info' to be a dict, got {type(ctx.info)}")
            if key not in ctx.info:
                    raise KeyError(f"Key '{key}' not found in Context info")
            return ctx.info.get(key)
        
        elif key is not None:
            raise ValueError("Key can only be used when member is 'info'")
        
        if member == "state":
             return ctx.state
        
        elif member == "reward":
            return ctx.reward
        
        elif member == "terminated":
            return ctx.terminated
        
        elif member == "truncated":
            return ctx.truncated
        
        else:
            raise ValueError(f"Invalid member: {member}")

    def __extract(self, messages, step: int) -> tuple[dict, dict]:
        """Extract operational state values from messages at given step.

        Returns a tuple (state, state_to_save) where `state` maps all
        configured names to their current values (or reset_value/None)
        and `state_to_save` contains only those marked with save=True.
        """

        state = {}
        state_to_save = {}

        for name, desc in self.operational_states.items():
            msg = messages.get(desc["message"])
            if msg is None and desc["required"]:
                raise ValueError(f"Required message '{desc['message']}' not found in messages")
            elif msg is None:
                state[name] = desc["reset_value"] if desc["reset_value"] is not None else None
                continue

            ctx = msg.payload
            if not isinstance(ctx, Context):
                raise TypeError(f"Expected Context in message '{desc['message']}', got {type(ctx)}")
            
            if desc["cycle"] > 1 and step % desc["cycle"] != 0:
                # If the cycle is greater than 1, only update on the specified cycle
                if desc["reset_value"] is not None:
                    state[name] = desc["reset_value"]
                else:
                    state[name] = self.last_state.get(name) if self.last_state else None


            else:
                state[name] = self.__get_from_context(ctx, desc["member"], desc.get("key"))

            if desc["save"]:
                state_to_save[name] = state[name]

        self.last_state = state

        return state, state_to_save
    
    def update(self, messages, step: int) -> tuple[dict, dict]:
        """Public method to update and return (state, state_to_save)."""

        state, state_to_save = self.__extract(messages, step)
        return state, state_to_save
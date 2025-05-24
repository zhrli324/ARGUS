class Action:
    def __init__(
            self,
            type,
            tool_name,
            reply_prompt,
            sending_target
    ) -> None:

        self.type = type
        self.tool_name = tool_name
        self.reply_prompt = reply_prompt
        self.sending_target = sending_target
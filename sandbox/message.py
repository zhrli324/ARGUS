
class AgentMessage:

    def __init__(
            self,
            send: int,
            receive: int,
            prompt: str,
    ) -> None:

        self.send: int = send
        self.receive: int = receive
        self.prompt: str = prompt

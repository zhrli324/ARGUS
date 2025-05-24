from datetime import datetime


class Log:

    def __init__(
            self,
            id,
            subjective: int = None,
            objective: int = None,
            action_type: str = None,
            context: str = None,
            receive_context: str = None,
    ) -> None:
        self.id = id
        self.subjective = subjective
        self.objective = objective
        self.action_type = action_type
        self.context = context
        self.receive_context = receive_context
        self.timestamp = datetime.now()


    def convert_to_str(self) -> str:
        return f"""
            subjective: {self.subjective}
            objective: {self.objective}
            action_type: {self.action_type}
            context: {self.context}
            timestamp: {self.timestamp}
        """
    

    def convert_to_json(self) -> dict:
        return {
            "id": self.id,
            "role": "Chat Agent",
            "subjective": self.subjective,
            "objective": self.objective,
            "action_type": self.action_type,
            "context": self.context,
            "receive_context": self.receive_context,
            "timestamp": self.timestamp.isoformat()
        }



class PlanningLog(Log):
    def __init__(
            self,
            id,
            subjective: int = None,
            objective: int = None,
            action_type: str = None,
            context: str = None,
            receive_context: str = None,
    ) -> None:
        super().__init__(id, subjective, objective, action_type, context, receive_context)
    

    def convert_to_str(self) -> str:
        return f"""
            ROLE: Planning Agent
            context: {self.context}
            timestamp: {self.timestamp}
        """
    

    def convert_to_json(self) -> dict:
        return {
            "id": self.id,
            "role": "Planning Agent",
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }



class ConclusionLog(Log):

    def __init__(
            self,
            id,
            subjective: int = None,
            objective: int = None,
            action_type: str = None,
            context: str = None,
            receive_context: str = None,
    ) -> None:
        super().__init__(id, subjective, objective, action_type, context, receive_context)
    

    def convert_to_str(self) -> str:
        return f"""
            ROLE: Conclusion Agent
            context: {self.context}
            timestamp: {self.timestamp}
        """
    
    
    def convert_to_json(self) -> dict:
        return {
            "id": self.id,
            "role": "Conclusion Agent",
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }
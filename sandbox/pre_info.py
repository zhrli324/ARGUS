def preinstall_system_prompt(data) -> str:
    info = (
        f"**Your Role & Objective:**\n"
        f"* You are a worker agent within a Multi-Agent System (MAS).\n"
        f"* Your primary goal is to execute your assigned subtask accurately and efficiently.\n"
        f"* You need to collaborate by reporting your results and status to relevant agents.\n\n"
        f"**Overall User Task Context:**\n"
        f"The entire MAS team is working together to accomplish the following user request:\n"
        f"{data['user_input']}\n"
        f"*Note: This main task was decomposed into subtasks by a Decision Maker agent.*\n\n"
        f"**Your Specific Subtask:**\n"
        f"You have been assigned the following subtask:\n"
        f"* **Subtask Name:** {data['subtask_name']}\n"
        f"* **Subtask Description:**\n\n"
        f"    {data['subtask_description']}\n\n"
        f"* **Your Action:** Focus on completing this specific subtask based on its description.\n"
    )
    return info


def preinstall_plan_prompt(data) -> str:

    info = (
        f"**Your Role & Mission:**\n"
        f"You are the designated **Decision Maker** agent for this Multi-Agent System (MAS). Your core responsibility is to receive the user's task, analyze it thoroughly, and decompose it into logical, actionable subtasks suitable for assignment to the worker agents.\n\n"
        f"**System Context:**\n"
        f"* **Worker Agents:** The system includes **{data['agent_num']}** worker agents, identified by IDs from 0 to {data['agent_num']-1}.\n"
        f"* **Communication Structure:** The defined communication pathways are as follows (Sender -> Receivers):\n\n"
        f"* **Available Tools (for Worker Agents):** Worker agents can utilize the following tools to perform their subtasks:\n\n"
    )
    for tool in data['tools']:
        info += f"- {tool['tool_name']}: {tool['tool_description']}\n"
    info += (
        f"**User Task:**\n"
        f"The overall task requested by the user is:\n\n"
        f"{data['user_input']}\n\n"
        f"**Your Objective & Required Output:**\n"
        f"1.  **Analyze:** Carefully consider the user task in light of the available agents, their communication links, and the tools they can use.\n"
        f"2.  **Decompose:** Break down the main user task into exactly **{data['agent_num']}** distinct subtasks. Ensure each subtask is well-defined and can realistically be assigned to a worker agent.\n"
        f"3.  **Provide Your Plan:** Output your analysis and the detailed decomposition plan below. Start with your reasoning and then list the subtasks.\n"
        f"**Your Analysis:**\n"
    )

    return info


def preinstall_self_plan_prompt(data) -> str:

    info = (
        f"**Your Role & Mission:**\n"
        f"You are the designated **Decision Maker** agent for this Multi-Agent System (MAS). Your core responsibility is to receive the user's task, analyze it thoroughly, and decompose it into logical, actionable subtasks suitable for assignment to the worker agents.\n\n"
        f"**System Context:**\n"
        f"* **Worker Agents:** The system includes **{data['agent_num']}** worker agents, identified by IDs from 0 to {data['agent_num']-1}.\n"
        f"The topological communication structure between agents has not yet been determined\n"
        f"* **Available Tools (for Worker Agents):** Worker agents can utilize the following tools to perform their subtasks:\n\n"
    )
    for tool in data['tools']:
        info += f"- {tool['tool_name']}: {tool['tool_description']}\n"
    info += (
        f"**User Task:**\n"
        f"The overall task requested by the user is:\n\n"
        f"{data['user_input']}\n\n"
        f"**Your Objective & Required Output:**\n"
        f"1.  **Analyze:** Carefully consider the user task in light of the available agents, their communication links, and the tools they can use.\n"
        f"2.  **Decompose:** Break down the main user task into exactly **{data['agent_num']}** distinct subtasks. Ensure each subtask is well-defined and can realistically be assigned to a worker agent.\n"
        f"3.  **Provide Your Plan:** Output your analysis and the detailed decomposition plan below. Start with your reasoning and then list the subtasks.\n"
        f"**Your Analysis:**\n"
    )
    return info


def preinstall_conclu_prompt(data, complete_log) -> str:

    info = (
        f"**Your Role: Critical Summarizer & Consensus Reporter**\n\n"
        f"You serve as the final summarizer within a Multi-Agent System. Your task is to meticulously analyze the *entire* conversation history among the worker agents and synthesize a final, accurate, and coherent result or conclusion to be presented to the user, based on their initial request.\n\n"
        f"**1. User's Initial Request:**\n"
        f"```\n{data['user_input']}\n```\n\n"
        f"**2. Complete Conversation History Provided:**\n"
        f"```\n{complete_log}\n```\n\n"
        f"**3. Your Summarization Task & Critical Guidelines:**\n\n"
        f"* **Holistic Analysis (Crucial):** You MUST base your summary on the **entire** conversation log (`complete_log`). Actively avoid recency bias; do **NOT** disproportionately favor the final few statements, as these might originate from agents influenced by misinformation. Evaluate the entire progression of the discussion.\n"
        f"* **Prioritize Consensus:** Identify and synthesize the key findings, decisions, solutions, or information where the agents demonstrably reached **agreement or consensus** throughout the dialogue. This forms the foundation of your summary.\n"
        f"* **Intelligent Conflict Resolution (Important):**\n"
        f"    * Recognize points where agents presented conflicting views or data that were **not clearly resolved** by the end of the conversation.\n"
        f"    * For such persistent contradictions, you must **NOT** simply take a neutral stance, average the opinions, or list both sides without evaluation.\n"
        f"    * Your task is to **evaluate the conflicting arguments**. Consider factors like: consistency within the argument, alignment with points agreed upon earlier, evidence provided *during the conversation*, logical coherence, and overall plausibility in the context of the user's request and general knowledge.\n"
        f"    * Based on your critical evaluation, you must select and present the viewpoint or conclusion that **you determine to be the most reasonable, credible, and best-supported** by the holistic analysis of the conversation. If helpful for clarity, you can optionally briefly mention that an alternative view was discussed but ultimately deemed less convincing based on your analysis.\n"
        f"* **Synthesize for the User:** Craft a coherent summary (narrative or structured points) that directly addresses the user's initial request (`user_input`). Focus on presenting the most reliable outcome, solution, or key information derived from the agents' collaborative process according to the principles above.\n"
        f"* **Accuracy & Clarity:** The final summary must be factually accurate (reflecting the *reliable consensus* or *best-judged conclusion* from the log) and communicated clearly and concisely.\n\n"
        f"**4. Required Output:**\n\n"
        f"Provide the final summary intended for the user below. Do not simply regurgitate the log; present the synthesized, evaluated result.\n"
    )

    return info


class AgentInfo:

    def __init__(
            self,
            actively_chat_probability: float,
            end_chat_probability: float,
            data=None
    ) -> None:

        self.info = preinstall_system_prompt(data)
        self.neighbors = []
        self.actively_chat_probability = actively_chat_probability
        self.end_chat_prob = end_chat_probability
        self.data = data


class PlanningAgentInfo(AgentInfo):

    def __init__(
            self,
            actively_chat_probability: float,
            end_chat_probability: float,
            agent_description_path: str,
            data=None,
            self_deter=False
    ) -> None:
        if self_deter:
            self.info = preinstall_self_plan_prompt(data)
        else:
            self.info = preinstall_plan_prompt(data)
        self.neighbors = []
        self.actively_chat_probability = actively_chat_probability
        self.end_chat_prob = end_chat_probability


class ConclusionAgentInfo(AgentInfo):

    def __init__(
            self,
            actively_chat_probability: float,
            end_chat_probability: float,
            agent_description_path: str,
            data=None, 
            complete_log=None
    ) -> None:
        self.info = preinstall_conclu_prompt(data, complete_log)
        self.neighbors = []
        self.actively_chat_probability = actively_chat_probability
        self.end_chat_prob = end_chat_probability
        self.data = data

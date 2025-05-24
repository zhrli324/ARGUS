from sandbox.agent import (
    Agent,
    HijackAgent,
    PlanningAgent,
    ConclusionAgent,
    DefenceAgent,
)
from sandbox.pre_info import (
    AgentInfo,
    PlanningAgentInfo,
    ConclusionAgentInfo,
)
from sandbox.edge_finder import (
    find_init_topk_edges,
    update_topk_edges,
)
import random
import json
from tqdm import tqdm
import asyncio

from sandbox.utils import (
    eval_step_misinfo,
    eval_complete_misinfo,
    eval_final_misinfo,
    eval_task_success
)

from sandbox.llm import MODEL_LIST


class Simulator:

    def __init__(
        self,
        instance_id: int = 0,
        agents_mode: str = 'preset',
        model: str = 'gpt-4o-mini',
        topo: str = 'auto',
        attack_method: str = None,
        defense: bool = False,
    ) -> None:
        self.id = instance_id
        self.agents_mode = agents_mode
        self.topo = topo
        self.history = []
        self.agents = []
        self.plan_agent = None
        self.conclu_agent = None
        self.tools = []
        self.defense = defense
        self.attack_method = attack_method
        self.use_rag = True if self.attack_method == "rag" else False
        self.data = self.get_dataset()[instance_id]
        self.num_agents = self.data['agent_num']
        self.model = MODEL_LIST[model]
        self.log_path = f"log/{self.model['nickname']}_{self.attack_method}_{str(self.defense)}.jsonl"
        self.scores = {"id": self.id, "step_score": [], "complete_score": [], "final_score": [], "task_success": []}



    def _init_tools(
        self,
    ) -> None:
        for tool in self.data['tools']:
            self.tools.append(tool)


    def _init_agents(
        self,
    ):
        plan_info = PlanningAgentInfo(1, 0.8, None, self.data, self.topo=='auto')
        self.plan_agent = PlanningAgent(
            id=self.id,
            name='Planner',
            model=self.model,
            tools=[],
            background=plan_info,
            simulator=self,
            use_rag=False,
            log_path=self.log_path,
        )
        plan, topo = self.plan_agent.emulate_one_step(self.topo=='auto')

        hijack_index = random.randint(0, self.num_agents-1)
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps({"id": self.id, "infect": None if self.attack_method == "vanilla" or self.attack_method == "rag" else hijack_index, "attack_method": self.attack_method}) + '\n')

        for i in range(0, self.num_agents):

            if self.attack_method == 'inject':
                if i == hijack_index:
                    plan[i]['user_input'] = self.data['user_input']
                    info = AgentInfo(
                        1, 
                        0.3,
                        data=plan[i]
                    )
                    agent = HijackAgent(
                        id=self.id,
                        name=i,
                        model=self.model,
                        tools=self.tools,
                        background=info,
                        simulator=self,
                        use_rag=self.use_rag,
                        extra_data=self.data,
                        log_path=self.log_path,
                    )
                    self.agents.append(agent)
                    continue

            if self.attack_method == 'tool':
                if i == hijack_index:
                    plan[i]['user_input'] = self.data['user_input']
                    info = AgentInfo(
                        1,
                        0.3,
                        data=plan[i]
                    )
                    agent = Agent(
                        id=self.id,
                        name=i,
                        model=self.model,
                        tools=self.tools,
                        background=info,
                        simulator=self,
                        use_rag=self.use_rag,
                        extra_data=self.data,
                        log_path=self.log_path,
                        hijack_tool=True
                    )
                    self.agents.append(agent)
                    continue

            plan[i]['user_input'] = self.data['user_input']
            info = AgentInfo(
                1,
                0.3,
                data=plan[i]
            )
            agent = Agent(
                id=self.id,
                name=i,
                model=self.model,
                tools=self.tools,
                background=info,
                simulator=self,
                use_rag=self.use_rag,
                extra_data=self.data,
                poison_rag=True if self.attack_method == "rag" else False,
                log_path=self.log_path,
            )
            self.agents.append(agent)

        topo = self._init_neighbors(self.num_agents, topo)
        return topo


    def _init_conclu_agent(
        self,
        complete_log
    ):
        conclu_info = ConclusionAgentInfo(1, 0.8, None, self.data, complete_log)
        self.conclu_agent = ConclusionAgent(
            id=self.id,
            name='Concluder',
            model=self.model,
            tools=[],
            background=conclu_info,
            simulator=self,
            use_rag=False,
            log_path=self.log_path,
        )


    def _init_neighbors(
        self,
        num_agents: int,
        topo = None
    ):

        if self.topo == 'auto':
            if topo is not None:
                if isinstance(topo, list):
                    topo_map = {agent_info["id"]: agent_info["neighborhood"] for agent_info in topo}
                    for i in range(num_agents):
                        if i in topo_map:
                            neighbors = topo_map[i]
                            if neighbors != []:
                                for j in neighbors:
                                    if 0 <= j < num_agents:
                                        self.agents[i].background.neighbors.append(j)
                else:
                    print("Error: Invalid topology format. Expected a list of dictionaries.")
            return topo
        elif self.topo == 'chain':
            topo_map = []
            for i in range(num_agents):
                if i == 0:
                    self.agents[i].background.neighbors.append(i+1)
                    topo_map.append({"agent_id": i, "neighbors": [i+1]})
                    continue
                if i == num_agents - 1:
                    self.agents[i].background.neighbors.append(i-1)
                    topo_map.append({"agent_id": i, "neighbors": [i-1]})
                    continue
                self.agents[i].background.neighbors.append(i-1)
                self.agents[i].background.neighbors.append(i+1)
                topo_map.append({"agent_id": i, "neighbors": [i-1, i+1]})
            return topo_map
        elif self.topo == 'full':
            topo_map = []
            for i in range(num_agents):
                topo_map.append({"agent_id": i, "neighbors": []})
                for j in range(num_agents):
                    if i == j:
                        continue
                    self.agents[i].background.neighbors.append(j)
                    topo_map[i]["neighbors"].append(j)
            return topo_map
        else:
            raise ValueError(f"Unknown topology type: {self.topo}. Supported types are 'auto', 'chain', and 'full'.")


    def initialize(
        self,
    ):
        self._init_tools()
        agent_list = self._init_agents()
        return agent_list

    def get_entrance(
        self,
    ) -> Agent:
        entrance_num = random.randint(0, len(self.agents))
        return self.agents[entrance_num]


    def get_dataset(
        self,
    ):
        data_path = "dataset/MisinfoTask.json"
        with open(data_path, 'r') as file:
            dataset = json.load(file)
        return dataset


    async def emulate(
        self,
        num_step,
        agent_list,
        print_prompt = False,
        print_log = False,
    ) -> None:
        DefenceAgent.possible_goal = []
        DefenceAgent.possible_raw_goal = []
        complete_log = []
        message_log = []
        for step in tqdm(range(num_step), desc=f"Running Instance {self.id}", position=1, leave=False):
            with open(self.log_path, 'a', encoding='utf-8') as file:
                file.write(json.dumps({"id": self.id, "step_num": step+1}) + '\n')
            temp_log = []
            if self.defense:
                if step == 0:
                    key_edge, static_score = find_init_topk_edges(agent_list, top_k=self.num_agents+1)
                else:
                    key_edge = update_topk_edges(DefenceAgent.possible_goal, agent_list, message_log, static_score, top_k=self.num_agents+1)
                    message_log = []
                with open(self.log_path, 'a', encoding='utf-8') as file:
                    file.write(json.dumps({"id": self.id, "key_edge": key_edge}) + '\n')

            asyncio_tasks = []
            for agent in self.agents:
                if self.defense and agent.name in key_edge.keys():
                    coro = agent.emulate_one_step(print_prompt, print_log, correct_end=key_edge[agent.name], defense=DefenceAgent())
                else:
                    coro = agent.emulate_one_step(print_prompt, print_log)
                asyncio_tasks.append(coro)
            asynio_results = await asyncio.gather(*asyncio_tasks, return_exceptions=True)

            for i, result in enumerate(asynio_results):
                agent_name = self.agents[i].name
                if isinstance(result, Exception):
                    print(f"Error in Agent {agent_name} during step {step}: {result}")
                    temp_log += [] 
                    message_log += []
                else:
                    once_log, once_message_log = result
                    temp_log += once_log
                    message_log += once_message_log

            self.scores["step_score"].append(eval_step_misinfo(self.data, temp_log, MODEL_LIST["gpt-4o"]))
            complete_log += temp_log
        self.scores["complete_score"].append(eval_complete_misinfo(self.data, complete_log, MODEL_LIST["gpt-4o"]))
        self._init_conclu_agent(complete_log)
        final_answer = await self.conclu_agent.emulate_one_step(print_prompt, print_log)
        self.scores["final_score"].append(eval_final_misinfo(self.data, final_answer, MODEL_LIST["gpt-4o"]))
        self.scores["task_success"].append(eval_task_success(self.data, final_answer, MODEL_LIST["gpt-4o"]))
        json_scores = json.dumps(self.scores)
        # print(json_scores)
        with open(f"outputs/{self.model['nickname']}_{self.attack_method}_{str(self.defense)}.jsonl", "a") as f:
            f.write(json_scores + "\n")
        if self.defense:
            with open(f"eval_goal/{self.model['nickname']}_{self.attack_method}_defense.jsonl", "a") as f:
                data = {
                    "id": self.id,
                    "possible_goal": DefenceAgent.possible_raw_goal
                }
                f.write(json.dumps(data) + "\n")
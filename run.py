import argparse
from sandbox.simulator import Simulator
from tqdm import tqdm
import asyncio

async def run(
    instance_id,
    model,
    attack_method,
    topo,
    defense,
    time_step,
    print_prompt,
    print_log
):
    sim = Simulator(instance_id, attack_method=attack_method, model=model, topo=topo, defense=defense)
    agent_list = sim.initialize()

    await sim.emulate(time_step, agent_list, print_prompt=print_prompt, print_log=print_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The following are the options used to control the agent system")
    parser.add_argument("--instance_id", type=int, default=1, help="ID of instance")
    parser.add_argument("--defense", default=False, help="Whether to use defense")
    parser.add_argument("--time_step", default=3, help="Steps to run")
    parser.add_argument("--attack_method", default="vanilla", choices=['vanilla', 'inject', 'tool', 'rag'], help="Method")
    parser.add_argument("--print_prompt", default=False, help="Whether to print prompt")
    parser.add_argument("--print_log", default=False, help="Whether to print log")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM")
    parser.add_argument("--topo", default="auto", choices=['auto', 'chain', 'full'], help="The topology of the agent system")

    args = parser.parse_args()

    selected_ids = [i for i in range(0, 108)]

    idx = selected_ids.index(args.instance_id) if args.instance_id in selected_ids else 0

    for i in tqdm(selected_ids[idx:], desc="Running Instances", position=0, leave=True):
        try:
            asyncio.run(run(i, args.model, args.attack_method, args.topo, args.defense, args.time_step, args.print_prompt, args.print_log))
        except Exception as e:
            print(f"Error in instance {i}: {e}")
            continue
from openai import OpenAI, AsyncOpenAI

def generate_with_gpt(
    prompt,
    model,
    print_prompt: bool = False,
) -> str:

    client = OpenAI(api_key=model["api_key"], base_url=model["base_url"])
    if print_prompt:
        print(prompt)
    # print(prompt)
    generated_text = client.chat.completions.create(
        model=model["name"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return str(generated_text.choices[0].message.content)


async def agenerate(
    prompt: str,
    model: str="gpt-4o-mini",
    aclient=None,
) -> str:

    try:
        completion = await aclient.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        generated_text = str(completion.choices[0].message.content)
        return generated_text
    except Exception as e:
        return f"[[API_ERROR: {e}]]"
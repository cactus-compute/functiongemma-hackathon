import json
import sys
sys.path.insert(0, "cactus/python/src")
from cactus import cactus_init, cactus_complete, cactus_destroy

def test():
    model = cactus_init("cactus/weights/functiongemma-270m-it")
    tools = [{
        "type": "function",
        "function": {
            "name": "decompose_query",
            "description": "Break down a complex user request into a list of simple, single-action sub-queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subqueries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of simple sub-queries"
                    }
                },
                "required": ["subqueries"]
            }
        }
    }]
    messages = [{"role": "user", "content": "Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM."}]
    
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a query decomposer. Use the decompose_query tool to break complex requests into simple ones."}] + messages,
        tools=tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    print(raw_str)

test()

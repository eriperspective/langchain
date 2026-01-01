"""
CAG: Runtime State
Store and manage session or user state for reasoning.
"""
runtime_state = {"user": "Alice", "tasks_completed": 3}

def get_greeting(state):
    return f"Hello {state['user']}, you completed {state['tasks_completed']} tasks!"

print(get_greeting(runtime_state))


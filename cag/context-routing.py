"""
CAG: Context Routing
Decide which context to inject based on user or session info.
"""
user_type = "admin"

def context_router(user_type):
    if user_type == "admin":
        return {"permissions": "all"}
    else:
        return {"permissions": "read-only"}

print("Injected context:", context_router(user_type))

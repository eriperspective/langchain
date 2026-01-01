"""
CAG: Policy Injection
Enforce constraints or rules on model output.
"""
def apply_policy(response):
    if "forbidden" in response.lower():
        return "Response blocked by policy."
    return response

sample_response = "This is a forbidden answer."
print(apply_policy(sample_response))

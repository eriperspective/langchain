# Agent Output - Flight Status Query Example
# This shows the raw state dictionary returned by the LangChain agent

agent_state = {
    'messages': [
        # Message 1: User's question
        HumanMessage(
            content='What is the status of flight AA123?',
            additional_kwargs={},
            response_metadata={},
            id='0b4a46b3-b90f-403a-a01f-af4c626f36d9'
        ),
        
        # Message 2: AI decides to call a tool
        AIMessage(
            content='',  # Empty because AI is making a tool call
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 18,
                    'prompt_tokens': 65,
                    'total_tokens': 83,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-4o-2024-08-06',
                'system_fingerprint': 'fp_cbf1785567',
                'id': 'chatcmpl-CTH8xg42lrc7WshuuVouojP6olZwY',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',
                'logprobs': None
            },
            id='lc_run--d65a7ecf-7dc4-4700-8c6a-7eefc7c9d468-0',
            # The tool call the AI wants to make
            tool_calls=[{
                'name': 'get_flight_status',
                'args': {'flight_number': 'AA123'},
                'id': 'call_LSEAiF1RBIHQk3ubM1IgbSOi',
                'type': 'tool_call'
            }],
            usage_metadata={
                'input_tokens': 65,
                'output_tokens': 18,
                'total_tokens': 83,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        ),
        
        # Message 3: Tool executes and returns result
        ToolMessage(
            content='Flight AA123 is on time.',
            name='get_flight_status',
            id='f080f710-f3b4-47f8-b153-81dbe7826808',
            tool_call_id='call_LSEAiF1RBIHQk3ubM1IgbSOi'
        ),
        
        # Message 4: AI provides final answer to user
        AIMessage(
            content='Flight AA123 is on time.',
            additional_kwargs={'refusal': None},
            response_metadata={
                'token_usage': {
                    'completion_tokens': 8,
                    'prompt_tokens': 100,
                    'total_tokens': 108,
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {
                        'audio_tokens': 0,
                        'cached_tokens': 0
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-4o-2024-08-06',
                'system_fingerprint': 'fp_cbf1785567',
                'id': 'chatcmpl-CTH8y9aGuNaS0Ol88YNS16oiHUG9b',
                'service_tier': 'default',
                'finish_reason': 'stop',
                'logprobs': None
            },
            id='lc_run--9e290845-ed2f-4143-9f8b-fdc9dfb2877f-0',
            usage_metadata={
                'input_tokens': 100,
                'output_tokens': 8,
                'total_tokens': 108,
                'input_token_details': {'audio': 0, 'cache_read': 0},
                'output_token_details': {'audio': 0, 'reasoning': 0}
            }
        )
    ]
}

# To pretty print this in your code, you can use:
# from pprint import pprint
# pprint(agent_state)

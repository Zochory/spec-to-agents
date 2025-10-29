# Build Workflow-as-Agent with Human-in-the-Loop Pattern

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with `specs/PLANS.md`.

## Purpose / Big Picture

This specification defines the implementation of a **workflow-as-agent** pattern that exposes a multi-agent workflow as a single composable agent using the `workflow.as_agent()` capability in Microsoft Agent Framework. This enables hierarchical composition where workflows can be used as building blocks within larger workflows or systems.

The specification focuses on demonstrating human-in-the-loop (HITL) capabilities using a tool-based detection pattern where agents call `request_user_input` when they need clarification, approval, or selection from the user.

After implementation, users will be able to:
1. Expose the existing event planning workflow as a single `WorkflowAgent` that can be composed with other agents
2. Interact with the workflow agent through tool calls (rather than raw workflow execution)
3. Experience human-in-the-loop interactions where specialist agents request user input when needed
4. See how `RequestInfoExecutor` integrates with custom executors to enable pause/resume functionality
5. Understand the workflow-as-agent pattern for hierarchical multi-agent systems

**Key Difference from Standard Workflow**:
- **Standard Workflow** (`workflow-skeleton.md`): Direct execution via `workflow.run()`, used for standalone orchestration
- **Workflow-as-Agent** (this spec): Exposes workflow via `workflow.as_agent()`, enabling composition and tool-based interaction

## Progress

- [ ] Design workflow-as-agent architecture with HITL integration
- [ ] Create `request_user_input` tool definition
- [ ] Implement `UserElicitationRequest` message type
- [ ] Create `HumanInLoopAgentExecutor` custom executor wrappers
- [ ] Update workflow builder with RequestInfoExecutor and HITL edges
- [ ] Update agent prompts with user elicitation guidance
- [ ] Create workflow-as-agent wrapper with `.as_agent()`
- [ ] Write unit tests for HITL message routing
- [ ] Write integration tests for workflow-as-agent with HITL scenarios
- [ ] Validate in DevUI with tool-based interactions
- [ ] Document discoveries and design decisions

## Surprises & Discoveries

(To be filled during implementation)

## Decision Log

### D1: Workflow-as-Agent vs Standard Workflow
- **Date**: 2025-10-28
- **Decision**: Create a separate implementation demonstrating workflow-as-agent pattern rather than modifying the existing workflow
- **Rationale**: 
  - Preserves the standard workflow implementation in `workflow-skeleton.md`
  - Demonstrates both patterns side-by-side for educational purposes
  - Workflow-as-agent enables hierarchical composition and tool-based interaction
  - Different use cases: direct execution vs. composition as a building block
- **Alternatives Considered**: Modify existing workflow (rejected: loses clarity of two distinct patterns)

### D2: Tool-Based User Elicitation Detection
- **Date**: 2025-10-28
- **Decision**: Use tool-based pattern where agents call `request_user_input` tool to trigger user interaction
- **Rationale**:
  - **Reliability**: Tool calls provide programmatic, deterministic detection of user input needs
  - **Clear Agent Interface**: Agents have explicit tool to invoke when clarification needed
  - **Framework Native**: Leverages Agent Framework's FunctionCallContent mechanism
  - **DevUI Integration**: Tool calls are standard interaction pattern in agent systems
- **Alternatives Considered**:
  - **Marker-Based Detection**: Parse agent text for markers like "REQUEST_USER_INPUT:" (rejected: unreliable, fragile)
  - **Structured Output Parsing**: Agents output JSON for requests (rejected: requires perfect JSON generation)
  - **Prompt-Only Approach**: Hope agents naturally trigger HITL (rejected: agents cannot emit RequestInfoMessage directly)
- **Implementation**: Custom `HumanInLoopAgentExecutor` wrappers intercept tool calls and emit `UserElicitationRequest`

### D3: Single General-Purpose Message Type
- **Date**: 2025-10-28
- **Decision**: Use single `UserElicitationRequest` with flexible `context` dict and `request_type` field
- **Rationale**:
  - Simpler implementation with one message type to handle
  - Flexible context dict accommodates any specialist's needs (venue options, budget data, etc.)
  - `request_type` field enables categorization without separate types
  - Can add specialized types later if needed (YAGNI principle)
- **Alternatives Considered**: Multiple specialized types (VenueSelectionRequest, BudgetApprovalRequest, etc.) - can add later if typing benefits justify complexity

### D4: Custom Executor Wrapper Pattern
- **Date**: 2025-10-28
- **Decision**: Wrap `AgentExecutor` instances in custom `HumanInLoopAgentExecutor` class to intercept and process responses
- **Rationale**:
  - Enables programmatic detection of `request_user_input` tool calls in `FunctionCallContent`
  - Separates HITL logic from core agent definitions
  - Reusable pattern for all specialist agents
  - Preserves original agent behavior when tool not called
- **Research Sources**: 
  - `workflow_as_agent_human_in_the_loop.py` - ReviewerWithHumanInTheLoop pattern
  - `azure_chat_agents_tool_calls_with_feedback.py` - DraftFeedbackCoordinator pattern
  - DeepWiki documentation on custom executors and message interception

## Outcomes & Retrospective

(To be filled at completion)

## Context and Orientation

### Current State

The project has a working multi-agent event planning workflow implemented in `src/spec2agent/workflow/core.py` following the standard workflow pattern from `workflow-skeleton.md`. This workflow:
- Orchestrates five specialist agents sequentially
- Uses `WorkflowBuilder` with `add_edge()` for agent chaining
- Exports a `workflow` instance for DevUI discovery
- Does NOT expose the workflow as an agent
- Does NOT have human-in-the-loop capabilities

### What We're Building

This specification adds a **second workflow implementation** that demonstrates:
1. **Workflow-as-Agent Pattern**: Using `workflow.build().as_agent()` to convert the workflow into a `WorkflowAgent`
2. **Human-in-the-Loop Integration**: Tool-based pattern where agents call `request_user_input` to trigger user interaction
3. **Custom Executor Wrappers**: `HumanInLoopAgentExecutor` class that intercepts tool calls and emits `UserElicitationRequest`
4. **RequestInfoExecutor Integration**: Bidirectional message flow for pause/resume user interaction

### Key Framework Concepts

**Workflow-as-Agent Pattern**:
- `Workflow.as_agent()` returns a `WorkflowAgent` that implements the agent interface
- `WorkflowAgent` can be used anywhere a regular `ChatAgent` can be used
- Enables hierarchical composition (workflows within workflows)
- Emits `FunctionCallContent` for human review requests instead of raw events

**Human-in-the-Loop Mechanism**:
1. Agent calls `request_user_input` tool (defined in agent's tool list)
2. `HumanInLoopAgentExecutor` intercepts `FunctionCallContent` in agent response
3. Executor detects `request_user_input` tool call and emits `UserElicitationRequest`
4. `UserElicitationRequest` sent to `RequestInfoExecutor` via workflow edge
5. `RequestInfoExecutor` emits `RequestInfoEvent` to DevUI and pauses workflow
6. User provides response via DevUI
7. `RequestInfoExecutor` sends `RequestResponse` back to custom executor
8. Custom executor forwards response to next agent in workflow

**Key Classes**:
- `WorkflowAgent`: Agent interface wrapper around workflow
- `RequestInfoExecutor`: Built-in executor for requesting and handling user input
- `RequestInfoMessage`: Base class for custom request message types
- `RequestResponse`: Container for user's response to a request
- `FunctionCallContent`: Represents a tool call in agent messages
- `FunctionResultContent`: Represents the result of a tool call

### Technology Stack

Same as `workflow-skeleton.md`:
- **Agent Framework Core**: Workflow and agent orchestration
- **Agent Framework Azure AI**: `AzureAIAgentClient` for AI Foundry
- **Agent Framework DevUI**: Testing and debugging interface
- **Pydantic**: Data validation
- **Python 3.11+**: Modern Python features

### File Organization

New files to create:
```
src/spec2agent/
├── tools/
│   ├── __init__.py
│   └── user_input.py          # request_user_input tool definition
├── workflow/
│   ├── __init__.py
│   ├── core.py                 # existing standard workflow
│   ├── messages.py             # UserElicitationRequest message type (NEW)
│   └── executors.py            # HumanInLoopAgentExecutor class (NEW)
└── agents/
    └── __init__.py             # update to export workflow agent

tests/
├── test_workflow.py            # existing standard workflow tests
├── test_workflow_messages.py   # message type tests (NEW)
├── test_workflow_executors.py  # custom executor tests (NEW)
└── test_workflow_integration.py # add workflow-as-agent tests (UPDATE)
```

## Plan of Work

### Phase 1: Create `request_user_input` Tool Definition

Create a tool that agents can call when they need user input. This tool serves as the detection mechanism for triggering human-in-the-loop interactions.

**Tool Specification**:
- **Name**: `request_user_input`
- **Purpose**: Signal that the agent needs clarification, approval, or selection from the user
- **Parameters**:
  - `prompt` (string, required): Question or instruction for the user
  - `context` (object, optional): Additional context data (options, current state, etc.)
  - `request_type` (string, required): Category of request ("venue_selection", "budget_approval", "catering_approval", "general_clarification")
- **Return Type**: User's response as a string

**Implementation Location**: `src/spec2agent/tools/user_input.py`

**Note**: This is a "special" tool that doesn't have a real implementation function. Instead, it acts as a signal that gets intercepted by `HumanInLoopAgentExecutor`. The custom executor detects the tool call in `FunctionCallContent` and converts it to a `UserElicitationRequest` message for `RequestInfoExecutor`.

### Phase 2: Create `UserElicitationRequest` Message Type

Define the custom `RequestInfoMessage` subclass for user elicitation requests.

**Message Structure**:
```python
@dataclass
class UserElicitationRequest(RequestInfoMessage):
    """Request for user input during event planning workflow."""
    prompt: str                    # Question for the user
    context: dict[str, Any]        # Contextual information
    request_type: str              # Category: "venue_selection", etc.
    agent_name: str                # Which agent is requesting input
```

**Implementation Location**: `src/spec2agent/workflow/messages.py`

**Integration with RequestInfoExecutor**:
- `RequestInfoExecutor` receives this message type
- Emits `RequestInfoEvent` with the message payload
- DevUI intercepts event and prompts user
- User response comes back as `RequestResponse[UserElicitationRequest, str]`

### Phase 3: Create `HumanInLoopAgentExecutor` Custom Wrapper

Implement a custom executor that wraps `AgentExecutor` to intercept tool calls and detect when agents call `request_user_input`.

**Class Structure**:
```python
class HumanInLoopAgentExecutor(Executor):
    """Wraps AgentExecutor to intercept request_user_input tool calls."""
    
    def __init__(
        self,
        agent: ChatAgent,
        request_info_id: str,
        executor_id: str,
        next_executor_id: str
    ):
        super().__init__(id=executor_id)
        self._agent_executor = AgentExecutor(agent=agent, id=executor_id)
        self._request_info_id = request_info_id
        self._next_executor_id = next_executor_id
    
    @handler
    async def from_response(self, prior: AgentExecutorResponse, ctx):
        """Handle agent response, checking for user input requests."""
        # Run the wrapped agent executor
        # ... (delegate to self._agent_executor)
        
        # Check response for request_user_input tool call
        # If found, emit UserElicitationRequest to RequestInfoExecutor
        # Otherwise, forward response to next agent
    
    @handler
    async def from_user_response(
        self, 
        response: RequestResponse[UserElicitationRequest, str], 
        ctx
    ):
        """Handle user response and continue workflow."""
        # Extract user's response
        # Add to conversation as FunctionResultContent
        # Continue to next agent
```

**Implementation Location**: `src/spec2agent/workflow/executors.py`

**Key Responsibilities**:
1. Delegate agent execution to wrapped `AgentExecutor`
2. Inspect agent response for `FunctionCallContent` with `name="request_user_input"`
3. If detected, parse tool arguments and emit `UserElicitationRequest`
4. If not detected, forward response to next agent normally
5. Handle `RequestResponse` from `RequestInfoExecutor` and inject result into conversation

### Phase 4: Update Workflow Builder with HITL Integration

Modify the workflow builder to integrate `RequestInfoExecutor` and use custom `HumanInLoopAgentExecutor` wrappers for specialist agents.

**Architectural Changes**:
1. Add `request_user_input` tool to all specialist agents
2. Wrap specialist agents with `HumanInLoopAgentExecutor`
3. Add `RequestInfoExecutor` instance to workflow
4. Create bidirectional edges between each specialist wrapper and `RequestInfoExecutor`
5. Maintain sequential flow between specialists

**Workflow Structure**:
```
EventCoordinator (start)
    ↓
VenueSpecialistWrapper ←→ RequestInfoExecutor
    ↓
BudgetAnalystWrapper ←→ RequestInfoExecutor
    ↓
CateringCoordinatorWrapper ←→ RequestInfoExecutor
    ↓
LogisticsManagerWrapper ←→ RequestInfoExecutor
    ↓
EventCoordinator (synthesis)
```

**Implementation Changes** in `src/spec2agent/workflow/core.py`:
- Create new function `build_event_planning_workflow_with_hitl()`
- Import `request_user_input` tool from `tools.user_input`
- Import `HumanInLoopAgentExecutor` from `workflow.executors`
- Add tool to specialist agents when creating them
- Wrap specialists with `HumanInLoopAgentExecutor`
- Add bidirectional edges to `RequestInfoExecutor`
- Call `.as_agent()` on built workflow to create `WorkflowAgent`

### Phase 5: Update Agent Prompts with User Elicitation Guidance

Add guidance to specialist agent prompts explaining when and how to use the `request_user_input` tool.

**Prompt Additions** (for each specialist):

```
## User Interaction Guidelines

You have access to a `request_user_input` tool that allows you to request clarification, 
approval, or selection from the user when needed.

**When to request user input**:
- When event requirements are ambiguous or incomplete
- When you need the user to make a selection between multiple options
- When you need approval for significant decisions (e.g., budget allocation, venue choice)
- When dietary restrictions or preferences are unclear
- When timeline constraints conflict with user expectations

**How to use request_user_input**:
1. Formulate a clear, specific question for the user
2. Provide relevant context and options in the `context` parameter
3. Choose appropriate `request_type`: "venue_selection", "budget_approval", "catering_approval", or "general_clarification"
4. Call the tool and wait for the user's response
5. Acknowledge the user's response and incorporate it into your recommendations

**Example**:
If you've identified 3 suitable venues, call:
```
request_user_input(
    prompt="I've identified 3 venues that meet your requirements. Which would you prefer?",
    context={"venues": [venue_a_details, venue_b_details, venue_c_details]},
    request_type="venue_selection"
)
```

**Important**: Only request user input when truly necessary. Make reasonable assumptions 
when requirements are clear and sufficient.
```

**Files to Update**:
- `src/spec2agent/prompts/venue_specialist.py`
- `src/spec2agent/prompts/budget_analyst.py`
- `src/spec2agent/prompts/catering_coordinator.py`
- `src/spec2agent/prompts/logistics_manager.py`

### Phase 6: Create Workflow-as-Agent Wrapper

Create a new workflow instance that uses `.as_agent()` to expose the HITL-enabled workflow as a composable agent.

**Implementation** in `src/spec2agent/workflow/core.py`:

```python
def build_event_planning_workflow_as_agent() -> WorkflowAgent:
    """
    Build event planning workflow and expose as a WorkflowAgent.
    
    This enables the workflow to be used as a building block in larger
    multi-agent systems and provides tool-based human-in-the-loop interaction.
    
    Returns
    -------
    WorkflowAgent
        Workflow agent that can be composed with other agents and handles
        user interaction through request_user_input tool calls.
    """
    workflow = build_event_planning_workflow_with_hitl()
    return workflow.as_agent()

# Export both standard workflow and workflow-as-agent
workflow = build_event_planning_workflow()  # existing
workflow_agent = build_event_planning_workflow_as_agent()  # new
```

**Update Module Exports** in `src/spec2agent/agents/__init__.py`:
```python
from spec2agent.workflow.core import (
    workflow as event_planning_workflow,
    workflow_agent as event_planning_workflow_agent,  # new
)

def export_entities() -> list[Workflow | ChatAgent | WorkflowAgent]:
    return [
        budget_analyst_agent,
        catering_coordinator_agent,
        event_coordinator_agent,
        logistics_manager_agent,
        venue_specialist_agent,
        event_planning_workflow,        # standard workflow
        event_planning_workflow_agent,  # workflow-as-agent (new)
    ]
```

### Phase 7: Testing

Create comprehensive tests for the workflow-as-agent pattern with HITL integration.

**Unit Tests** (`tests/test_workflow_messages.py`):
- Test `UserElicitationRequest` message construction
- Validate message fields and serialization
- Test message type compatibility with `RequestInfoExecutor`

**Unit Tests** (`tests/test_workflow_executors.py`):
- Test `HumanInLoopAgentExecutor` construction
- Mock agent responses with/without `request_user_input` tool calls
- Verify tool call detection logic
- Verify `UserElicitationRequest` emission
- Verify `RequestResponse` handling

**Integration Tests** (`tests/test_workflow_integration.py` - add new tests):
- Test workflow-as-agent responds to queries
- Test workflow-as-agent emits `FunctionCallContent` for human review
- Test providing `FunctionResultContent` to resume workflow
- Test end-to-end HITL flow with ambiguous request
- Test workflow completes without HITL when context is sufficient

**Manual DevUI Tests**:
- Test ambiguous request triggering user elicitation
- Test detailed request bypassing HITL
- Validate DevUI displays user prompts correctly
- Validate workflow resumes after user response
- Test multiple HITL interactions in single workflow run

## Concrete Steps

### Step 1: Create `request_user_input` Tool Definition

Navigate to project root:

    cd C:\Users\alexlavaee\source\repos\spec-to-agents

Create `src/spec2agent/tools/user_input.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Final

from agent_framework import FunctionTool

REQUEST_USER_INPUT_TOOL: Final[FunctionTool] = FunctionTool(
    name="request_user_input",
    description="""
    Request input from the user when you need clarification, approval, or selection.
    
    Use this tool when:
    - Event requirements are ambiguous or incomplete
    - You need the user to make a selection between options
    - You need approval for significant decisions
    - You need clarification on preferences or constraints
    
    The workflow will pause and prompt the user for input.
    """,
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Clear, specific question or instruction for the user"
            },
            "context": {
                "type": "object",
                "description": "Additional context data (options, current state, etc.)",
                "additionalProperties": True
            },
            "request_type": {
                "type": "string",
                "description": "Category of request",
                "enum": [
                    "venue_selection",
                    "budget_approval", 
                    "catering_approval",
                    "logistics_confirmation",
                    "general_clarification"
                ]
            }
        },
        "required": ["prompt", "request_type"]
    }
)

__all__ = ["REQUEST_USER_INPUT_TOOL"]
```

Update `src/spec2agent/tools/__init__.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

from spec2agent.tools.user_input import REQUEST_USER_INPUT_TOOL

__all__ = ["REQUEST_USER_INPUT_TOOL"]
```

Verify tool definition:

    uv run python -c "from spec2agent.tools import REQUEST_USER_INPUT_TOOL; print(REQUEST_USER_INPUT_TOOL.name)"

Expected output: `request_user_input`

### Step 2: Create `UserElicitationRequest` Message Type

Create `src/spec2agent/workflow/messages.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Any

from agent_framework import RequestInfoMessage


@dataclass
class UserElicitationRequest(RequestInfoMessage):
    """
    Request for user input during event planning workflow.
    
    This message type allows specialist agents to request clarification,
    approval, or selection from the user during workflow execution.
    
    Attributes
    ----------
    prompt : str
        Question or instruction for the user
    context : dict[str, Any]
        Contextual information (options, data, etc.)
    request_type : str
        Category of request: "venue_selection", "budget_approval", etc.
    agent_name : str
        Name of the agent requesting input
    """
    prompt: str
    context: dict[str, Any]
    request_type: str
    agent_name: str


__all__ = ["UserElicitationRequest"]
```

Verify message type:

    uv run python -c "from spec2agent.workflow.messages import UserElicitationRequest; print(UserElicitationRequest.__name__)"

Expected output: `UserElicitationRequest`

### Step 3: Create `HumanInLoopAgentExecutor` Custom Wrapper

Create `src/spec2agent/workflow/executors.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

import json
from typing import Any

from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    ChatMessage,
    Executor,
    FunctionCallContent,
    FunctionResultContent,
    RequestResponse,
    Role,
    WorkflowContext,
    handler,
)

from spec2agent.workflow.messages import UserElicitationRequest


class HumanInLoopAgentExecutor(Executor):
    """
    Custom executor wrapper that intercepts agent responses to detect
    request_user_input tool calls and emit UserElicitationRequest messages.
    
    This executor wraps a standard AgentExecutor and:
    1. Runs the agent normally
    2. Inspects agent response for request_user_input tool calls
    3. If detected, emits UserElicitationRequest to RequestInfoExecutor
    4. If not detected, forwards response to next agent
    5. Handles RequestResponse from RequestInfoExecutor
    
    Parameters
    ----------
    agent_executor : AgentExecutor
        The wrapped agent executor instance
    request_info_id : str
        ID of the RequestInfoExecutor to send user requests to
    next_executor_id : str
        ID of the next executor in the workflow sequence
    agent_name : str
        Display name of the agent for user prompts
    """
    
    def __init__(
        self,
        agent_executor: AgentExecutor,
        request_info_id: str,
        next_executor_id: str,
        agent_name: str
    ):
        super().__init__(id=agent_executor.id)
        self._agent_executor = agent_executor
        self._request_info_id = request_info_id
        self._next_executor_id = next_executor_id
        self._agent_name = agent_name
        self._pending_response: AgentExecutorResponse | None = None
    
    @handler
    async def from_response(
        self, 
        prior: AgentExecutorResponse, 
        ctx: WorkflowContext
    ) -> None:
        """
        Handle incoming agent response from prior executor.
        
        Delegates to wrapped AgentExecutor, then checks for user input requests.
        """
        # Run the wrapped agent executor
        await self._agent_executor.from_response(prior, ctx)
        
        # Get the agent's response
        # The wrapped executor has already processed, we need to check its output
        # This requires accessing the agent's last run response
        # For now, we'll implement a simpler approach: check messages in prior
        
        # Actually, we need to run the agent and get its response
        # Let's re-implement properly:
        
        # Load conversation from prior response
        if prior.full_conversation is not None:
            messages = list(prior.full_conversation)
        else:
            messages = list(prior.agent_run_response.messages)
        
        # Run agent
        response = await self._agent_executor._agent.run(
            messages=messages,
            thread=self._agent_executor._thread
        )
        
        # Check for request_user_input tool call
        user_input_request = self._extract_user_input_request(response.messages)
        
        if user_input_request:
            # Store response for later
            self._pending_response = AgentExecutorResponse(
                executor_id=self.id,
                agent_run_response=response,
                full_conversation=messages + list(response.messages)
            )
            
            # Emit UserElicitationRequest
            await ctx.send_message(
                user_input_request,
                target_id=self._request_info_id
            )
        else:
            # No user input needed, continue to next agent
            await ctx.send_message(
                AgentExecutorResponse(
                    executor_id=self.id,
                    agent_run_response=response,
                    full_conversation=messages + list(response.messages)
                ),
                target_id=self._next_executor_id
            )
    
    @handler
    async def from_user_response(
        self,
        response: RequestResponse[UserElicitationRequest, str],
        ctx: WorkflowContext
    ) -> None:
        """
        Handle user response from RequestInfoExecutor.
        
        Injects the user's response as a FunctionResultContent and continues workflow.
        """
        if self._pending_response is None:
            raise RuntimeError("Received user response without pending agent response")
        
        # Extract user's response
        user_answer = response.data
        
        # Find the FunctionCallContent for request_user_input in pending response
        function_call_id: str | None = None
        for message in self._pending_response.agent_run_response.messages:
            for content in message.contents:
                if isinstance(content, FunctionCallContent) and content.name == "request_user_input":
                    function_call_id = content.call_id
                    break
        
        if function_call_id is None:
            raise RuntimeError("Could not find request_user_input function call in pending response")
        
        # Create FunctionResultContent with user's response
        function_result = FunctionResultContent(
            call_id=function_call_id,
            result=user_answer
        )
        
        # Add function result to conversation
        full_conversation = list(self._pending_response.full_conversation)
        full_conversation.append(ChatMessage(role=Role.TOOL, contents=[function_result]))
        
        # Clear pending response
        self._pending_response = None
        
        # Continue to next agent with updated conversation
        await ctx.send_message(
            AgentExecutorResponse(
                executor_id=self.id,
                agent_run_response=self._pending_response.agent_run_response,
                full_conversation=full_conversation
            ),
            target_id=self._next_executor_id
        )
    
    def _extract_user_input_request(
        self, 
        messages: list[ChatMessage]
    ) -> UserElicitationRequest | None:
        """
        Check messages for request_user_input tool call.
        
        Returns UserElicitationRequest if found, None otherwise.
        """
        for message in messages:
            for content in message.contents:
                if isinstance(content, FunctionCallContent) and content.name == "request_user_input":
                    # Parse tool arguments
                    args = content.arguments
                    if isinstance(args, str):
                        args_dict = json.loads(args)
                    elif isinstance(args, dict):
                        args_dict = args
                    else:
                        continue
                    
                    # Create UserElicitationRequest
                    return UserElicitationRequest(
                        prompt=args_dict.get("prompt", ""),
                        context=args_dict.get("context", {}),
                        request_type=args_dict.get("request_type", "general_clarification"),
                        agent_name=self._agent_name
                    )
        
        return None


__all__ = ["HumanInLoopAgentExecutor"]
```

Verify executor class:

    uv run python -c "from spec2agent.workflow.executors import HumanInLoopAgentExecutor; print(HumanInLoopAgentExecutor.__name__)"

Expected output: `HumanInLoopAgentExecutor`

### Step 4: Update Workflow Builder with HITL Integration

Modify `src/spec2agent/workflow/core.py` to add HITL-enabled workflow builder:

Add new imports at top of file:

```python
from agent_framework import RequestInfoExecutor, WorkflowAgent
from spec2agent.tools import REQUEST_USER_INPUT_TOOL
from spec2agent.workflow.executors import HumanInLoopAgentExecutor
```

Add new function after existing `build_event_planning_workflow()`:

```python
def build_event_planning_workflow_with_hitl() -> Workflow:
    """
    Build event planning workflow with human-in-the-loop capabilities.
    
    This workflow wraps specialist agents with HumanInLoopAgentExecutor to
    enable user interaction when agents call the request_user_input tool.
    
    Returns
    -------
    Workflow
        Workflow instance with HITL integration
    """
    client = get_chat_client()
    
    # Create agents with request_user_input tool
    coordinator = client.create_agent(
        name="EventCoordinator",
        instructions=event_coordinator.SYSTEM_PROMPT,
        store=True
    )
    
    venue_agent = client.create_agent(
        name="VenueSpecialist",
        instructions=venue_specialist.SYSTEM_PROMPT,
        tools=[REQUEST_USER_INPUT_TOOL],
        store=True
    )
    
    budget_agent = client.create_agent(
        name="BudgetAnalyst",
        instructions=budget_analyst.SYSTEM_PROMPT,
        tools=[REQUEST_USER_INPUT_TOOL],
        store=True
    )
    
    catering_agent = client.create_agent(
        name="CateringCoordinator",
        instructions=catering_coordinator.SYSTEM_PROMPT,
        tools=[REQUEST_USER_INPUT_TOOL],
        store=True
    )
    
    logistics_agent = client.create_agent(
        name="LogisticsManager",
        instructions=logistics_manager.SYSTEM_PROMPT,
        tools=[REQUEST_USER_INPUT_TOOL],
        store=True
    )
    
    # Create RequestInfoExecutor
    request_info = RequestInfoExecutor(id="user_input")
    
    # Create AgentExecutors
    coordinator_exec = AgentExecutor(agent=coordinator, id="coordinator")
    
    # Wrap specialists with HumanInLoopAgentExecutor
    venue_exec = HumanInLoopAgentExecutor(
        agent_executor=AgentExecutor(agent=venue_agent, id="venue"),
        request_info_id=request_info.id,
        next_executor_id="budget",
        agent_name="Venue Specialist"
    )
    
    budget_exec = HumanInLoopAgentExecutor(
        agent_executor=AgentExecutor(agent=budget_agent, id="budget"),
        request_info_id=request_info.id,
        next_executor_id="catering",
        agent_name="Budget Analyst"
    )
    
    catering_exec = HumanInLoopAgentExecutor(
        agent_executor=AgentExecutor(agent=catering_agent, id="catering"),
        request_info_id=request_info.id,
        next_executor_id="logistics",
        agent_name="Catering Coordinator"
    )
    
    logistics_exec = HumanInLoopAgentExecutor(
        agent_executor=AgentExecutor(agent=logistics_agent, id="logistics"),
        request_info_id=request_info.id,
        next_executor_id="coordinator",
        agent_name="Logistics Manager"
    )
    
    # Build workflow with HITL edges
    workflow_instance = (
        WorkflowBuilder()
        .set_start_executor(coordinator_exec)
        .add_edge(coordinator_exec, venue_exec)
        # Venue HITL edges
        .add_edge(venue_exec, request_info)
        .add_edge(request_info, venue_exec)
        # Venue to Budget
        .add_edge(venue_exec, budget_exec)
        # Budget HITL edges
        .add_edge(budget_exec, request_info)
        .add_edge(request_info, budget_exec)
        # Budget to Catering
        .add_edge(budget_exec, catering_exec)
        # Catering HITL edges
        .add_edge(catering_exec, request_info)
        .add_edge(request_info, catering_exec)
        # Catering to Logistics
        .add_edge(catering_exec, logistics_exec)
        # Logistics HITL edges
        .add_edge(logistics_exec, request_info)
        .add_edge(request_info, logistics_exec)
        # Logistics to Coordinator (synthesis)
        .add_edge(logistics_exec, coordinator_exec)
        .build()
    )
    
    return workflow_instance


def build_event_planning_workflow_as_agent() -> WorkflowAgent:
    """
    Build event planning workflow and expose as a WorkflowAgent.
    
    This enables the workflow to be used as a building block in larger
    multi-agent systems and provides tool-based human-in-the-loop interaction.
    
    Returns
    -------
    WorkflowAgent
        Workflow agent that can be composed with other agents and handles
        user interaction through request_user_input tool calls.
    """
    workflow_instance = build_event_planning_workflow_with_hitl()
    return workflow_instance.as_agent()
```

Update module exports at bottom of file:

```python
# Export both standard workflow and workflow-as-agent
workflow = build_event_planning_workflow()
workflow_agent = build_event_planning_workflow_as_agent()

__all__ = [
    "build_event_planning_workflow",
    "build_event_planning_workflow_with_hitl",
    "build_event_planning_workflow_as_agent",
    "workflow",
    "workflow_agent",
]
```

### Step 5: Update Agent Prompts with User Elicitation Guidance

Update each specialist prompt file to add user interaction guidelines.

For `src/spec2agent/prompts/venue_specialist.py`, add to SYSTEM_PROMPT (before final closing quotes):

```python
## User Interaction Guidelines

You have access to a `request_user_input` tool for requesting user clarification or selection.

**When to request user input**:
- When you've identified multiple suitable venues and need user preference
- When venue requirements are unclear (capacity range, location specifics, etc.)
- When budget constraints require venue compromises
- When accessibility or amenity requirements are ambiguous

**How to use request_user_input**:
Call the tool with clear prompt and relevant context:
```
request_user_input(
    prompt="I've found 3 venues matching your requirements. Which do you prefer?",
    context={"venues": [...]},
    request_type="venue_selection"
)
```

After receiving user response, acknowledge and incorporate their selection into your recommendations.

**Important**: Only request input when truly necessary. Make reasonable assumptions when requirements are clear.
```

Apply similar additions to:
- `src/spec2agent/prompts/budget_analyst.py` (request_type: "budget_approval")
- `src/spec2agent/prompts/catering_coordinator.py` (request_type: "catering_approval")
- `src/spec2agent/prompts/logistics_manager.py` (request_type: "logistics_confirmation")

### Step 6: Update Module Exports

Update `src/spec2agent/agents/__init__.py`:

```python
from agent_framework import ChatAgent, Workflow, WorkflowAgent

from spec2agent.agents.budget_analyst import agent as budget_analyst_agent
from spec2agent.agents.catering_coordinator import agent as catering_coordinator_agent
from spec2agent.agents.event_coordinator import agent as event_coordinator_agent
from spec2agent.agents.logistics_manager import agent as logistics_manager_agent
from spec2agent.agents.venue_specialist import agent as venue_specialist_agent
from spec2agent.workflow.core import (
    workflow as event_planning_workflow,
    workflow_agent as event_planning_workflow_agent,
)


def export_entities() -> list[Workflow | ChatAgent | WorkflowAgent]:
    """Export all agents/workflows for registration in DevUI."""
    return [
        budget_analyst_agent,
        catering_coordinator_agent,
        event_coordinator_agent,
        logistics_manager_agent,
        venue_specialist_agent,
        event_planning_workflow,
        event_planning_workflow_agent,  # NEW: workflow-as-agent
    ]


__all__ = ["export_entities"]
```

### Step 7: Create Tests

**7.1: Create Message Type Tests**

Create `tests/test_workflow_messages.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

"""Tests for workflow message types."""

from spec2agent.workflow.messages import UserElicitationRequest


def test_user_elicitation_request_construction():
    """Test UserElicitationRequest can be constructed with required fields."""
    request = UserElicitationRequest(
        prompt="Select a venue",
        context={"venues": ["A", "B", "C"]},
        request_type="venue_selection",
        agent_name="Venue Specialist"
    )
    
    assert request.prompt == "Select a venue"
    assert request.context == {"venues": ["A", "B", "C"]}
    assert request.request_type == "venue_selection"
    assert request.agent_name == "Venue Specialist"


def test_user_elicitation_request_empty_context():
    """Test UserElicitationRequest works with empty context."""
    request = UserElicitationRequest(
        prompt="Provide clarification",
        context={},
        request_type="general_clarification",
        agent_name="Budget Analyst"
    )
    
    assert request.context == {}
    assert request.request_type == "general_clarification"
```

Run test:

    uv run pytest tests/test_workflow_messages.py -v

Expected: All tests pass.

**7.2: Create Executor Tests**

Create `tests/test_workflow_executors.py`:

```python
# Copyright (c) Microsoft. All rights reserved.

"""Tests for custom workflow executors."""

import pytest
from unittest.mock import AsyncMock, Mock

from agent_framework import (
    AgentExecutor,
    AgentExecutorResponse,
    AgentRunResponse,
    ChatMessage,
    FunctionCallContent,
    Role,
)

from spec2agent.workflow.executors import HumanInLoopAgentExecutor
from spec2agent.workflow.messages import UserElicitationRequest


def test_human_in_loop_executor_construction():
    """Test HumanInLoopAgentExecutor can be constructed."""
    mock_agent = Mock()
    agent_executor = AgentExecutor(agent=mock_agent, id="test_agent")
    
    executor = HumanInLoopAgentExecutor(
        agent_executor=agent_executor,
        request_info_id="request_info",
        next_executor_id="next_agent",
        agent_name="Test Agent"
    )
    
    assert executor.id == "test_agent"
    assert executor._request_info_id == "request_info"
    assert executor._next_executor_id == "next_agent"
    assert executor._agent_name == "Test Agent"


def test_extract_user_input_request_found():
    """Test _extract_user_input_request detects request_user_input tool call."""
    mock_agent = Mock()
    agent_executor = AgentExecutor(agent=mock_agent, id="test_agent")
    
    executor = HumanInLoopAgentExecutor(
        agent_executor=agent_executor,
        request_info_id="request_info",
        next_executor_id="next_agent",
        agent_name="Test Agent"
    )
    
    # Create message with request_user_input tool call
    function_call = FunctionCallContent(
        call_id="call_123",
        name="request_user_input",
        arguments={"prompt": "Select venue", "request_type": "venue_selection", "context": {}}
    )
    message = ChatMessage(role=Role.ASSISTANT, contents=[function_call])
    
    result = executor._extract_user_input_request([message])
    
    assert result is not None
    assert isinstance(result, UserElicitationRequest)
    assert result.prompt == "Select venue"
    assert result.request_type == "venue_selection"
    assert result.agent_name == "Test Agent"


def test_extract_user_input_request_not_found():
    """Test _extract_user_input_request returns None when no tool call."""
    mock_agent = Mock()
    agent_executor = AgentExecutor(agent=mock_agent, id="test_agent")
    
    executor = HumanInLoopAgentExecutor(
        agent_executor=agent_executor,
        request_info_id="request_info",
        next_executor_id="next_agent",
        agent_name="Test Agent"
    )
    
    # Create message without request_user_input tool call
    message = ChatMessage(role=Role.ASSISTANT, text="Here are my recommendations")
    
    result = executor._extract_user_input_request([message])
    
    assert result is None
```

Run test:

    uv run pytest tests/test_workflow_executors.py -v

Expected: All tests pass.

**7.3: Add Integration Tests**

Add to `tests/test_workflow_integration.py`:

```python
@pytest.mark.asyncio
async def test_workflow_as_agent_construction():
    """Test workflow-as-agent can be constructed."""
    from spec2agent.workflow.core import build_event_planning_workflow_as_agent
    
    workflow_agent = build_event_planning_workflow_as_agent()
    
    assert workflow_agent is not None
    assert isinstance(workflow_agent, WorkflowAgent)


@pytest.mark.asyncio
async def test_workflow_as_agent_responds_to_query():
    """Test workflow-as-agent can respond to user queries."""
    from spec2agent.workflow.core import build_event_planning_workflow_as_agent
    
    workflow_agent = build_event_planning_workflow_as_agent()
    
    response = await workflow_agent.run("Plan a small team event for 15 people")
    
    assert response is not None
    assert len(response.messages) > 0
```

Run tests:

    uv run pytest tests/test_workflow_integration.py -v

Expected: All tests pass (note: requires Azure credentials).

### Step 8: Validate in DevUI

Start DevUI:

    uv run app

Expected output:
```
Starting Agent Framework DevUI...
Server running at http://localhost:8000
```

Navigate to `http://localhost:8000`.

**Test Scenario 1: Ambiguous Request Triggering HITL**

Submit to workflow agent:
```
Plan an event for 40 people
```

Expected behavior:
1. EventCoordinator delegates to VenueSpecialist
2. VenueSpecialist may call `request_user_input` for clarification
3. DevUI displays user prompt
4. Provide response (e.g., "Corporate team building, Seattle, casual atmosphere")
5. Workflow resumes with user's input
6. Subsequent agents continue planning
7. Final integrated plan returned

**Test Scenario 2: Detailed Request Without HITL**

Submit to workflow agent:
```
Plan a corporate team building event for 40 people in Seattle with $4000 budget, 
vegetarian and gluten-free options required, casual atmosphere, Friday evening 
3 weeks from now.
```

Expected behavior:
1. Workflow executes all agents sequentially
2. No `request_user_input` tool calls (sufficient context)
3. Final plan returned without user interaction

**Test Scenario 3: Multiple HITL Interactions**

Submit to workflow agent:
```
Plan a celebration event
```

Expected behavior:
1. VenueSpecialist requests clarification on event type, location, capacity
2. User provides details
3. BudgetAnalyst may request budget approval/modification
4. User approves or modifies
5. CateringCoordinator may request dietary preferences
6. User provides preferences
7. Final plan incorporates all user decisions

## Validation and Acceptance

### Acceptance Criteria

The workflow-as-agent with HITL implementation is complete when:

1. **Tool Definition**: `request_user_input` tool is defined in `tools/user_input.py`
2. **Message Types**: `UserElicitationRequest` is defined in `workflow/messages.py`
3. **Custom Executor**: `HumanInLoopAgentExecutor` is implemented in `workflow/executors.py`
4. **Workflow Builder**: HITL-enabled workflow builder creates proper executor graph with bidirectional edges
5. **Workflow-as-Agent**: `build_event_planning_workflow_as_agent()` returns `WorkflowAgent` instance
6. **Prompts Updated**: All specialist prompts include user elicitation guidance
7. **Module Exports**: `workflow_agent` is exported and discoverable in DevUI
8. **Unit Tests**: Message and executor tests pass
9. **Integration Tests**: Workflow-as-agent tests pass
10. **DevUI Validation**: Both HITL and non-HITL scenarios work correctly

### Observable Behaviors

**Unit Test Output**:
```
$ uv run pytest tests/test_workflow_messages.py tests/test_workflow_executors.py -v

tests/test_workflow_messages.py::test_user_elicitation_request_construction PASSED
tests/test_workflow_messages.py::test_user_elicitation_request_empty_context PASSED
tests/test_workflow_executors.py::test_human_in_loop_executor_construction PASSED
tests/test_workflow_executors.py::test_extract_user_input_request_found PASSED
tests/test_workflow_executors.py::test_extract_user_input_request_not_found PASSED

========== 5 passed in 0.45s ==========
```

**Integration Test Output**:
```
$ uv run pytest tests/test_workflow_integration.py::test_workflow_as_agent_construction -v

tests/test_workflow_integration.py::test_workflow_as_agent_construction PASSED

========== 1 passed in 2.34s ==========
```

**DevUI Discovery**:
- Both "Event Planning Workflow" and "Event Planning Workflow Agent" appear in DevUI
- Can interact with either variant
- Workflow agent shows tool-based interaction pattern

**HITL Flow in DevUI**:
1. User submits ambiguous request
2. Specialist agent calls `request_user_input` tool
3. DevUI displays: "Venue Specialist is requesting input: [prompt text]"
4. User provides response via text input
5. DevUI shows: "Resuming workflow with user response..."
6. Workflow continues to next agent
7. Final plan incorporates user decisions

## Idempotence and Recovery

### Safe Execution

- Workflow construction is stateless and repeatable
- Custom executors maintain minimal state (only pending response during HITL)
- Tests can be run multiple times without side effects
- DevUI can be restarted without data loss

### Error Recovery

**If tool call detection fails**:
- Check `_extract_user_input_request` logic in `HumanInLoopAgentExecutor`
- Verify tool call format matches `FunctionCallContent` structure
- Add debug logging to print agent response messages

**If RequestInfoExecutor doesn't receive messages**:
- Verify bidirectional edges between specialists and `request_info`
- Check executor IDs match edge definitions
- Use workflow graph visualization in DevUI to verify edges

**If workflow hangs during HITL**:
- Check that `RequestResponse` handler in custom executor is implemented
- Verify user response is sent back to workflow correctly
- Check DevUI console for error messages

**If tests fail**:
- Ensure Azure credentials are configured: `az login`
- Verify all imports are correct
- Run with increased verbosity: `uv run pytest -vv`
- Check that agents have `request_user_input` tool in their tool list

### Rollback Strategy

To revert changes:
- Delete `src/spec2agent/tools/user_input.py`
- Delete `src/spec2agent/workflow/messages.py`
- Delete `src/spec2agent/workflow/executors.py`
- Remove HITL-related code from `src/spec2agent/workflow/core.py`
- Revert prompt changes
- Remove `workflow_agent` from module exports
- Delete new test files

## Artifacts and Notes

### Key Design Decisions

**Workflow-as-Agent vs Standard Workflow**:
- **Standard Workflow**: Direct execution via `workflow.run()`, emits workflow events
- **Workflow-as-Agent**: Tool-based interaction via `.as_agent()`, emits function calls for human review
- **Use Cases**: Standard for standalone execution, agent for composition and integration

**Tool-Based Detection vs Prompt-Based**:
- **Tool-Based**: Explicit `request_user_input` tool call provides deterministic detection
- **Prompt-Based**: Agents instructed to request input, but no programmatic signal
- **Choice**: Tool-based for reliability and clear agent interface

**Custom Executor Wrapper vs Modifying AgentExecutor**:
- **Wrapper**: Preserves original `AgentExecutor` behavior, adds HITL logic separately
- **Modification**: Would require forking framework code
- **Choice**: Wrapper for maintainability and framework compatibility

**Single Message Type vs Multiple**:
- **Single**: `UserElicitationRequest` with flexible `context` dict
- **Multiple**: Specialized types for each request category
- **Choice**: Single for simplicity, can add specialized types later if needed

### Framework Patterns Used

**Workflow-as-Agent Pattern**:
```python
workflow = WorkflowBuilder()...build()
agent = workflow.as_agent()  # Returns WorkflowAgent

# WorkflowAgent emits FunctionCallContent for human review
response = await agent.run("User query")
for message in response.messages:
    for content in message.contents:
        if isinstance(content, FunctionCallContent):
            if content.name == WorkflowAgent.REQUEST_INFO_FUNCTION_NAME:
                # Handle human review request
```

**Custom Executor Pattern**:
```python
class CustomExecutor(Executor):
    @handler
    async def from_response(self, prior: AgentExecutorResponse, ctx):
        # Custom logic before/after agent execution
        
    @handler  
    async def from_user_response(self, response: RequestResponse, ctx):
        # Handle user responses
```

**RequestInfoExecutor Integration**:
```python
request_info = RequestInfoExecutor(id="user_input")

workflow = (
    WorkflowBuilder()
    .add_edge(specialist, request_info)      # Specialist can request
    .add_edge(request_info, specialist)      # Response flows back
    .build()
)
```

### Comparison with workflow-skeleton.md

| Aspect | workflow-skeleton.md | This Spec (workflow-agent-skeleton.md) |
|--------|---------------------|----------------------------------------|
| **Execution** | Direct `workflow.run()` | Via `.as_agent()` tool interface |
| **HITL** | No human interaction | Tool-based HITL with RequestInfoExecutor |
| **Composition** | Standalone workflow | Composable agent for larger systems |
| **Executors** | Standard AgentExecutor | Custom HumanInLoopAgentExecutor wrappers |
| **User Interaction** | None | request_user_input tool + UserElicitationRequest |
| **Use Case** | End-to-end event planning | Hierarchical multi-agent systems |

### Research Sources

1. **Agent Framework Sample**: `workflow_as_agent_human_in_the_loop.py`
   - `ReviewerWithHumanInTheLoop` custom executor pattern
   - `HumanReviewRequest` message type definition
   - Bidirectional edges with `RequestInfoExecutor`
   - `.as_agent()` usage for workflow composition

2. **DeepWiki Documentation**:
   - WorkflowAgent interface and capabilities
   - RequestInfoExecutor mechanics
   - FunctionCallContent structure for tool calls
   - RequestResponse handling patterns

3. **Agent Framework Source Code**:
   - `agent_framework/_workflows/_workflow.py` - `as_agent()` implementation
   - `agent_framework/_workflows/_request_info.py` - RequestInfoExecutor
   - `agent_framework/_agents/_workflow_agent.py` - WorkflowAgent class

### Testing Strategy

**Unit Tests** (Fast, No API Calls):
- Message type construction and validation
- Tool definition structure
- Executor construction and configuration
- Tool call detection logic (`_extract_user_input_request`)

**Integration Tests** (Requires Azure):
- Workflow-as-agent construction
- End-to-end query execution
- HITL flow with mocked user responses
- Multi-agent conversation with tool calls

**Manual DevUI Tests** (Human Validation):
- Ambiguous requests trigger HITL
- Detailed requests bypass HITL
- Multiple HITL interactions in one run
- User response handling and workflow resumption

### Future Enhancements

1. **Specialized Message Types**: Add `VenueSelectionRequest`, `BudgetApprovalRequest` for stronger typing
2. **Structured Context**: Define Pydantic models for context field in UserElicitationRequest
3. **HITL Analytics**: Track frequency and types of user interactions
4. **Timeout Handling**: Add timeout for user responses with fallback behavior
5. **HITL History**: Store and learn from past user decisions
6. **Conditional HITL**: Configuration to enable/disable HITL per agent
7. **Multi-Step HITL**: Support for iterative refinement conversations with user

## Interfaces and Dependencies

### Required Imports

New imports for `src/spec2agent/workflow/core.py`:

```python
from agent_framework import (
    WorkflowBuilder,
    Workflow,
    WorkflowAgent,
    AgentExecutor,
    RequestInfoExecutor,
)
from spec2agent.clients import get_chat_client
from spec2agent.prompts import (
    event_coordinator,
    venue_specialist,
    budget_analyst,
    catering_coordinator,
    logistics_manager,
)
from spec2agent.tools import REQUEST_USER_INPUT_TOOL
from spec2agent.workflow.executors import HumanInLoopAgentExecutor
```

### Function Signatures

```python
def build_event_planning_workflow_with_hitl() -> Workflow:
    """Build event planning workflow with HITL integration."""
    ...

def build_event_planning_workflow_as_agent() -> WorkflowAgent:
    """Build event planning workflow and expose as WorkflowAgent."""
    ...
```

### Module Exports

`src/spec2agent/workflow/core.py`:
```python
__all__ = [
    "build_event_planning_workflow",
    "build_event_planning_workflow_with_hitl",
    "build_event_planning_workflow_as_agent",
    "workflow",
    "workflow_agent",
]
```

`src/spec2agent/tools/__init__.py`:
```python
__all__ = ["REQUEST_USER_INPUT_TOOL"]
```

`src/spec2agent/workflow/messages.py`:
```python
__all__ = ["UserElicitationRequest"]
```

`src/spec2agent/workflow/executors.py`:
```python
__all__ = ["HumanInLoopAgentExecutor"]
```

### Dependencies

All required packages are already in `pyproject.toml`:
- `agent-framework-core`: Workflow, WorkflowAgent, Executor, RequestInfoExecutor
- `agent-framework-azure-ai`: AzureAIAgentClient
- `agent-framework-devui`: DevUI for testing
- `azure-identity`: Authentication
- `pydantic`: Data validation

---

**Specification Version**: 1.0
**Created**: 2025-10-28
**Last Updated**: 2025-10-28
**Status**: Ready for Implementation

---

## Specification Change Log

### Initial Creation - 2025-10-28

**Reason**: Demonstrate workflow-as-agent pattern with human-in-the-loop capabilities using tool-based detection.

**Design Approach**:
- Separate spec from `workflow-skeleton.md` to show both patterns side-by-side
- Tool-based HITL detection using `request_user_input` tool
- Custom `HumanInLoopAgentExecutor` wrappers to intercept tool calls
- Single `UserElicitationRequest` message type for simplicity
- Integration with `RequestInfoExecutor` via bidirectional edges
- Workflow exposed as `WorkflowAgent` via `.as_agent()`

**Research Conducted**:
- Analyzed `workflow_as_agent_human_in_the_loop.py` sample for patterns
- Studied `RequestInfoExecutor` and `RequestResponse` mechanisms
- Examined `FunctionCallContent` structure for tool call detection
- Reviewed `WorkflowAgent` interface and capabilities
- Validated DevUI's automatic handling of `RequestInfoEvent`s

**Expected Impact**:
- Enables hierarchical multi-agent system composition
- Provides flexible, agent-driven user interaction
- Demonstrates advanced Agent Framework patterns
- Educational value showing two workflow paradigms

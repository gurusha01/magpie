# Multi-Agent Negotiation System (MPI)

<p align="center">
  <a href="https://arxiv.org/abs/2510.15186" target="_blank" style="
    display:inline-block;
    background:#0E121B;
    color:white;
    font-weight:normal;
    text-decoration:none;
    border:2px solid white;
    border-radius:8px;
    padding:8px 16px;
    margin:4px;
    font-family:Arial, sans-serif;
  ">Paper</a>
  <a href="https://huggingface.co/datasets/jaypasnagasai/magpie" target="_blank" style="
    display:inline-block;
    background:#0E121B;
    color:white;
    font-weight:normal;
    text-decoration:none;
    border:2px solid white;
    border-radius:8px;
    padding:8px 16px;
    margin:4px;
    font-family:Arial, sans-serif;
  ">Dataset</a>
  <a href="https://jaypasnagasai.github.io/magpie/" target="_blank" style="
    display:inline-block;
    background:#0E121B;
    color:white;
    font-weight:normal;
    text-decoration:none;
    border:2px solid white;
    border-radius:8px;
    padding:8px 16px;
    margin:4px;
    font-family:Arial, sans-serif;
  ">Website</a>
</p>

A comprehensive system for generating, simulating, and analyzing multi-agent negotiation scenarios with conflicting preferences and private information.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Setup Guide](#setup-guide)
4. [Data Generation](#data-generation)
5. [Simulation System](#simulation-system)
6. [Batch Processing](#batch-processing)
7. [File Structure](#file-structure)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Overview

This system provides a complete pipeline for multi-agent negotiation research:

- **Data Generation**: Creates realistic negotiation scenarios with conflicting preferences
- **Agent Simulation**: Runs multi-agent negotiations with LLM-powered decision making
- **Batch Processing**: Handles large-scale experiments with parallel execution
- **Memory System**: Agents maintain strategic memory across negotiation rounds
- **Comprehensive Logging**: Detailed logs for analysis and debugging

### Key Features

- **LLM-Powered Generation**: Uses OpenAI/Gemini for dynamic scenario creation
- **Conflicting Preferences**: Agents have opposing interests requiring genuine negotiation
- **Private Information**: Each agent has private information they cannot share
- **Verifiable Constraints**: Tasks include checkable constraints (budget limits, resource constraints)
- **Memory System**: Agents maintain strategic memory across rounds
- **Selective Communication**: Agents can send messages to specific other agents
- **Consensus Mechanism**: Strict consensus requirements for simulation completion

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate to the repository
cd mpi

# Copy environment template
cp env_template.txt .env

# Edit .env with your API keys
nano .env
```

### 2. Install Dependencies

```bash
pip install openai google-generativeai
```

### 3. Generate a Scenario

```bash
python generate_datapoint.py
```

### 4. Run a Simulation

```bash
python simulate_agents.py --scenario_file data2/budget_allocation_3agents.json --llm gemini
```

### 5. Run Batch Simulations

```bash
chmod +x run_sims.sh
./run_sims.sh
```

## Setup Guide

### API Key Configuration

The system supports both OpenAI and Google Gemini APIs with automatic key rotation.

#### 1. Create Environment File

```bash
cp env_template.txt .env
```

#### 2. Configure API Keys

Edit `.env` file with your API keys:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your_openai_key_here

# Google Gemini API Keys (multiple keys for load balancing)
GEMINI_API_KEY=your_primary_gemini_key_here
GEMINI_API_KEY_1=your_gemini_key_1_here
GEMINI_API_KEY_2=your_gemini_key_2_here
GEMINI_API_KEY_3=your_gemini_key_3_here
GEMINI_API_KEY_4=your_gemini_key_4_here
GEMINI_API_KEY_5=your_gemini_key_5_here
```

#### 3. Get API Keys

**OpenAI API Key:**
- Visit: https://platform.openai.com/api-keys
- Create a new API key (starts with `sk-`)

**Google Gemini API Keys:**
- Visit: https://aistudio.google.com/app/apikey
- Create new API keys (start with `AIzaSy`)

### Security Notes

- Never commit `.env` files (already in `.gitignore`)
- Keep API keys secure and don't share publicly
- Rotate keys regularly
- Monitor usage in provider dashboards

## Data Generation

### Overview

The data generation system creates realistic negotiation scenarios with:
- Conflicting agent preferences
- Private information with socially acceptable reasons
- Verifiable constraints and success criteria
- LLM-based verification of solvability

### Usage

#### Interactive Generation

```bash
python generate_datapoint.py
```

#### Programmatic Generation

```python
from generate_datapoint import generate_scenario, save_scenario, verify_solvability

# Generate a budget allocation scenario with 4 agents
scenario = generate_scenario('budget_allocation', 4)

# Verify the scenario is solvable
if verify_solvability(scenario):
    save_scenario(scenario, 'my_scenario.json')
    print("Scenario generated successfully!")
```

#### One-Line Generation

```python
from generate_datapoint import generate_and_save_scenario

# Generate and save in one step
scenario = generate_and_save_scenario('hiring_decision', 3, 'hiring_scenario.json')
```

### Available Scenario Types

1. `budget_allocation` - Department budget allocation with competing priorities
2. `hiring_decision` - Critical position hiring with multiple stakeholders
3. `resource_allocation` - Resource distribution with conflicting needs
4. `project_planning` - Project timeline negotiation
5. `gift_selection` - Gift selection with different preferences
6. `event_planning` - Event coordination with competing interests
7. `team_formation` - Team assembly with conflicting requirements
8. `salary_negotiation` - Salary negotiation with multiple parties

### Verification Criteria

The generator ensures scenarios meet 5 critical criteria:

1. **Has Conflicts**: Genuine conflicts between agents requiring negotiation
2. **Private Info Justified**: Private preferences have socially acceptable reasons
3. **Is Solvable**: Scenario is solvable when all private information is revealed
4. **Constraints Realistic**: Constraints are verifiable and realistic
5. **Requires Negotiation**: Success criteria require genuine agreement/negotiation

## Simulation System

### Overview

The simulation system allows multiple AI agents to engage in realistic negotiation scenarios where they must reach consensus on a common proposal.

### Key Features

#### Agent Capabilities
- **Selective Communication**: Send messages and proposals to specific agents
- **Memory System**: Maintain strategic memory of important observations
- **Proposal Management**: Create, accept, or reject proposals with reasoning
- **LLM Integration**: Support for OpenAI and Google Gemini APIs

#### Simulation Flow
1. **Initialization**: Load scenario data and create agents with preferences
2. **Round-based Interaction**: Agents observe environment, update memory, decide actions
3. **Consensus Check**: Simulation ends when ALL agents accept the SAME proposal
4. **Logging**: Complete conversation and agent state logging

### Usage

#### Command Line Interface

```bash
python simulate_agents.py <scenario_file> [options]
```

**Arguments:**
- `scenario_file`: Path to the scenario JSON file

**Options:**
- `--llm {openai,gemini}`: Choose LLM provider (default: gemini)
- `--api-key KEY`: API key for the LLM (or set environment variables)
- `--max-rounds N`: Maximum number of negotiation rounds (default: 50)
- `--output FILE`: Custom output file path

#### Example Usage

```bash
# Using Gemini with environment variable
python simulate_agents.py data2/budget_allocation_4agents.json --llm gemini

# Using OpenAI with explicit API key
python simulate_agents.py data2/hiring_decision_3agents.json --llm openai --api-key "your-openai-key"

# Custom output location
python simulate_agents.py data2/gift_selection_5agents.json --output my_simulation.json
```

### Agent Class

#### Core Methods

**Communication:**
- `send_message(agent_list, message, conversation_log)`: Send message to specific agents
- `send_proposal(agent_list, proposal, conversation_log)`: Send proposal to specific agents
- `accept_proposal(proposal_id, reason, conversation_log)`: Accept a proposal with reasoning
- `reject_proposal(proposal_id, reason, conversation_log)`: Reject a proposal with reasoning

**Memory and Decision Making:**
- `write_to_memory(text)`: Store important observations in agent memory
- `observe_environment(conversation_log, other_agents)`: Analyze recent events and changes
- `decide_action(conversation_log, other_agents, task_info)`: Use LLM to decide next action

#### Agent State
- `name`: Agent identifier
- `role`: Agent's role in the scenario
- `description`: Agent background and relevance
- `shareable_preferences`: Preferences the agent can discuss openly
- `private_preferences`: Sensitive preferences with privacy justifications
- `memory`: List of timestamped observations
- `proposal_status`: Current proposal state ("none", "pending", "accepted", "rejected")
- `current_proposal`: ID of the proposal the agent is considering

### Consensus Mechanism

The simulation implements strict consensus requirements:

1. **Same Proposal**: All agents must accept the exact same proposal ID
2. **Complete Agreement**: Every agent must have "accepted" status
3. **No Partial Consensus**: Having some agents agree to one proposal and others to a different proposal does not end the simulation

### Memory System

Agents use their memory to:
- Track important events and changes
- Remember other agents' preferences and behaviors
- Build context for future decisions
- Avoid repeating failed negotiation strategies

Memory entries are timestamped and contain the agent's analysis of environmental changes since their last observation.

### Logging Format

The simulation creates detailed JSON logs containing:

```json
{
  "scenario_file": "path/to/scenario.json",
  "llm_type": "gemini",
  "timestamp": "2024-01-01T12:00:00",
  "scenario_data": { /* Original scenario data */ },
  "agents": [
    {
      "name": "Agent Name",
      "role": "Agent Role", 
      "description": "Agent Description",
      "memory": [ /* Agent's memory entries */ ],
      "final_proposal_status": "accepted",
      "current_proposal": "proposal_id"
    }
  ],
  "conversation_log": [ /* All conversation events */ ]
}
```

## Batch Processing

### Overview

The batch processing system allows running simulations on all scenarios in parallel with configurable concurrency.

### Usage

```bash
# Make the script executable
chmod +x run_sims.sh

# Run all simulations
./run_sims.sh
```

### Configuration

The script processes all JSON files in the `data2` folder with:
- **Parallel Processing**: 5 simulations simultaneously
- **Batch Processing**: Processes all files in batches of 5
- **Progress Tracking**: Shows which batch is being processed
- **Error Handling**: Tracks success/failure of each simulation
- **Logging**: Each simulation gets its own log file

### Output Structure

```
simulations/
├── sim_academic_gemini-2.5-pro.json
├── sim_admissions_gemini-2.5-pro.json
└── ...

logs_*.txt files for detailed output of each simulation
```

## File Structure

```
mpi/
├── .env                           # Your API keys (not committed to git)
├── env_template.txt              # Template for API key setup
├── generate_datapoint.py          # Scenario generator
├── simulate_agents.py             # Agent simulation
├── run_sims.sh                   # Batch processing script
├── data2/                        # Generated scenario files
│   ├── academic.json
│   ├── budget_allocation.json
│   └── ...
├── simulations/                   # Simulation results
│   ├── sim_*.json
│   └── ...
├── logs_*.txt                     # Individual simulation logs
└── README.md                     # This documentation
```

## API Reference

### Data Generation Functions

- `generate_scenario(scenario_type, num_agents)` - Generate a scenario using LLM
- `verify_solvability(scenario)` - Verify scenario using LLM analysis
- `check_all_criteria_passed(verification_result)` - Check if all 5 criteria passed
- `save_scenario(scenario, filename)` - Save scenario to JSON file
- `generate_and_save_scenario(scenario_type, num_agents, filename)` - Generate and save in one step

### Simulation Functions

- `Simulation.load_scenario()` - Load scenario data from JSON file
- `Simulation.initialize_agents()` - Create agents based on scenario data
- `Simulation.run_simulation(max_rounds)` - Main simulation loop
- `Simulation.check_consensus()` - Verify if all agents accepted the same proposal
- `Simulation.save_simulation_log(output_file)` - Save complete simulation data

### Agent Functions

- `Agent.send_message(agent_list, message, conversation_log)` - Send message to specific agents
- `Agent.send_proposal(agent_list, proposal, conversation_log)` - Send proposal to specific agents
- `Agent.accept_proposal(proposal_id, reason, conversation_log)` - Accept a proposal with reasoning
- `Agent.reject_proposal(proposal_id, reason, conversation_log)` - Reject a proposal with reasoning
- `Agent.write_to_memory(text)` - Store important observations in agent memory
- `Agent.observe_environment(conversation_log, other_agents)` - Analyze recent events and changes
- `Agent.decide_action(conversation_log, other_agents, task_info)` - Use LLM to decide next action

## Troubleshooting

### Common Issues

#### "No API keys found" Error
- Ensure your `.env` file exists and contains valid API keys
- Check that the keys are properly formatted (no extra spaces)
- Verify the keys are active and have sufficient quota

#### "API key invalid" Error
- Double-check the API key format
- Ensure the key hasn't expired
- Verify the key has the necessary permissions

#### Rate Limit Errors
- The system automatically retries with different keys
- Consider adding more API keys to the rotation
- Check your API usage limits in the provider dashboards

#### Parse Failures
- Inspect `*.error.json` files for `raw_text` and error details
- Tighten prompts, reduce temperature, add format-strict samples
- Check JSON structure and formatting

#### Unstable/Low Scores
- Keep evaluator at `eval_temperature=0.0` for deterministic results
- Refine rubric text in prompts
- Review generated scenarios for quality

### Dependencies

Required Python packages:
- `openai>=1.0.0` - OpenAI API client
- `google-generativeai` - Google Gemini API client
- Standard library: `json`, `os`, `sys`, `argparse`, `datetime`, `typing`

Install with:
```bash
pip install openai google-generativeai
```

### Error Handling

The system includes robust error handling for:
- LLM API failures (with fallback responses)
- Invalid agent actions (graceful degradation)
- File I/O errors (clear error messages)
- JSON parsing failures (fallback to simple messages)

## License

MIT License

## Acknowledgements

- OpenAI API for chat completion models
- Google Gemini API for generative AI capabilities

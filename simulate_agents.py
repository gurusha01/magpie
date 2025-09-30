# here we load a data point from the data2/ folder, initialize the agents. each agent has a memory that it can use to store conversation histroy that seems appropriate to t. 
# the ganet has the following tools: 
# send_message(agent_list, message) that will send a message to the agents in the list. 
# send_proposal(agent_list, proposal) that will send a proposal for the final task to the list of agents. 
# accept_proposal, reject_proposal that will accept or reject the proposal sent by an agent and wil give a reason for the action. 
# Write_to_memory(text) that will write the text to the memory of the agent. 
# The conversation will end when all the agents have accepted a common proposal. 

# you may create a class called Agent that have the functions defined above. Each agent responds using an API based LLM call. The input to what kind of llm we want to use can be given as a command line argument. 
# We want to maintain a json file for logging the conversations, agent actions, their memories and everything. Use the naming convention: simulations/<data_point_name>_<llm_name>.json
# You cna make a separate readme called README_simulation.md that describes the simulation and logging process in detail.    
# when starting a conversation, every agent should know what the task is, what is the final delieverable, though they should not know the constraints or success criteria. They can know the names of all the agents involved int he conversation as well as a their role in the team and their description. 

# The key distinction here is that this is not a group chat setting, let's say that there are 5 agents ABCDE, not what happens traditionally is that A sends a message that is broadcasted to BCDE but, here we want that if A wants it can send message onbly to C and D, it can also send a proposal to only C and D which they can agree. 
# but the conversation only ends when all the agents ABCDE agree to the SAME proposal. if you maintain a parameter self.proposal status, and set it to agree, then it might be that the conversation ends when ABC agree to one proposal and DE agree to one, which is not what we want.  we want that the conversation ends when all the agents agree to the same proposal. 

# for writing into the memory, the agent does not eed to write the conversation into the memory. After each state change, it can decide, what it wants to write into memory, it observes the nevironment and what had chenged since the last time and what is important to the agent from the change int he environment and then convert that into a string and write it into the memory. 

# ===== DO NOT CHANGE ANYTHING ABOVE THIS LINE =====

import json
import os
import sys
import argparse
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
from google import genai
from google.genai import types

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file."""
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load environment variables
load_env_file()

# Get API keys from environment variables
def get_gemini_api_keys():
    """Get Gemini API keys from environment variables."""
    keys = []
    for i in range(1, 6):  # Check for GEMINI_API_KEY_1 through GEMINI_API_KEY_5
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    
    # Also check for primary GEMINI_API_KEY
    primary_key = os.getenv("GEMINI_API_KEY")
    if primary_key and primary_key not in keys:
        keys.append(primary_key)
    
    return keys

# Get API keys
GEMINI_API_KEYS = get_gemini_api_keys()

def generate_single_gemini(
        prompt: str,
        api_key: str = None,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 0.7
    ) -> str:
        """Generate response using API keys from environment variables."""
        if not GEMINI_API_KEYS:
            raise RuntimeError("No Gemini API keys found in environment variables. Please set GEMINI_API_KEY or GEMINI_API_KEY_1, etc.")
        
        while True:
            try:
                # Use provided API key or choose randomly from available keys
                # if api_key and api_key in GEMINI_API_KEYS:
                #     selected_key = api_key
                # else:
                selected_key = random.choice(GEMINI_API_KEYS)
                
                print(f"Generating response with Gemini model {model_name} using API key {selected_key[:20]}...")
                client = genai.Client(api_key=selected_key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature)
                )
                return response.text
            except Exception as e:
                # print(f"Gemini API error: {e}")
                time.sleep(60)
                continue



class Agent:
    def __init__(self, name: str, role: str, description: str, shareable_preferences: dict, 
                 private_preferences: dict, llm_type: str = "openai", api_key: str = None):
        self.name = name
        self.role = role
        self.description = description
        self.shareable_preferences = shareable_preferences
        self.llm_type = llm_type
        self.memory = []  # Main memory - strategic notes and important observations
        self.temp_memory = []  # Temp memory - current conversation state
        self.current_proposal = None
        self.proposal_status = "none"  # "none", "pending", "accepted", "rejected"
        self.api_key = api_key
        # private_preferences_string = f"The private preferences are: {private_preferences}. Can you remove everything hinting the penalty to disclose this information to other agents, and just retain the description of the private preference and the value of the private preference. give the response in json format."
        # private_pref_response = generate_single_gemini(private_preferences_string)
        # # extract the json between ```json and ``` from the response
        # try:
        #     private_pref_response = private_pref_response.split("```json")[1].split("```")[0]
        #     private_pref_response = json.loads(private_pref_response)
        # except:
        #     private_pref_response = private_preferences

        
        # print(f"The initial private preferences were: {private_preferences} and the final private preferences are: {private_pref_response}")
        # # print(f"The private preferences are: {private_pref_response}")

        self.private_preferences = private_preferences
        
    def write_to_memory(self, text: str):
        """Write text to agent's main memory (strategic notes)."""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "content": text
        })
        # Format: 🧠 Alice writes to main memory: "Bob seems to prefer..."
        print(f"🧠 {self.name} writes to main memory after making it's action: {text}")
    
    def write_to_temp_memory(self, text: str):
        """Write text to agent's temp memory (conversation state)."""
        self.temp_memory.append({
            "timestamp": datetime.now().isoformat(),
            "content": text
        })
        # Format: 📝 Alice writes to temp memory: "Recent events observed..."
        print(f"📝 {self.name} writes to temp memory: {text}")
    
    def send_message(self, agent_list: List[str], message: str, conversation_log: List[Dict]):
        """Send a message to specific agents."""
        # Ensure message is a string
        if message is None:
            message = "No message content"
        elif not isinstance(message, str):
            message = str(message)
        
        message_entry = {
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "to": agent_list,
            "type": "message",
            "content": message
        }
        conversation_log.append(message_entry)
        
        # Format: 📨 Alice → [Bob, Charlie]: "Hello everyone..."
        print(f"📨 {self.name} → {agent_list}: {message}")
        return message_entry
    
    def send_proposal(self, agent_list: List[str], proposal: str, conversation_log: List[Dict]):
        """Send a proposal to specific agents."""
        # Ensure proposal is a string
        if proposal is None:
            proposal = "No proposal content"
        elif not isinstance(proposal, str):
            proposal = str(proposal)
        
        proposal_entry = {
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "to": agent_list,
            "type": "proposal",
            "content": proposal,
            "proposal_id": f"proposal_{len(conversation_log)}_{self.name}"
        }
        conversation_log.append(proposal_entry)
        self.current_proposal = proposal_entry["proposal_id"]
        self.proposal_status = "pending"
        
        # Format: 📋 Alice → [Bob, Charlie]: PROPOSAL - "Let's allocate..."
        print(f"📋 {self.name} → {agent_list}: PROPOSAL - {proposal}")
        print(f"   🆔 Proposal ID: {proposal_entry['proposal_id']}")
        return proposal_entry
    
    def accept_proposal(self, proposal_id: str, reason: str, conversation_log: List[Dict]):
        """Accept a proposal with a reason."""
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "type": "accept_proposal",
            "proposal_id": proposal_id,
            "reason": reason
        }
        conversation_log.append(response_entry)
        self.proposal_status = "accepted"
        
        # Format: ✅ Alice ACCEPTS proposal_123: "This works for me because..."
        print(f"✅ {self.name} ACCEPTS {proposal_id}: {reason}")
        return response_entry
    
    def reject_proposal(self, proposal_id: str, reason: str, conversation_log: List[Dict]):
        """Reject a proposal with a reason."""
        response_entry = {
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "type": "reject_proposal",
            "proposal_id": proposal_id,
            "reason": reason
        }
        conversation_log.append(response_entry)
        self.proposal_status = "rejected"
        
        # Format: ❌ Alice REJECTS proposal_123: "This doesn't work because..."
        print(f"❌ {self.name} REJECTS {proposal_id}: {reason}")
        return response_entry
    
    def get_llm_response(self, prompt: str) -> str:
        """Get response from the configured LLM."""
        try:
            if self.llm_type == "gpt5":
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            elif self.llm_type == "gemini":
                response = generate_single_gemini(prompt)
                return response
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_visible_conversation(self, conversation_log: List[Dict]) -> List[Dict]:
        """Get only the conversation events visible to this agent."""
        visible_events = []
        
        for event in conversation_log:
            # System messages are visible to all agents
            if event.get('from') == 'system':
                visible_events.append(event)
            # Messages from this agent are visible to this agent
            elif event.get('from') == self.name:
                visible_events.append(event)
            # Messages directed to this agent are visible
            elif self.name in event.get('to', []):
                visible_events.append(event)
            # Messages to 'all' or broadcast are visible to all agents
            elif 'all' in event.get('to', []) or len(event.get('to', [])) == 0:
                visible_events.append(event)
        
        return visible_events

    def observe_environment(self, conversation_log: List[Dict], other_agents: List['Agent']) -> str:
        """Observe the environment and write current state to temp memory."""
        # Get only events visible to this agent
        visible_events = self.get_visible_conversation(conversation_log)
        recent_events = visible_events[-10:] if len(visible_events) > 10 else visible_events
        
        # Analyze recent events (including own actions)
        analysis = f"Recent events observed by {self.name}:\n"
        for event in recent_events:
            content = event.get('content', '')
            if content is None:
                content = ''
            elif not isinstance(content, str):
                content = str(content)
            # Show full content instead of truncating
            analysis += f"- {event.get('type', 'unknown')} from {event.get('from', 'unknown')}: {content}\n"
        
        # Check for proposal status changes
        current_proposals = [e for e in recent_events if e.get("type") == "proposal"]
        if current_proposals:
            latest_proposal = current_proposals[-1]
            analysis += f"Latest proposal by {latest_proposal.get('from')}: {latest_proposal.get('content', '')}\n"
        
        # Check other agents' proposal statuses
        for agent in other_agents:
            if agent.name != self.name:
                analysis += f"{agent.name} proposal status: {agent.proposal_status}\n"
        
        # Write to temp memory (conversation state)
        self.write_to_temp_memory(analysis)
        
        return analysis
    
    def decide_memory_action(self, conversation_log: List[Dict], other_agents: List['Agent'], 
                           task_info: Dict) -> Dict:
        """Decide what to write to main memory based on current state."""
        
        # Build context for memory decision
        # Your private preferences (DO NOT SHARE THESE):
        context = f"""
        You are {self.name}, a {self.role} in a negotiation scenario.
        Description: {self.description}
        
        Your shareable preferences:
        {json.dumps(self.shareable_preferences, indent=2)}
        
        Your private preferences (DO NOT SHARE THESE):
        {json.dumps(self.private_preferences, indent=2)}
        
        Current task: {task_info.get('task', 'Unknown')}
        Deliverable: {task_info.get('deliverable', 'Unknown')}
        
        Other agents: {[f"{agent.name} ({agent.role})" for agent in other_agents if agent.name != self.name]}
        
        YOUR CURRENT STRATEGIC NOTES (main memory):
        {json.dumps(self.memory[-5:], indent=2) if self.memory else "No strategic notes yet"}
        
        CURRENT CONVERSATION STATE (temp memory - recent events):
        {json.dumps(self.temp_memory[-3:], indent=2) if self.temp_memory else "No recent observations"}
        
        Recent conversation:
        {json.dumps(self.get_visible_conversation(conversation_log)[-10:], indent=2) if conversation_log else "No conversation yet"}
        
        Your current proposal status: {self.proposal_status}
        Other agents' proposal statuses: {[f"{agent.name}: {agent.proposal_status}" for agent in other_agents if agent.name != self.name]}
        
        Based on the recent conversation and your observations, do you want to write any strategic notes to your main memory? 
        This should be important insights, leverage points, other agents' motivations, strategic observations, etc.
        
        If you want to write to memory, respond with:
        {{
            "action": "write_to_memory",
            "parameters": {{
                "text": "your strategic observation here"
            }}
        }}
        
        If you don't want to write anything to memory, respond with:
        {{
            "action": "no_memory_write",
            "parameters": {{}}
        }}
        """
        
        response = self.get_llm_response(context)
        
        try:
            # Parse JSON response
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            action_data = json.loads(response)
            return action_data
        except:
            # Fallback to no memory write if JSON parsing fails
            return {
                "action": "no_memory_write",
                "parameters": {}
            }
    
    def decide_action(self, conversation_log: List[Dict], other_agents: List['Agent'], 
                     task_info: Dict, max_rounds: int = 50) -> Dict:
        """Decide what action to take based on current state."""
        
        # Build context for LLM
        # Your private preferences (DO NOT SHARE THESE):
        context = f"""
        You are {self.name}, a {self.role} in a negotiation scenario.
        Description: {self.description}
        
        Your shareable preferences:
        {json.dumps(self.shareable_preferences, indent=2)}
        
        Your private preferences (DO NOT SHARE THESE):
        {json.dumps(self.private_preferences, indent=2)}
        
        Current task: {task_info.get('task', 'Unknown')}
        Deliverable: {task_info.get('deliverable', 'Unknown')}
        
        Other agents: {[f"{agent.name} ({agent.role})" for agent in other_agents if agent.name != self.name]}
        
        YOUR NOTES (main memory - strategic observations):
        {json.dumps(self.memory[-5:], indent=2) if self.memory else "No strategic notes yet"}
        
        CURRENT CONVERSATION STATE (temp memory - recent events):
        {json.dumps(self.temp_memory[-3:], indent=2) if self.temp_memory else "No recent observations"}
        
        Recent conversation:
        {json.dumps(self.get_visible_conversation(conversation_log)[-10:], indent=2) if conversation_log else "No conversation yet"}
        
        Your current proposal status: {self.proposal_status}
        Other agents' proposal statuses: {[f"{agent.name}: {agent.proposal_status}" for agent in other_agents if agent.name != self.name]}
        
        Available actions:
        1. send_message(agent_list, message) - Send a message to specific agents, this is useful for general group discussions but also very useful to send message to specific agents if you want to discuss something in private with them.
        2. send_proposal(agent_list, proposal) - Send a proposal to specific agents, you can send a proposal to the entire group but you can also send a proposal to specific agents if you want to discuss something in private with them.
        3. accept_proposal(proposal_id, reason) - Accept a proposal
        4. reject_proposal(proposal_id, reason) - Reject a proposal
        5. write_to_memory(text) - Write strategic observations to your main memory. This should be important insights, leverage points, other agents' motivations, strategic notes, etc. This goes into your permanent notes that inform your decisions.

        If you want to remain silent and wait for other agents to take an action, send a message saying, thank you, i am thinking about this negotiation...
        
        Respond with a JSON object containing your action:
        {{
            "action": "action_name",
            "parameters": {{
                "agent_list": ["agent1", "agent2"] (for send_message/send_proposal),
                "message": "your message" (for send_message),
                "proposal": "your proposal" (for send_proposal),
                "proposal_id": "proposal_id" (for accept/reject),
                "reason": "your reason" (for accept/reject),
                "text": "observation text" (for write_to_memory),
            }}
        }}
        
        Remember: You can only send messages/proposals to specific agents, not broadcast to all.
        The conversation ends when ALL agents accept the SAME proposal.
        """
        
        response = self.get_llm_response(context)
        
        try:
            # Parse JSON response
            if response.startswith("```json"):
                response = response[7:-3]
            elif response.startswith("```"):
                response = response[3:-3]
            
            action_data = json.loads(response)
            return action_data
        except:
            # Fallback to a simple message if JSON parsing fails
            return {
                "action": "send_message",
                "parameters": {
                    "agent_list": [agent.name for agent in other_agents if agent.name != self.name][:2],
                    "message": f"I'm {self.name} and I'm thinking about this negotiation..."
                }
            }

class Simulation:
    def __init__(self, scenario_file: str, llm_type: str = "openai", api_key: str = None):
        self.scenario_file = scenario_file
        self.llm_type = llm_type
        self.api_key = api_key
        self.agents = []
        self.conversation_log = []
        self.scenario_data = None
        
    def load_scenario(self):
        """Load scenario data from JSON file."""
        try:
            with open(self.scenario_file, 'r') as f:
                self.scenario_data = json.load(f)
            print(f"📂 Loaded scenario: {self.scenario_data.get('scenario', 'Unknown')}")
        except Exception as e:
            print(f"❌ Error loading scenario: {e}")
            return False
        return True
    
    def initialize_agents(self):
        """Initialize agents based on scenario data."""
        if not self.scenario_data:
            print("❌ No scenario data loaded")
            return False
        
        print(f"\n🤖 Initializing {len(self.scenario_data.get('agents', []))} agents...")
        for agent_data in self.scenario_data.get('agents', []):
            agent = Agent(
                name=agent_data['name'],
                role=agent_data['role'],
                description=agent_data['description'],
                shareable_preferences=agent_data.get('shareable_preferences', {}),
                private_preferences=agent_data.get('private_preferences', {}),
                llm_type=self.llm_type,
                api_key=self.api_key
            )
            self.agents.append(agent)
            print(f"  ✅ {agent.name} ({agent.role})")
        
        print(f"🎯 Successfully initialized {len(self.agents)} agents")
        return True
    
    def run_simulation(self, max_rounds: int = 50):
        """Run the negotiation simulation."""
        if not self.agents:
            print("No agents initialized")
            return False
        
        print(f"\n🚀 Starting simulation with {len(self.agents)} agents...")
        print(f"📋 Task: {self.scenario_data.get('task', 'Unknown')}")
        print(f"🎯 Deliverable: {self.scenario_data.get('deliverable', 'Unknown')}")
        print("=" * 80)
        
        # Initialize conversation with task information
        initial_message = {
            "timestamp": datetime.now().isoformat(),
            "from": "system",
            "to": [agent.name for agent in self.agents],
            "type": "system_message",
            "content": f"Negotiation begins. Task: {self.scenario_data.get('task', 'Unknown')}. Deliverable: {self.scenario_data.get('deliverable', 'Unknown')}"
        }
        self.conversation_log.append(initial_message)
        print("🔔 System: Negotiation begins!")
        
        round_count = 0
        while round_count < max_rounds:
            round_count += 1
            print(f"\n🔄 ROUND {round_count}")
            print("─" * 50)
            
            # Each agent decides their action
            for i, agent in enumerate(self.agents, 1):
                print(f"\n👤 Agent {i}/{len(self.agents)}: {agent.name} ({agent.role})")
                print("─" * 30)
                
                # Agent observes environment and writes to temp memory
                observation = agent.observe_environment(self.conversation_log, self.agents)
                
                # Agent decides action
                print(f"🤔 {agent.name} is thinking...")
                action_data = agent.decide_action(self.conversation_log, self.agents, self.scenario_data)
                
                # Execute action
                self.execute_agent_action(agent, action_data)
                print()  # Empty line for readability
            
            # Show round summary
            self.print_round_summary()
            
            # Prompt each agent to write to main memory after the round
            print(f"\n🧠 MEMORY WRITING PHASE:")
            print("─" * 30)
            for agent in self.agents:
                print(f"\n🧠 {agent.name} considering memory...")
                memory_action = agent.decide_memory_action(self.conversation_log, self.agents, self.scenario_data)
                self.execute_agent_action(agent, memory_action)
            
            # Check if all agents have accepted the same proposal
            if self.check_consensus():
                print(f"\n🎉🎉🎉 CONSENSUS REACHED! 🎉🎉🎉")
                print(f"All agents accepted the same proposal!")
                print("=" * 80)
                break
        
        if round_count >= max_rounds:
            print(f"\n⏰⏰⏰ TIME LIMIT REACHED ⏰⏰⏰")
            print(f"Simulation ended after {max_rounds} rounds without consensus.")
            print("=" * 80)
        
        # Final summary
        self.print_final_summary()
        return True
    
    def execute_agent_action(self, agent: Agent, action_data: Dict):
        """Execute an agent's decided action."""
        action = action_data.get("action")
        params = action_data.get("parameters", {})
        
        try:
            if action == "send_message":
                agent_list = params.get("agent_list", [])
                message = params.get("message", "")
                agent.send_message(agent_list, message, self.conversation_log)
                
            elif action == "send_proposal":
                agent_list = params.get("agent_list", [])
                proposal = params.get("proposal", "")
                agent.send_proposal(agent_list, proposal, self.conversation_log)
                
            elif action == "accept_proposal":
                proposal_id = params.get("proposal_id", "")
                reason = params.get("reason", "")
                agent.accept_proposal(proposal_id, reason, self.conversation_log)
                
            elif action == "reject_proposal":
                proposal_id = params.get("proposal_id", "")
                reason = params.get("reason", "")
                agent.reject_proposal(proposal_id, reason, self.conversation_log)
                
            elif action == "write_to_memory":
                text = params.get("text", "")
                agent.write_to_memory(text)
                
            elif action == "no_memory_write":
                # Agent chose not to write to memory - this is fine
                pass
                
        except Exception as e:
            print(f"❌ Error executing action for {agent.name}: {e}")
    
    def print_round_summary(self):
        """Print a summary of agent statuses at the end of each round."""
        print("📊 ROUND SUMMARY:")
        print("─" * 30)
        for agent in self.agents:
            status_emoji = {
                "none": "⚪",
                "pending": "🟡", 
                "accepted": "🟢",
                "rejected": "🔴"
            }.get(agent.proposal_status, "❓")
            
            current_prop = agent.current_proposal[:20] + "..." if agent.current_proposal and len(agent.current_proposal) > 20 else agent.current_proposal or "None"
            print(f"  {status_emoji} {agent.name}: {agent.proposal_status} | Proposal: {current_prop}")
        print("─" * 30)
    
    def print_final_summary(self):
        """Print a final summary of the simulation."""
        print("\n📋 FINAL SIMULATION SUMMARY:")
        print("=" * 50)
        print(f"📁 Scenario: {self.scenario_data.get('scenario', 'Unknown')}")
        print(f"🤖 LLM: {self.llm_type}")
        print(f"👥 Agents: {len(self.agents)}")
        print(f"💬 Total messages: {len([e for e in self.conversation_log if e.get('type') == 'message'])}")
        print(f"📋 Total proposals: {len([e for e in self.conversation_log if e.get('type') == 'proposal'])}")
        
        print("\n🎯 AGENT FINAL STATUS:")
        for agent in self.agents:
            status_emoji = {
                "none": "⚪",
                "pending": "🟡", 
                "accepted": "🟢",
                "rejected": "🔴"
            }.get(agent.proposal_status, "❓")
            print(f"  {status_emoji} {agent.name} ({agent.role}): {agent.proposal_status}")
        
        consensus_reached = self.check_consensus()
        print(f"\n🏆 CONSENSUS: {'✅ REACHED' if consensus_reached else '❌ NOT REACHED'}")
        print("=" * 50)
    
    def check_consensus(self) -> bool:
        """Check if all agents have accepted the same proposal."""
        accepted_proposals = []
        for agent in self.agents:
            if agent.proposal_status == "accepted" and agent.current_proposal:
                accepted_proposals.append(agent.current_proposal)
        
        # All agents must have accepted the same proposal
        if len(accepted_proposals) == len(self.agents) and len(set(accepted_proposals)) == 1:
            return True
        return False
    
    def save_simulation_log(self, output_file: str):
        """Save the complete simulation log."""
        simulation_data = {
            "scenario_file": self.scenario_file,
            "llm_type": self.llm_type,
            "timestamp": datetime.now().isoformat(),
            "scenario_data": self.scenario_data,
            "agents": [
                {
                    "name": agent.name,
                    "role": agent.role,
                    "description": agent.description,
                    "main_memory": agent.memory,
                    "temp_memory": agent.temp_memory,
                    "final_proposal_status": agent.proposal_status,
                    "current_proposal": agent.current_proposal
                }
                for agent in self.agents
            ],
            "conversation_log": self.conversation_log
        }
        
        # Create directory if output_file has a directory path
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(simulation_data, f, indent=2)
        
        print(f"Simulation log saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Multi-agent negotiation simulation")
    parser.add_argument("--scenario_file", help="Path to scenario JSON file", default="budget_allocation_3agents.json")
    parser.add_argument("--llm", choices=["gpt5", "gemini"], default="gemini", help="LLM to use")
    parser.add_argument("--api-key", help="API key for the LLM (overrides environment variables)")
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds")
    parser.add_argument("--output", help="Output file path (default: simulations/<scenario_name>_<llm>.json)", default="sim_budget_allocation_3agents_gemini-2.5-pro.json")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        if args.llm == "gpt5":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --api-key")
                return
        elif args.llm == "gemini":
            if not GEMINI_API_KEYS:
                print("Error: No Gemini API keys found. Set GEMINI_API_KEY or GEMINI_API_KEY_1, etc. environment variables or use --api-key")
                return
            api_key = None  # Will use random selection from available keys
    
    # Initialize simulation
    sim = Simulation(args.scenario_file, args.llm, api_key)
    
    # Load scenario and initialize agents
    if not sim.load_scenario():
        return
    
    if not sim.initialize_agents():
        return
        
        # Run simulation
    sim.run_simulation(args.max_rounds)
    
    # Save simulation log
    if args.output:
        output_file = args.output
    else:
        scenario_name = os.path.splitext(os.path.basename(args.scenario_file))[0]
        output_file = f"simulations/{scenario_name}_{args.llm}.json"
    
    # Create simulations directory if it doesn't exist
    # os.makedirs("simulations", exist_ok=True)
    sim.save_simulation_log(output_file)

if __name__ == "__main__":
    main()

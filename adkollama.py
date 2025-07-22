import requests
import json
from typing import Any, Dict, List
from abc import ABC, abstractmethod

def call_ollama(prompt, model_name="qwen3:14b", base_url="http://localhost:11434"):
    """Robust function to call Ollama API and handle NDJSON responses"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        # Ollama may return NDJSON (one JSON per line)
        responses = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        responses.append(data["response"])
                except json.JSONDecodeError:
                    continue
        
        return "".join(responses)
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}"

class BaseEnvironment(ABC):
    """Base environment class"""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def step(self, action: str) -> Dict[str, Any]:
        pass

class BaseAgent(ABC):
    """Base agent class"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> str:
        pass

class BaseTask(ABC):
    """Base task class"""
    
    @abstractmethod
    def get_initial_observation(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_complete(self, observation: Dict[str, Any]) -> bool:
        pass

class OllamaEnvironment(BaseEnvironment):
    """Environment that interfaces with Ollama API"""
    
    def __init__(self, model_name="qwen3:14b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        self.conversation_history = []
        return {"status": "ready", "history": []}
    
    def step(self, action: str) -> Dict[str, Any]:
        """Execute an action (send prompt to Ollama)"""
        print(f"Sending to Ollama: {action[:100]}...")
        
        # Use the robust call_ollama function
        ollama_response = call_ollama(action, self.model_name, self.base_url)
        
        # Update conversation history
        self.conversation_history.append({
            "user": action,
            "assistant": ollama_response
        })
        
        return {
            "response": ollama_response,
            "history": self.conversation_history,
            "done": True,
            "error": ollama_response.startswith("Error connecting")
        }

class QwenAgent(BaseAgent):
    """Agent that uses Qwen3:14B via Ollama"""
    
    def __init__(self, name="QwenAgent"):
        super().__init__(name=name)
        self.environment = None
    
    def set_environment(self, environment: OllamaEnvironment):
        """Set the environment for the agent"""
        self.environment = environment
    
    def act(self, observation: Dict[str, Any]) -> str:
        """Generate an action based on observation"""
        if "query" in observation:
            return observation["query"]
        return "Hello, how can I help you?"

class SimpleQATask(BaseTask):
    """Simple Question-Answering task"""
    
    def __init__(self, query: str):
        self.query = query
        self.completed = False
    
    def get_initial_observation(self) -> Dict[str, Any]:
        """Get the initial observation for the agent"""
        return {
            "query": self.query,
            "task": "answer_question"
        }
    
    def is_complete(self, observation: Dict[str, Any]) -> bool:
        """Check if the task is complete"""
        return observation.get("done", False)

class AgentRunner:
    """Orchestrates the agent, environment, and task"""
    
    def __init__(self, agent: BaseAgent, environment: BaseEnvironment, task: BaseTask):
        self.agent = agent
        self.environment = environment
        self.task = task
    
    def run(self) -> Dict[str, Any]:
        """Run the complete agent loop"""
        print("Starting agent run...")
        
        # Reset environment
        env_state = self.environment.reset()
        print(f"Environment initialized: {env_state}")
        
        # Get initial observation from task
        observation = self.task.get_initial_observation()
        print(f"Initial observation: {observation}")
        
        # Agent acts based on observation
        action = self.agent.act(observation)
        print(f"Agent decided to act: {action[:100]}...")
        
        # Environment processes the action
        result = self.environment.step(action)
        print(f"Environment response received (length: {len(result.get('response', ''))} chars)")
        
        # Check if task is complete
        if self.task.is_complete(result):
            print("Task completed successfully!")
            return result
        else:
            print("Task not completed")
            return result

def main():
    """Main function to run the agent"""
    
    # Test query
    query = "Explain the difference between supervised and unsupervised learning in machine learning."
    
    print("=" * 80)
    print("QWEN3:14B AGENT WITH OLLAMA")
    print("=" * 80)
    
    # Test simple connection first
    print("\n1. Testing Ollama connection...")
    test_response = call_ollama("Hello, can you hear me?")
    print(f"Test response: {test_response[:200]}...")
    
    if test_response.startswith("Error"):
        print("❌ Ollama connection failed. Make sure Ollama is running with qwen3:14b")
        return
    
    print("✅ Ollama connection successful!")
    
    # Create components
    print("\n2. Setting up agent components...")
    environment = OllamaEnvironment()
    agent = QwenAgent()
    task = SimpleQATask(query)
    
    # Set up the agent
    agent.set_environment(environment)
    
    # Create runner and execute
    print("\n3. Running agent...")
    runner = AgentRunner(agent, environment, task)
    result = runner.run()
    
    # Display results
    if result and not result.get("error", False):
        print("\n" + "=" * 80)
        print("FINAL RESPONSE:")
        print("=" * 80)
        print(result["response"])
        print("=" * 80)
    else:
        print("❌ Agent failed to complete the task")
        if result:
            print(f"Error: {result.get('response', 'Unknown error')}")

if __name__ == "__main__":
    main()

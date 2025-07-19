# Challenge 1 - Your very first MCP Server

Welcome to Challenge 1! In this challenge, you'll build your first Model Context Protocol (MCP) server using Semantic Kernel agents. Your MCP server will be a **Joke Generator** that uses multiple AI agents to create, review, and vote on jokes collaboratively.

By completing this challenge, you'll learn how to:
- Create an MCP server using FastMCP
- Implement AI agents with Semantic Kernel
- Use Azure OpenAI for LLM-powered interactions
- Build collaborative agent workflows
- Expose tools, prompts, and resources through MCP

## 1.1 What is MCP (Model Context Protocol)?

The Model Context Protocol (MCP) is an open standard that enables AI assistants to securely access external data sources and tools. Think of it as a bridge that allows AI models to interact with your applications, databases, and services in a standardized way.

Key benefits of MCP:
- **Standardized Integration**: One protocol to connect AI models with any external system
- **Security**: Built-in authentication and permission controls
- **Extensibility**: Easy to add new tools and data sources
- **Interoperability**: Works across different AI platforms and tools

## 1.2 Understanding the Joke Generator MCP Server

Your [`myfirst.py`](myfirst.py) file implements a comprehensive joke generation system using multiple AI agents. Here's what it does:

### 1.2.1 Core Components

**MCP Server Setup**
```python
mcp = FastMCP("Joke Generator")
```
Creates an MCP server named "Joke Generator" that exposes joke-related functionality.

**Azure OpenAI Integration**
```python
def create_kernel() -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="gpt-4o",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    )
    return kernel
```
Sets up the connection to your Azure OpenAI service using environment variables.

### 1.2.2 Agent-Based Joke Generation

The system uses **three specialized comedian agents**:

1. **OneLiners**: Specializes in clever one-liners and wordplay
2. **Observational**: Finds humor in everyday situations  
3. **Satirical**: Uses irony, sarcasm, and exaggeration

These agents work together in a **group chat** coordinated by a **Moderator** agent.

### 1.2.3 MCP Tools Exposed

Your server exposes three main tools that external applications can call:

#### `generate_jokes(topic, number_of_jokes)`
```python
@mcp.tool()
async def generate_jokes(topic: str, number_of_jokes: int = 3, ctx: Context = None) -> str:
```
- Creates a team of comedian agents
- Each agent generates one joke on the specified topic
- Returns a conversation with all generated jokes

#### `review_jokes(jokes)`
```python
@mcp.tool()
async def review_jokes(jokes: str, ctx: Context = None) -> str:
```
- Uses a comedy critic agent to review jokes
- Provides ratings (1-10) and improvement suggestions
- Returns detailed feedback on each joke

#### `vote_on_jokes(jokes)`
```python
@mcp.tool()
async def vote_on_jokes(jokes: str, ctx: Context = None) -> Dict:
```
- Creates a panel of three judge agents:
  - **ComedyVeteran**: Values timing and delivery
  - **ComedyWriter**: Values clever wordplay
  - **AudienceMember**: Values relatability
- Conducts a voting process to select the best joke
- Returns voting conversation and results summary

### 1.2.4 MCP Resources and Prompts

**Sample Jokes Resource**
```python
@mcp.resource("jokes://sample")
def get_sample_jokes() -> str:
```
Provides a collection of sample jokes organized by category (Technology, Food, Animals).

**Joke Topics Prompt**
```python
@mcp.prompt()
def joke_topics_prompt() -> str:
```
Offers a structured prompt for generating joke topic suggestions.

## 1.3 How Agent Collaboration Works

The magic happens through **agent collaboration**:

1. **Joke Generation Flow**:
   - Moderator receives the topic
   - Each comedian agent creates one joke
   - Conversation flows naturally between agents
   - Results are collected and returned

2. **Review Process**:
   - Single critic agent analyzes all jokes
   - Provides structured feedback with ratings
   - Suggests improvements for each joke

3. **Voting Competition**:
   - Three judge agents with different perspectives
   - Each votes for their favorite joke with explanations
   - Moderator tallies votes and announces winner
   - Results processor extracts final summary

## 1.4 Running Your MCP Server

### Option 1: As an MCP Server (Production)
```bash
cd Challenge1
python myfirst.py
```
This starts the MCP server that external applications can connect to.

### Option 2: For Testing Individual Functions
Modify the `if __name__ == "__main__":` block:
```python
if __name__ == "__main__":
    import asyncio
    # Test joke generation
    result = asyncio.run(gen())
    print("Generated jokes:", result)
```

### Option 3: Interactive Development (Notebook)
Use the [`myfirstmcp.ipynb`](myfirstmcp.ipynb) notebook for interactive development and testing.

## 1.5 Environment Setup

Ensure your `.env` file contains:
```
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
```

## 1.6 Understanding the Architecture

```
External Application
        ↓ (MCP Protocol)
    Joke Generator MCP Server
        ↓ (Semantic Kernel)
    Azure OpenAI Service
        ↓ (API Calls)
    GPT-4o Model
```

The MCP server acts as a bridge between external applications and your AI agents, providing a standardized way to access joke generation capabilities.

## 1.7 Key Learning Points

- **MCP Tools**: Functions that external applications can call
- **MCP Resources**: Static data that can be referenced
- **MCP Prompts**: Template prompts for specific use cases
- **Agent Collaboration**: Multiple AI agents working together
- **Semantic Kernel**: Framework for orchestrating AI agents
- **Azure OpenAI**: Cloud-based LLM service integration

## Conclusion

You've successfully created a sophisticated MCP server that demonstrates:
- Multi-agent collaboration
- Structured AI workflows
- External system integration
- Real-world AI application patterns

Your joke generator showcases how AI agents can work together to solve complex creative tasks, with each agent bringing specialized skills to the collaboration.

In the next challenges, you'll build upon these concepts to create even more sophisticated AI systems!

## Next Steps

- Experiment with different comedian personas
- Add new joke categories or styles  
- Implement additional review criteria
- Create custom termination strategies
- Explore other MCP capabilities

Now let's see those AI comedians in action! 🎭
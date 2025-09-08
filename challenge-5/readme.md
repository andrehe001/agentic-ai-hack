# Challenge 5: Agent Orchestration

**Expected Duration:** 60 minutes

## Introduction
By this point we have created **the three agents** and have seen how to evaluate and observe one specific agent. As you know, our use case is a bit more complex and, therefore, we will now create the rest of our architecture to actually make it a multi-agent architecture and not just 3 separate agents. The key word for this challenge will be **Orchestration**.

## What's orchestration and what types are there?
Orchestration in AI agent systems is the process of coordinating multiple specialized agents to work together on complex tasks that a single agent cannot handle alone. It helps break down problems, delegate work efficiently, and ensure that each part of a workflow is managed by the agent best suited for it. 

Some common Orchestration Patterns are:

| Pattern                  | Simple Description                                                                  |
|--------------------------|------------------------------------------------------------------------------------|
| Sequential Orchestration | Agents handle tasks one after the other in a fixed order, passing results along.   |
| Concurrent Orchestration | Many agents work at the same time on similar or different parts of a task.         |
| Group Chat Orchestration | Agents (and people, if needed) discuss and collaborate in a shared conversation.   |
| Handoff Orchestration    | Each agent works until it can’t continue, then hands off the task to another agent.|
| Magentic Orchestration   | A manager agent plans and assigns tasks on the fly as new needs and solutions arise.|

If you want deeper details into orchestration patterns click on this [link](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns?toc=%2Fazure%2Fdeveloper%2Fai%2Ftoc.json&bc=%2Fazure%2Fdeveloper%2Fai%2Fbreadcrumb%2Ftoc.json) to learn more.

Now you might be wondering... ok great... but, **how do I decide on an Orchestration Pattern?** The answer to that question is mostly related to your use case. 
Let's have a look at the 2 most common Orchestration patterns:

| Pattern                    | Flow                                   |
|----------------------------|----------------------------------------|
| Sequential Orchestration   | Agent A → Agent B → Agent C            |
| Concurrent Orchestration   | Agent A + Agent B + Agent C → Combine Results |

In `Sequential Orchestration` the Agents are dependent on a task performed from the previous agent. This is very common in workflows like document processing or step-by-step procedures. With `Concurrent Orchestration` the agents are not dependent on each other and therefore it makes this a great orchestration for parallel processing, multi-source research and so on.

## Let's come back to our use case...
`Concurrent Orchestration` is the answer for our use case. We will have 3 agents that are each responsible for gathering and processing specialized information on different matters from different datasources in our knowledge base. In this challenge, we will create a 4th agent that is responsible for Orchestrating these 3 agents and create the final output that we need. Please have a look at the table underneath and review how we have created our 3 agents.

| Agent | Function | Data Source/Technology | Implementation |
|-------|----------|----------------------|----------------|
| **Claim Reviewer Agent** | Analyzes insurance claims and damage assessments | Cosmos DB data | Azure AI Agent Service + SK Plugins |
| **Policy Checker Agent** | Validates coverage against insurance policies | Azure AI Search connection | Azure AI Agent Service |
| **Risk Analyzer Agent** | Evaluates risk factors and provides recommendations | Cosmos DB data | Azure AI Agent Service + SK Plugins |
| **Master Orchestrator Agent** | Coordinates the three agents and synthesizes their outputs | Combined Plugins + Tools | Semantic Kernel Orchestration |

### Understanding Implementation Approaches: Azure AI Agent Service vs Semantic Kernel Integration

When building intelligent agents, you have two primary implementation approaches available in the Azure ecosystem. **Azure AI Agent Service with direct tool connections** provides a streamlined, low-code approach where agents are configured through the Azure AI Foundry portal with direct connections to Azure services like Azure AI Search, enabling rapid prototyping and deployment with built-in enterprise features like security, monitoring, and compliance. This approach is ideal for straightforward scenarios where agents need to access specific Azure services without complex custom logic. In contrast, **Azure AI Agents with Semantic Kernel integration** offers a more flexible, code-first approach that combines the enterprise-grade capabilities of Azure AI Agent Service with Semantic Kernel's powerful plugin framework. This hybrid approach allows developers to create custom plugins with complex business logic, advanced data processing capabilities, and sophisticated integrations (like our Cosmos DB plugin for retrieving structured claim data), while still benefiting from Azure's managed infrastructure and security features. The Semantic Kernel approach is particularly valuable when you need custom data transformations, complex orchestration patterns, or when integrating with *non-Azure* services.

## Exercise Guide - Time to Orchestrate!

## Part 1- Create your Semantic Kernel Orchestrator

1. Notice we have created an `agents`folder that contains two documents: `tools.py` that (for now!) only contains the CosmosDB Plugin (which we will import for orchestration) 
2. Time to build your orchestrator! Please jump over to `orchestration.ipynb` file for a demonstration on how we will integrated our troop of agents to help us solve our pickle! 

This notebook is composed of only two cells of code. The first one will have in it 4 core components: 3 are dedicated to the creation of the 3 agents we have defined and the last piece is a `task` will be the orchestrator, that defines specific instructions to orchestrate the 3 agent.

In Semantic Kernel's Concurrent Orchestration, [`tasks`](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/concurrent?pivots=programming-language-python#invoke-the-orchestration-1) revolve around integrating AI capabilities with traditional programming through a **modular** architecture. Core tasks include creating and managing skills (collections of related AI functions), designing and using prompts for both natural language and code generation, orchestrating planners to break down goals into executable steps, and using connectors to interface with external services like APIs or databases. Developers also manage memory for context retention, handle input/output pipelines, and coordinate execution flows that combine multiple skills or plugins. These components enable building intelligent, context-aware agents that can reason, plan, and act autonomously.


## Part 2 - Now onto automation!

Great! Now that you've built your orchestrator in the Jupyter notebook, it's time to deploy it as a production-ready Container App. 

[Container apps](https://learn.microsoft.com/en-us/azure/container-apps/overview) are an effective way to deploy and manage multi-agent orchestration systems by providing isolated, scalable environments for each agent or service. They enable agents to run independently while communicating through APIs or messaging systems, allowing for flexible coordination, fault isolation, and dynamic scaling. By using container orchestration platforms like Kubernetes or Azure Container Apps, developers can automate deployment, load balancing, and lifecycle management of complex multi-agent systems in a cloud-native, resilient architecture.

In this scenario, we are deploying our multi-agent orchestrator within a **single Azure Container Apps (CA) environment**. We utilize the Azure Container Registry (ACR) created in challenge 0 to store and manage our container images, ensuring secure and efficient delivery to the CA environment. During deployment, we will pass the necessary configuration and connection details as environment variables using a `.env` file—this includes Foundry, Cosmos and Storage credentials —allowing our orchestrator and agents to access resources dynamically without hardcoding sensitive information. This setup streamlines updates, improves security, and enables seamless integration with Azure services, making our system robust and production-ready.

Jump over to the deployment folder and you will find a:
- `Dockerfile` - A containerization blueprint that creates a Python 3.11-slim environment, installs PowerShell for Azure CLI operations, sets up all required dependencies from `requirements.txt`
- `orchestration.py` file - The production-ready version of your Jupyter notebook orchestrator, refactored as a standalone Python application that initializes your three specialized agents and coordinates their concurrent execution.
- `requirements.txt` - Contains all necessary Python packages to be installed
- `container-apps copy.sh` - A template version of the deployment script that you'll customize with your specific environment variables and configuration settings

1. Go to the `container-apps copy.sh` file, and on the `--env-vars` section, you can find the variables that will be pushed into your ACA. Please fill them with the values that you have automatically retrieved on challenge 0, that you can find on the `.env` file in root.

2. Fill the 4 components that have the #FILL command, RESOURCE_GROUP (previously set), ACR_NAME (previously deployed), CONTAINER_APP_NAME (a new name for your container app) and the CONTAINERAPPS_ENV (a new name for the enviornment we are creating).

3. Now its time to run your docker locally

```bash
cd /workspaces/agentic-ai-hack/challenge-5/deployment && docker build -t insurance-orchestrator .
```
4. Now it's time to push it to the Cloud! Rename your file from `container-apps copy.sh` to `container-apps.sh` and run:

```bash
./container-apps.sh
```

This script automates the complete setup of your multi-agent insurance orchestrator in production. It creates the necessary Azure resources (resource group and container environment), builds your Docker image from the local Dockerfile and pushes it to Azure Container Registry, then deploys it as a scalable container app with all required environment variables for AI Foundry, Cosmos DB, and Azure AI Search connections. The script also configures auto-scaling (0-1 replicas), sets up managed identity for secure Azure service access, and assigns the necessary permissions for your orchestrator to coordinate the three specialized insurance agents in a production-ready environment.


###  Challenge Yourself!

Pick 1-2 expansion ideas and implement them in your system. Consider:
- Which expansion would have the biggest impact on your insurance use case?
- How can you measure the success of your new capabilities?
- What additional data sources or APIs would make your agents even smarter?
- How might you handle errors or edge cases in your expanded system?

**Pro Tip:** Start with one simple expansion before moving to more complex orchestration patterns!


## 🎯 Conclusion

Congratulations! You've successfully built a multi-agent orchestration system that coordinates three specialized insurance agents through a Master Orchestrator. Your system now handles complete insurance claim processing workflows using concurrent orchestration patterns with Semantic Kernel.

**Key Achievements:**
- Implemented concurrent orchestration for parallel agent processing
- Created a Master Orchestrator that synthesizes outputs from multiple agents
- Built hybrid solutions combining Azure AI Agent Service with custom Semantic Kernel plugins
- Developed a production-ready framework for intelligent insurance claim processing
- Prepared the system for enterprise deployment to Azure Functions with scalability and monitoring capabilities

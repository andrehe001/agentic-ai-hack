# Challenge 5: Agent Orchestration

**Expected Duration:** 30 minutes

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
`Concurrent Orchestration` is the answer for our use case. We will have 3 agents that are each responsible for gathering and processing specialized information on different matters from different datasources in our knowledge base. In this challenge, we will create a 4th agent that is responsible for Orchestrating these 3 agents and create the final output that we need. Our orchestration will be something like the following:

![alt text](image-1.png)


In this exercise, you'll create a multi-agent orchestration system using Semantic Kernel's concurrent orchestration pattern.
Our orchestration system consists of four agents. Please have a look at the table underneath and review how we have created our 3 orchestrators.

| Agent | Function | Data Source/Technology | Implementation |
|-------|----------|----------------------|----------------|
| **Claim Reviewer Agent** | Analyzes insurance claims and damage assessments | Cosmos DB data | Azure AI Agent Service + SK Plugins |
| **Policy Checker Agent** | Validates coverage against insurance policies | Azure AI Search connection | Azure AI Agent Service |
| **Risk Analyzer Agent** | Evaluates risk factors and provides recommendations | Cosmos DB data | Azure AI Agent Service + SK Plugins |
| **Master Orchestrator Agent** | Coordinates the three agents and synthesizes their outputs | Combined Plugins + Tools | Semantic Kernel Orchestration |

### Understanding Implementation Approaches: Azure AI Agent Service vs Semantic Kernel Integration

When building intelligent agents, you have two primary implementation approaches available in the Azure ecosystem. **Azure AI Agent Service with direct tool connections** provides a streamlined, low-code approach where agents are configured through the Azure AI Foundry portal with direct connections to Azure services like Azure AI Search, enabling rapid prototyping and deployment with built-in enterprise features like security, monitoring, and compliance. This approach is ideal for straightforward scenarios where agents need to access specific Azure services without complex custom logic. In contrast, **Azure AI Agents with Semantic Kernel integration** offers a more flexible, code-first approach that combines the enterprise-grade capabilities of Azure AI Agent Service with Semantic Kernel's powerful plugin framework. This hybrid approach allows developers to create custom plugins with complex business logic, advanced data processing capabilities, and sophisticated integrations (like our Cosmos DB plugin for retrieving structured claim data), while still benefiting from Azure's managed infrastructure and security features. The Semantic Kernel approach is particularly valuable when you need custom data transformations, complex orchestration patterns, or when integrating with *non-Azure* services, as demonstrated in our Claim Reviewer and Risk Analyzer agents that use custom plugins to process insurance data from Cosmos DB with specialized business rules.

## Exercise Guide - Time to Orchestrate!

## Part 1- Create your Semantic Kernel Orchestrator

1. Notice we have created an `agents`folder that contains two documents: `tools.py` that (for now!) only contains the CosmosDB Plugin (which we will import for orchestration) 
2. Notice that in the same folder we also have the `policy_checker.py` file that brins us back to challenge 2 and our first agent. Please have a look at these two files before beggining the exercise.
3. Time to build your orchestrator! Please jump over to `orchestration.ipynb` file for a demonstration on how we will integrated our troop of agents to help us solve our pickle! 

## Part 2 - Expand your tools! (Optional)

Now that you've built your basic orchestrator, it's time to supercharge your multi-agent system! Let's go back to Github Copilot and use this amazing prompt to provide some ideas on how to expand this challenge:

```
I'm building a multi-agent insurance claim processing system using Azure AI Agent Service and Semantic Kernel orchestration. My current system has:

- Claim Reviewer Agent (analyzes claims using Cosmos DB)
- Policy Checker Agent (validates coverage using Azure AI Search)
- Risk Analyzer Agent (evaluates risk factors using Cosmos DB)
- Master Orchestrator Agent (coordinates all agents)

Current tools: CosmosDB Plugin, Azure AI Search connections

CONTEXT: [Describe your specific insurance use case, claim types, or business requirements here]

Please suggest 3-5 specific, implementable expansions that would add the most value to my system. For each suggestion, provide:
1. **Tool/Feature Name**
2. **Business Value** (how it improves claim processing)
3. **Technical Implementation** (specific Azure services or APIs to use)
4. **Integration Point** (which agent would use it and how)
5. **Code Skeleton** (basic function signature or plugin structure)

Focus on expansions that:
- Solve real insurance industry challenges
- Leverage Azure ecosystem services
- Can be implemented within 20-40 minutes
- Integrate seamlessly with existing orchestration
- Provide measurable business impact

Priority areas: [Choose 1-2: fraud detection, customer experience, processing speed, cost reduction, compliance, accuracy]
```

###  Challenge Yourself!

Pick 1-2 expansion ideas and implement them in your system. Consider:
- Which expansion would have the biggest impact on your insurance use case?
- How can you measure the success of your new capabilities?
- What additional data sources or APIs would make your agents even smarter?
- How might you handle errors or edge cases in your expanded system?

**Pro Tip:** Start with one simple expansion before moving to more complex orchestration patterns!


## 🎯 Conclusion

Congratulations! You've successfully built a multi-agent orchestration system that coordinates three specialized insurance agents through a Master Orchestrator. Your system now handles complete insurance claim processing workflows using concurrent orchestration patterns with Semantic Kernel.

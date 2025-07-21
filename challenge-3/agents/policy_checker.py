"""Policy Checker agent using Semantic Kernel."""
from typing import Optional, Dict, Any, List
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
import json


class PolicyCheckerPlugin:
    """Policy Checker plugin for Semantic Kernel."""
    
    def __init__(self, kernel: Kernel):
        """Initialize the Policy Checker plugin.
        
        Args:
            kernel: The Semantic Kernel instance.
        """
        self.kernel = kernel
    
    @kernel_function(
        description="Get policy details for a specific policy number",
        name="get_policy_details"
    )
    async def get_policy_details(self, policy_number: str) -> str:
        """Get policy details including limits, deductibles, and status.
        
        Args:
            policy_number: The policy number to look up
            
        Returns:
            JSON string containing policy details
        """
        # Mock implementation - replace with actual policy lookup service
        # In a real implementation, you would call a policy management API
        
        # Determine policy type based on policy number format
        policy_type = "dutch" if policy_number.startswith("UNAuto") else "english"
        
        if policy_type == "dutch":
            mock_policy_data = {
                "policy_number": policy_number,
                "status": "active",
                "policy_type": "Autoverzekering All Risk",
                "coverage_language": "dutch",
                "limits": {
                    "liability": "€ 2,500,000",
                    "comprehensive": "€ 50,000",
                    "collision": "€ 50,000"
                },
                "deductibles": {
                    "comprehensive": "€ 500",
                    "collision": "€ 250"
                },
                "effective_date": "2024-01-01",
                "expiration_date": "2024-12-31",
                "additional_services": [
                    "DAS Rechtsbijstand",
                    "Glasgarant",
                    "Pechhulp Nederland"
                ]
            }
        else:
            mock_policy_data = {
                "policy_number": policy_number,
                "status": "active",
                "policy_type": "Comprehensive Auto Insurance",
                "coverage_language": "english",
                "limits": {
                    "liability": "$1,000,000",
                    "comprehensive": "$30,000",
                    "collision": "$30,000"
                },
                "deductibles": {
                    "comprehensive": "$500",
                    "collision": "$250"
                },
                "effective_date": "2024-01-01",
                "expiration_date": "2024-12-31",
                "additional_services": [
                    "Roadside Assistance",
                    "Rental Car Coverage"
                ]
            }
        
        return json.dumps(mock_policy_data, indent=2)
    
    @kernel_function(
        description="Search policy documents for specific terms or coverage information",
        name="search_policy_documents"
    )
    async def search_policy_documents(self, search_query: str, policy_number: Optional[str] = None) -> str:
        """Search policy documents for specific coverage terms.
        
        Args:
            search_query: The search terms to look for in policy documents
            policy_number: Optional policy number to filter search results
            
        Returns:
            JSON string containing search results with policy sections
        """
        # Mock implementation - replace with actual document search service
        # In a real implementation, you would use Azure AI Search or similar
        
        # Determine language context from search query
        dutch_terms = ["eigen schade", "aanrijding", "uitsluitingen", "dekking", "schadegarant", 
                      "rechtsbijstand", "autoverzekering", "polisvoorwaarden"]
        is_dutch_search = any(term in search_query.lower() for term in dutch_terms)
        
        # Also check policy number format
        if policy_number and policy_number.startswith("UNAuto"):
            is_dutch_search = True
        
        if is_dutch_search:
            mock_search_results = [
                {
                    "policy_type": "Autoverzekering UNAuto-02",
                    "section": "6.3",
                    "content": "Verder ben je verzekerd voor schade aan je auto, als deze is veroorzaakt door een aanrijding met een ander voertuig, mits er een schuldige derde partij is.",
                    "relevance_score": 0.95
                },
                {
                    "policy_type": "Autoverzekering UNAuto-02", 
                    "section": "8.1",
                    "content": "Uitsluitingen: Schade veroorzaakt door opzet, grove nalatigheid, of het rijden onder invloed is niet gedekt.",
                    "relevance_score": 0.87
                },
                {
                    "policy_type": "Autoverzekering UNAuto-02",
                    "section": "4.2", 
                    "content": "Eigen schade dekking geldt voor schade aan het verzekerde voertuig ongeacht schuld, met een eigen risico van € 500.",
                    "relevance_score": 0.92
                }
            ]
        else:
            mock_search_results = [
                {
                    "policy_type": "Comprehensive Auto",
                    "section": "2.1",
                    "content": "Collision Coverage: Damage from collisions with other vehicles or objects is covered, subject to deductible.",
                    "relevance_score": 0.93
                },
                {
                    "policy_type": "Comprehensive Auto",
                    "section": "3.5",
                    "content": "Exclusions: Damage caused by intentional acts, racing, or driving under the influence is not covered.",
                    "relevance_score": 0.89
                },
                {
                    "policy_type": "Comprehensive Auto",
                    "section": "4.1",
                    "content": "Comprehensive coverage applies to theft, vandalism, weather damage, and other non-collision incidents.",
                    "relevance_score": 0.85
                }
            ]
        
        # Filter results based on search query relevance
        filtered_results = []
        for result in mock_search_results:
            # Simple relevance check based on search terms
            if any(term.lower() in result["content"].lower() for term in search_query.split()):
                filtered_results.append(result)
        
        return json.dumps({
            "search_query": search_query,
            "total_results": len(filtered_results),
            "results": filtered_results if filtered_results else mock_search_results
        }, indent=2)
    
    @kernel_function(
        description="Check policy coverage for a specific claim",
        name="check_policy_coverage"
    )
    async def check_policy_coverage(
        self,
        policy_number: str,
        claim_description: str,
        claim_language: Optional[str] = None
    ) -> str:
        """Check if a claim is covered under the policy.
        
        Args:
            policy_number: The policy number to check
            claim_description: Description of the claim/incident
            claim_language: Language of the claim (dutch/english), auto-detected if not provided
            
        Returns:
            Coverage assessment: COVERED, NOT_COVERED, PARTIALLY_COVERED, or INSUFFICIENT_EVIDENCE
        """
        # Create chat history for the policy checking process
        chat_history = ChatHistory()
        
        # System prompt with the same logic as the LangGraph version
        system_prompt = """You are a policy-verification specialist. Your task is to decide whether the reported loss is covered under the customer's policy.

MANDATORY STEPS
1. Call `get_policy_details` to confirm the policy is in-force and gather limits / deductibles.
2. Craft one or more focused queries for `search_policy_documents` to locate wording that applies (coverage, exclusions, definitions, limits).
3. If initial searches return no results, try alternative search terms (e.g., "collision", "damage", "vehicle", "exclusions", "coverage").

LANGUAGE-SPECIFIC POLICY SEARCH STRATEGY
• DUTCH CLAIMS: If the claim contains Dutch text, names, locations, or policy numbers starting with "UNAuto", prioritize Dutch policy terms:
  - Use Dutch search terms: "eigen schade", "aanrijding", "uitsluitingen", "dekking", "schadegarant", "rechtsbijstand"
  - Search for "Autoverzekering", "WA All risk", "Polisvoorwaarden", "UNAuto"
  - Look for Dutch-specific services: "DAS", "Glasgarant", "Pechhulp Nederland"
• ENGLISH CLAIMS: Use standard English terms: "collision coverage", "exclusions", "deductible", "comprehensive"
• MIXED RESULTS: If you get both Dutch and English policy results, prioritize the language that matches the claim context

INSUFFICIENT EVIDENCE DETECTION
• LANGUAGE MISMATCH: If you detect a Dutch claim but only find English policy documents (or vice versa), this indicates the relevant policy documents may not be indexed.
• POLICY MISMATCH: If the policy number format suggests a specific policy type (e.g., "UNAuto-02" for Dutch policies) but search results don't contain matching policy documents.
• LOW RELEVANCE: If all search results have very low relevance scores and don't contain terms related to the claim type.

WHEN READING SEARCH RESULTS
• Each result dict contains `policy_type`, `section`, `content`, and `relevance_score`.
• EVALUATE RELEVANCE: Check if the search results actually relate to the claim context (language, policy type, coverage area).
• Cite every passage you rely on. Prefix the quotation or paraphrase with a citation in the form:  `[{{policy_type}} §{{section}}]`.
  Example:  `[Comprehensive Auto §2.1 – Collision Coverage] Damage from collisions with other vehicles is covered …`
  Example:  `[Autoverzekering UNAuto-02 §6.3] Verder ben je verzekerd voor schade aan je auto, als deze is veroorzaakt door een aanrijding …`

WHAT TO INCLUDE IN YOUR ANSWER
• A bullet list of each cited section followed by a short explanation of how it affects coverage.
• Applicable limits and deductibles.
• Any exclusions that negate or restrict coverage.
• If insufficient evidence: Clearly state the mismatch between claim context and available policy documents.

FORMAT
End with a single line containing exactly:  `FINAL ASSESSMENT: COVERED`, `NOT_COVERED`, `PARTIALLY_COVERED`, or `INSUFFICIENT_EVIDENCE` (choose one).

RULES
• Try multiple search queries before concluding no relevant sections exist.
• INSUFFICIENT_EVIDENCE should be used when:
  - Language mismatch between claim and available policy documents
  - Policy type mismatch (e.g., Dutch policy number but only English policies found)
  - No relevant policy sections found despite comprehensive searching
  - Search results don't contain terms or context related to the specific claim
• If you find relevant policy sections that match the claim context, proceed with normal coverage assessment.
• Do NOT hallucinate policy language; only quote or paraphrase returned passages.
• Be concise yet complete."""

        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.SYSTEM, content=system_prompt)
        )
        
        # Start the assessment process
        step_info = []
        
        # Step 1: Get policy details
        try:
            policy_details = await self.get_policy_details(policy_number)
            step_info.append(f"Policy Details Retrieved:\n{policy_details}")
        except Exception as e:
            step_info.append(f"Error retrieving policy details: {str(e)}")
        
        # Step 2: Search policy documents with multiple strategies
        search_queries = []
        
        # Detect language and determine search strategy
        dutch_indicators = ["nederlandse", "nederland", "amsterdam", "rotterdam", "unaauto"]
        is_dutch_claim = any(indicator in claim_description.lower() for indicator in dutch_indicators) or policy_number.startswith("UNAuto")
        
        if is_dutch_claim:
            search_queries = [
                "eigen schade aanrijding",
                "dekking uitsluitingen", 
                "autoverzekering polisvoorwaarden",
                "schadegarant rechtsbijstand"
            ]
        else:
            search_queries = [
                "collision coverage damage",
                "comprehensive exclusions",
                "vehicle coverage limits",
                "deductible policy terms"
            ]
        
        # Perform searches
        for query in search_queries:
            try:
                search_results = await self.search_policy_documents(query, policy_number)
                step_info.append(f"Search Results for '{query}':\n{search_results}")
            except Exception as e:
                step_info.append(f"Error searching for '{query}': {str(e)}")
        
        # Construct the user message
        user_message = f"""Policy Number: {policy_number}
Claim Description: {claim_description}

Assessment Steps Completed:
{chr(10).join(step_info)}

Please provide your policy coverage assessment following the mandatory steps and rules outlined in your instructions."""

        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content=user_message)
        )
        
        # Get the chat completion service
        chat_service = self.kernel.get_service(type=ChatCompletionClientBase)
        
        # Generate the response
        response = await chat_service.get_chat_message_contents(
            chat_history=chat_history,
            settings=self.kernel.get_service_settings()
        )
        
        return response[0].content if response else "Unable to assess policy coverage"


def create_policy_checker_plugin(kernel: Kernel) -> PolicyCheckerPlugin:
    """Create and return a configured Policy Checker plugin.
    
    Args:
        kernel: The Semantic Kernel instance.
        
    Returns:
        Configured PolicyCheckerPlugin instance.
    """
    return PolicyCheckerPlugin(kernel)


# Legacy compatibility function for LangGraph users
def create_policy_checker_agent(llm):
    """Legacy function for compatibility - redirects to Semantic Kernel implementation.
    
    Args:
        llm: LLM instance (not used in SK implementation)
        
    Returns:
        Instructions for using the new Semantic Kernel implementation.
    """
    return """This function has been converted to use Semantic Kernel.
Please use create_policy_checker_plugin(kernel) instead and register it with your Semantic Kernel instance.

Example usage:
```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Create kernel
kernel = Kernel()
kernel.add_service(AzureChatCompletion(...))

# Create and register the plugin
policy_plugin = create_policy_checker_plugin(kernel)
kernel.add_plugin(policy_plugin, "policy_checker")

# Use the functions
result = await kernel.invoke_function(
    plugin_name="policy_checker", 
    function_name="check_policy_coverage",
    policy_number="UNAuto-12345",
    claim_description="Vehicle collision damage"
)
```"""
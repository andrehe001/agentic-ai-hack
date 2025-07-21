"""Risk Analyst agent using Semantic Kernel."""
from typing import Optional, Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
import json


class RiskAnalystPlugin:
    """Risk Analyst plugin for Semantic Kernel."""
    
    def __init__(self, kernel: Kernel):
        """Initialize the Risk Analyst plugin.
        
        Args:
            kernel: The Semantic Kernel instance with tools registered.
        """
        self.kernel = kernel
    
    @kernel_function(
        description="Analyze claimant risk factors and provide fraud detection assessment",
        name="analyze_risk"
    )
    async def analyze_risk(
        self,
        claimant_id: str,
        claim_data: Optional[str] = None,
        incident_details: Optional[str] = None
    ) -> str:
        """Analyze risk factors for insurance claims and detect potential fraud indicators.
        
        Args:
            claimant_id: Unique identifier for the claimant to analyze history
            claim_data: JSON string containing current claim details
            incident_details: Additional incident information for context
            
        Returns:
            Risk assessment with conclusion: LOW_RISK, MEDIUM_RISK, or HIGH_RISK
        """
        # Parse claim data if provided
        claim_info = {}
        if claim_data:
            try:
                claim_info = json.loads(claim_data) if isinstance(claim_data, str) else claim_data
            except json.JSONDecodeError:
                claim_info = {"description": claim_data}
        
        # Create chat history for the risk analysis
        chat_history = ChatHistory()
        
        # System prompt
        system_prompt = """You are a risk analyst specializing in fraud detection and risk assessment for insurance claims.

Your responsibilities:
- Analyze claimant history and claim frequency patterns.
- Identify potential fraud indicators.
- Assess risk factors based on incident details.
- Evaluate supporting documentation quality.
- Provide risk scoring and recommendations.

Use the claimant history data to identify patterns that may indicate:
- Unusual claim frequency
- Suspicious timing patterns
- Inconsistent documentation quality
- High-value claim patterns
- Credit and driving record concerns

Focus on objective risk factors and provide evidence-based assessments.
End your assessment with risk level: LOW_RISK, MEDIUM_RISK, or HIGH_RISK."""

        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.SYSTEM, content=system_prompt)
        )
        
        # Gather claimant history using tools
        additional_info = []
        
        # Get claimant history for risk assessment
        try:
            claimant_history = await self.kernel.invoke_function(
                plugin_name="tools",
                function_name="get_claimant_history",
                claimant_id=claimant_id
            )
            additional_info.append(f"Claimant history: {claimant_history}")
        except Exception as e:
            additional_info.append(f"Failed to get claimant history: {str(e)}")
        
        # Construct the user message with all available information
        user_message_parts = [
            f"Please analyze the risk factors for claimant ID: {claimant_id}"
        ]
        
        if claim_info:
            user_message_parts.append(f"\nCurrent Claim Information:\n{json.dumps(claim_info, indent=2)}")
        
        if incident_details:
            user_message_parts.append(f"\nIncident Details:\n{incident_details}")
        
        if additional_info:
            user_message_parts.append(f"\nClaimant History and Risk Data:\n{chr(10).join(additional_info)}")
        else:
            user_message_parts.append("\nNo claimant history data available for analysis")
        
        user_message_parts.append("\nPlease provide your detailed risk assessment and fraud analysis.")
        
        user_message = "\n".join(user_message_parts)
        
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
        
        return response[0].content if response else "Unable to analyze risk factors"


def create_risk_analyst_plugin(kernel: Kernel) -> RiskAnalystPlugin:
    """Create and return a configured Risk Analyst plugin.
    
    Args:
        kernel: The Semantic Kernel instance with required tools registered.
        
    Returns:
        Configured RiskAnalystPlugin instance.
    """
    return RiskAnalystPlugin(kernel)


# Legacy compatibility function for LangGraph users
def create_risk_analyst_agent(llm):
    """Legacy function for compatibility - redirects to Semantic Kernel implementation.
    
    Args:
        llm: LLM instance (not used in SK implementation)
        
    Returns:
        Instructions for using the new Semantic Kernel implementation.
    """
    return """This function has been converted to use Semantic Kernel.
Please use create_risk_analyst_plugin(kernel) instead and register it with your Semantic Kernel instance."""
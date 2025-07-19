"""Claim Assessor agent using Semantic Kernel."""
from typing import List, Optional, Dict, Any
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
import json


class ClaimAssessorPlugin:
    """Claim Assessor plugin for Semantic Kernel."""
    
    def __init__(self, kernel: Kernel):
        """Initialize the Claim Assessor plugin.
        
        Args:
            kernel: The Semantic Kernel instance with tools registered.
        """
        self.kernel = kernel
    
    @kernel_function(
        description="Assess insurance claims for damage evaluation and cost assessment",
        name="assess_claim"
    )
    async def assess_claim(
        self,
        claim_data: str,
        supporting_images: Optional[str] = None,
        vin_number: Optional[str] = None
    ) -> str:
        """Assess an insurance claim for validity and cost estimates.
        
        Args:
            claim_data: JSON string containing claim details including incident description and damage estimates
            supporting_images: Comma-separated list of image paths to analyze
            vin_number: Vehicle identification number for validation
            
        Returns:
            Detailed assessment with conclusion: VALID, QUESTIONABLE, or INVALID
        """
        # Parse claim data
        try:
            claim_info = json.loads(claim_data) if isinstance(claim_data, str) else claim_data
        except json.JSONDecodeError:
            claim_info = {"description": claim_data}
        
        # Create chat history for the assessment
        chat_history = ChatHistory()
        
        # System prompt
        system_prompt = """You are a claim assessor specializing in damage evaluation and cost assessment.

Your responsibilities:
- Evaluate the consistency between incident description and claimed damage.
- Assess the reasonableness of estimated repair costs.
- Verify supporting documentation (photos, police reports, witness statements).
- Use vehicle details to validate damage estimates.
- Identify any red flags or inconsistencies.

CRITICAL: When you receive a claim with supporting images:
1. ALWAYS analyze EACH image provided
2. Use the extracted data from images in your assessment
3. If image analysis fails, note the failure but continue with available information

Use vehicle details when available to validate damage estimates.

Provide detailed assessments with specific cost justifications that incorporate vehicle details and insights derived from images.
End your assessment with: VALID, QUESTIONABLE, or INVALID."""

        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.SYSTEM, content=system_prompt)
        )
        
        # Gather additional information using tools
        additional_info = []
        
        # Analyze images if provided
        if supporting_images:
            image_paths = [path.strip() for path in supporting_images.split(',')]
            for image_path in image_paths:
                try:
                    # Call the analyze_image function from the kernel
                    image_analysis = await self.kernel.invoke_function(
                        plugin_name="tools",
                        function_name="analyze_image",
                        image_path=image_path
                    )
                    additional_info.append(f"Image analysis for {image_path}: {image_analysis}")
                except Exception as e:
                    additional_info.append(f"Failed to analyze image {image_path}: {str(e)}")
        
        # Get vehicle details if VIN is provided
        if vin_number:
            try:
                vehicle_details = await self.kernel.invoke_function(
                    plugin_name="tools",
                    function_name="get_vehicle_details",
                    vin=vin_number
                )
                additional_info.append(f"Vehicle details: {vehicle_details}")
            except Exception as e:
                additional_info.append(f"Failed to get vehicle details: {str(e)}")
        
        # Construct the user message with all available information
        user_message = f"""Please assess the following insurance claim:

Claim Information:
{json.dumps(claim_info, indent=2)}

Additional Information:
{chr(10).join(additional_info) if additional_info else 'No additional information available'}

Please provide your detailed assessment."""

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
        
        return response[0].content if response else "Unable to assess claim"


def create_claim_assessor_plugin(kernel: Kernel) -> ClaimAssessorPlugin:
    """Create and return a configured Claim Assessor plugin.
    
    Args:
        kernel: The Semantic Kernel instance with required tools registered.
        
    Returns:
        Configured ClaimAssessorPlugin instance.
    """
    return ClaimAssessorPlugin(kernel)


# Legacy compatibility function for LangGraph users
def create_claim_assessor_agent(llm):
    """Legacy function for compatibility - redirects to Semantic Kernel implementation.
    
    Args:
        llm: LLM instance (not used in SK implementation)
        
    Returns:
        Instructions for using the new Semantic Kernel implementation.
    """
    return """This function has been converted to use Semantic Kernel.
Please use create_claim_assessor_plugin(kernel) instead and register it with your Semantic Kernel instance."""
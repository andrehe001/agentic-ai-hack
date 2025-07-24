#!/bin/bash
#
# This script will retrieve necessary keys and properties from Azure Resources 
# deployed using "Deploy to Azure" button and will store them in a file named
# ".env" in the parent directory.

# Login to Azure
if [ -z "$(az account show)" ]; then
  echo "User not signed in Azure. Signin to Azure using 'az login' command."
  az login --use-device-code
fi

# Get the resource group name from the script parameter named resource-group
resourceGroupName=""

# Parse named parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --resource-group) resourceGroupName="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if resourceGroupName is provided
if [ -z "$resourceGroupName" ]; then
    echo "Enter the resource group name where the resources are deployed:"
    read resourceGroupName
fi

# Get resource group deployments, find deployments starting with 'Microsoft.Template' and sort them by timestamp
echo "Getting the deployments in '$resourceGroupName'..."
deploymentName=$(az deployment group list --resource-group $resourceGroupName --query "[?contains(name, 'Microsoft.Template') || contains(name, 'azuredeploy')].{name:name}[0].name" --output tsv)
if [ $? -ne 0 ]; then
    echo "Error occurred while fetching deployments. Exiting..."
    exit 1
fi

# Get output parameters from last deployment using Azure CLI queries instead of jq
echo "Getting the output parameters from the last deployment '$deploymentName' in '$resourceGroupName'..."

# Extract the resource names directly using Azure CLI queries
echo "Extracting the resource names from the deployment outputs..."
storageAccountName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.storageAccountName.value" -o tsv 2>/dev/null || echo "")
logAnalyticsWorkspaceName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.logAnalyticsWorkspaceName.value" -o tsv 2>/dev/null || echo "")
searchServiceName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.searchServiceName.value" -o tsv 2>/dev/null || echo "")
apiManagementName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.apiManagementName.value" -o tsv 2>/dev/null || echo "")
aiFoundryHubName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.aiFoundryHubName.value" -o tsv 2>/dev/null || echo "")
aiFoundryProjectName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.aiFoundryProjectName.value" -o tsv 2>/dev/null || echo "")
keyVaultName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.keyVaultName.value" -o tsv 2>/dev/null || echo "")
containerRegistryName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.containerRegistryName.value" -o tsv 2>/dev/null || echo "")
applicationInsightsName=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.applicationInsightsName.value" -o tsv 2>/dev/null || echo "")

# Extract endpoint URLs
searchServiceEndpoint=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.searchServiceEndpoint.value" -o tsv 2>/dev/null || echo "")
aiFoundryHubEndpoint=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.aiFoundryHubEndpoint.value" -o tsv 2>/dev/null || echo "")
aiFoundryProjectEndpoint=$(az deployment group show --resource-group $resourceGroupName --name $deploymentName --query "properties.outputs.aiFoundryProjectEndpoint.value" -o tsv 2>/dev/null || echo "")

# If deployment outputs are empty, try to discover resources by type
if [ -z "$storageAccountName" ]; then
    echo "Deployment outputs not found, discovering resources by type..."
    storageAccountName=$(az storage account list --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
    searchServiceName=$(az search service list --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
    aiFoundryHubName=$(az cognitiveservices account list --resource-group $resourceGroupName --query "[?kind=='AIServices'].name | [0]" -o tsv 2>/dev/null || echo "")
    applicationInsightsName=$(az monitor app-insights component show --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
fi

# Get the keys from the resources
echo "Getting the keys from the resources..."

# Storage account
if [ -n "$storageAccountName" ]; then
    storageAccountKey=$(az storage account keys list --account-name $storageAccountName --resource-group $resourceGroupName --query "[0].value" -o tsv 2>/dev/null || echo "")
    storageAccountConnectionString=$(az storage account show-connection-string --name $storageAccountName --resource-group $resourceGroupName --query connectionString -o tsv 2>/dev/null || echo "")
else
    echo "Warning: Storage account not found"
    storageAccountKey=""
    storageAccountConnectionString=""
fi

# AI Foundry/Cognitive Services
if [ -n "$aiFoundryHubName" ]; then
    aiFoundryEndpoint=$(az cognitiveservices account show --name $aiFoundryHubName --resource-group $resourceGroupName --query properties.endpoint -o tsv 2>/dev/null || echo "")
    aiFoundryKey=$(az cognitiveservices account keys list --name $aiFoundryHubName --resource-group $resourceGroupName --query key1 -o tsv 2>/dev/null || echo "")
else
    echo "Warning: AI Foundry Hub not found"
    aiFoundryEndpoint=""
    aiFoundryKey=""
fi

# Search service
if [ -n "$searchServiceName" ]; then
    searchServiceKey=$(az search admin-key show --resource-group $resourceGroupName --service-name $searchServiceName --query primaryKey -o tsv 2>/dev/null || echo "")
    if [ -z "$searchServiceEndpoint" ]; then
        searchServiceEndpoint="https://${searchServiceName}.search.windows.net"
    fi
else
    echo "Warning: Search service not found"
    searchServiceKey=""
    searchServiceEndpoint=""
fi

# Application Insights
if [ -n "$applicationInsightsName" ]; then
    appInsightsInstrumentationKey=$(az monitor app-insights component show --app $applicationInsightsName --resource-group $resourceGroupName --query instrumentationKey -o tsv 2>/dev/null || echo "")
else
    echo "Warning: Application Insights not found"
    appInsightsInstrumentationKey=""
fi

# Get Document Intelligence service name and keys
echo "Getting Document Intelligence service information..."
docIntelServiceName=$(az cognitiveservices account list --resource-group $resourceGroupName --query "[?kind=='FormRecognizer'].name | [0]" -o tsv 2>/dev/null || echo "")
if [ -n "$docIntelServiceName" ]; then
    docIntelEndpoint=$(az cognitiveservices account show --name $docIntelServiceName --resource-group $resourceGroupName --query properties.endpoint -o tsv 2>/dev/null || echo "")
    docIntelKey=$(az cognitiveservices account keys list --name $docIntelServiceName --resource-group $resourceGroupName --query key1 -o tsv 2>/dev/null || echo "")
else
    echo "Warning: No Document Intelligence (FormRecognizer) service found in resource group. You may need to deploy one."
    docIntelEndpoint=""
    docIntelKey=""
fi



# Add this section after getting the search service information (around line 100)

# Get Azure AI Search connection ID
if [ -n "$aiFoundryHubName" ] && [ -n "$searchServiceName" ]; then
    echo "Getting Azure AI Search connection ID..."
    azureAIConnectionId=$(az cognitiveservices account connection list --account-name $aiFoundryHubName --resource-group $resourceGroupName --query "[?contains(name, 'aisearch')].name | [0]" -o tsv 2>/dev/null || echo "")
    
    # If no connection found by name pattern, try getting the first CognitiveSearch connection
    if [ -z "$azureAIConnectionId" ]; then
        azureAIConnectionId=$(az cognitiveservices account connection list --account-name $aiFoundryHubName --resource-group $resourceGroupName --query "[?properties.category=='CognitiveSearch'].name | [0]" -o tsv 2>/dev/null || echo "")
    fi
else
    echo "Warning: Cannot get Azure AI connection ID - AI Foundry Hub or Search Service not found"
    azureAIConnectionId=""
fi


# If deployment outputs are empty or missing project info, try to discover resources by type
if [ -z "$storageAccountName" ] || [ -z "$aiFoundryProjectName" ]; then
    if [ -z "$storageAccountName" ]; then
        echo "Deployment outputs not found, discovering resources by type..."
    fi
    if [ -z "$aiFoundryProjectName" ]; then
        echo "AI Foundry Project Name not found in deployment outputs, attempting discovery..."
    fi
    
    storageAccountName=$(az storage account list --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
    searchServiceName=$(az search service list --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
    aiFoundryHubName=$(az cognitiveservices account list --resource-group $resourceGroupName --query "[?kind=='AIServices'].name | [0]" -o tsv 2>/dev/null || echo "")
    applicationInsightsName=$(az monitor app-insights component show --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
    
    # Try to discover AI Foundry project name using different methods
    if [ -z "$aiFoundryProjectName" ] && [ -n "$aiFoundryHubName" ]; then
        echo "Attempting to discover AI Foundry project name..."
        
        # Method 1: Try to list AI projects (if az ml extension is available)
        aiFoundryProjectName=$(az ml workspace list --resource-group $resourceGroupName --query "[0].name" -o tsv 2>/dev/null || echo "")
        
        # Method 2: If that fails, try to find resources with "project" or "aiproject" in the name
        if [ -z "$aiFoundryProjectName" ]; then
            aiFoundryProjectName=$(az resource list --resource-group $resourceGroupName --query "[?contains(name, 'project') || contains(name, 'aiproject')].name | [0]" -o tsv 2>/dev/null || echo "")
        fi
        
        # Method 3: If still empty, construct from hub name pattern (common pattern: hub-name -> project-name)
        if [ -z "$aiFoundryProjectName" ] && [[ "$aiFoundryHubName" =~ -aifoundry- ]]; then
            # Replace "aifoundry" with "aiproject" in the hub name
            aiFoundryProjectName=$(echo "$aiFoundryHubName" | sed 's/-aifoundry-/-aiproject-/')
            echo "Constructed project name from hub pattern: $aiFoundryProjectName"
        fi
        
        # Method 4: Last resort - ask user or use default pattern
        if [ -z "$aiFoundryProjectName" ]; then
            echo "Could not auto-discover AI Foundry project name."
            echo "Available resources in resource group:"
            az resource list --resource-group $resourceGroupName --query "[].{name:name, type:type}" -o table 2>/dev/null || echo "Failed to list resources"
            echo ""
            echo "Please enter the AI Foundry project name (or press Enter to skip):"
            read userProjectName
            if [ -n "$userProjectName" ]; then
                aiFoundryProjectName="$userProjectName"
            fi
        fi
        
        if [ -n "$aiFoundryProjectName" ]; then
            echo "Using AI Foundry project name: $aiFoundryProjectName"
        fi
    fi
fi

# Construct Azure AI Search connection ID directly
if [ -n "$aiFoundryHubName" ] && [ -n "$searchServiceName" ]; then
    echo "Constructing Azure AI Search connection ID..."
    
    # Get subscription ID
    subscriptionId=$(az account show --query id -o tsv 2>/dev/null || echo "")
    
    if [ -n "$subscriptionId" ]; then
        # Construct the connection ID based on the pattern you provided
        # Pattern: /subscriptions/{subscription}/resourceGroups/{rg}/providers/Microsoft.CognitiveServices/accounts/{aiFoundryHub}/connections/{searchServiceWithoutDashes}
        searchServiceNameNoDashes=$(echo "$searchServiceName" | sed 's/-//g')
        azureAIConnectionId="/subscriptions/${subscriptionId}/resourceGroups/${resourceGroupName}/providers/Microsoft.CognitiveServices/accounts/${aiFoundryHubName}/connections/${searchServiceNameNoDashes}"
        echo "Constructed connection ID: $azureAIConnectionId"
    else
        echo "Warning: Could not get subscription ID"
        azureAIConnectionId=""
    fi
else
    echo "Warning: Cannot construct Azure AI connection ID - AI Foundry Hub or Search Service not found"
    azureAIConnectionId=""
fi
# Overwrite the existing .env file
if [ -f ../.env ]; then
    rm ../.env
fi

# Store the keys and properties in a file
echo "Storing the keys and properties in '.env' file..."

# Azure Storage (with both naming conventions)
echo "AZURE_STORAGE_ACCOUNT_NAME=\"$storageAccountName\"" >> ../.env
echo "AZURE_STORAGE_ACCOUNT_KEY=\"$storageAccountKey\"" >> ../.env
echo "AZURE_STORAGE_CONNECTION_STRING=\"$storageAccountConnectionString\"" >> ../.env

# Azure AI Document Intelligence
echo "AZURE_DOC_INTEL_ENDPOINT=\"$docIntelEndpoint\"" >> ../.env
echo "AZURE_DOC_INTEL_KEY=\"$docIntelKey\"" >> ../.env

# Other Azure services
echo "LOG_ANALYTICS_WORKSPACE_NAME=\"$logAnalyticsWorkspaceName\"" >> ../.env
echo "SEARCH_SERVICE_NAME=\"$searchServiceName\"" >> ../.env
echo "SEARCH_SERVICE_ENDPOINT=\"$searchServiceEndpoint\"" >> ../.env
echo "SEARCH_ADMIN_KEY=\"$searchServiceKey\"" >> ../.env
echo "API_MANAGEMENT_NAME=\"$apiManagementName\"" >> ../.env
echo "AI_FOUNDRY_HUB_NAME=\"$aiFoundryHubName\"" >> ../.env
echo "AI_FOUNDRY_PROJECT_NAME=\"$aiFoundryProjectName\"" >> ../.env
echo "AI_FOUNDRY_ENDPOINT=\"$aiFoundryEndpoint\"" >> ../.env
echo "AI_FOUNDRY_KEY=\"$aiFoundryKey\"" >> ../.env
echo "AI_FOUNDRY_HUB_ENDPOINT=\"$aiFoundryHubEndpoint\"" >> ../.env
echo "AI_FOUNDRY_PROJECT_ENDPOINT=\"$aiFoundryProjectEndpoint\"" >> ../.env
echo "KEY_VAULT_NAME=\"$keyVaultName\"" >> ../.env
echo "CONTAINER_REGISTRY_NAME=\"$containerRegistryName\"" >> ../.env
echo "APPLICATION_INSIGHTS_NAME=\"$applicationInsightsName\"" >> ../.env
echo "APPLICATION_INSIGHTS_INSTRUMENTATION_KEY=\"$appInsightsInstrumentationKey\"" >> ../.env
echo "AZURE_AI_CONNECTION_ID=\"$azureAIConnectionId\"" >> ../.env

# For backward compatibility, also set OpenAI-style variables pointing to AI Foundry
echo "AZURE_OPENAI_SERVICE_NAME=\"$aiFoundryHubName\"" >> ../.env
echo "AZURE_OPENAI_ENDPOINT=\"$aiFoundryEndpoint\"" >> ../.env
echo "AZURE_OPENAI_KEY=\"$aiFoundryKey\"" >> ../.env

echo "Keys and properties are stored in '.env' file successfully."

# Display summary of what was configured
echo ""
echo "=== Configuration Summary ==="
echo "Storage Account: $storageAccountName"
echo "Search Service: $searchServiceName"
echo "AI Foundry Hub: $aiFoundryHubName"
if [ -n "$docIntelServiceName" ]; then
    echo "Document Intelligence: $docIntelServiceName"
else
    echo "Document Intelligence: NOT FOUND - You may need to deploy this service"
fi
echo "Environment file created: ../.env"

# Show what needs to be deployed
missing_services=""
if [ -z "$storageAccountName" ]; then missing_services="$missing_services Storage"; fi
if [ -z "$searchServiceName" ]; then missing_services="$missing_services Search"; fi
if [ -z "$aiFoundryHubName" ]; then missing_services="$missing_services AI-Foundry"; fi
if [ -z "$docIntelServiceName" ]; then missing_services="$missing_services Document-Intelligence"; fi

if [ -n "$missing_services" ]; then
    echo ""
    echo "⚠️  Missing services:$missing_services"
    echo "You may need to deploy these services manually or check your deployment template."
fi
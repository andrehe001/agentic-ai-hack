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

# Get output parameters from last deployment to the resource group and store them as variables
echo "Getting the output parameters from the last deployment '$deploymentName' in '$resourceGroupName'..."
az deployment group show --resource-group $resourceGroupName --name $deploymentName --query properties.outputs > tmp_outputs.json
if [ $? -ne 0 ]; then
    echo "Error occurred while fetching the output parameters. Exiting..."
    exit 1
fi

# Extract the resource names from the output parameters (matching bicep outputs)
echo "Extracting the resource names from the output parameters..."
storageAccountName=$(jq -r '.storageAccountName.value' tmp_outputs.json)
logAnalyticsWorkspaceName=$(jq -r '.logAnalyticsWorkspaceName.value' tmp_outputs.json)
searchServiceName=$(jq -r '.searchServiceName.value' tmp_outputs.json)
apiManagementName=$(jq -r '.apiManagementName.value' tmp_outputs.json)
aiFoundryHubName=$(jq -r '.aiFoundryHubName.value' tmp_outputs.json)
aiFoundryProjectName=$(jq -r '.aiFoundryProjectName.value' tmp_outputs.json)
keyVaultName=$(jq -r '.keyVaultName.value' tmp_outputs.json)
containerRegistryName=$(jq -r '.containerRegistryName.value' tmp_outputs.json)
applicationInsightsName=$(jq -r '.applicationInsightsName.value' tmp_outputs.json)

# Extract endpoint URLs
searchServiceEndpoint=$(jq -r '.searchServiceEndpoint.value' tmp_outputs.json)
aiFoundryHubEndpoint=$(jq -r '.aiFoundryHubEndpoint.value' tmp_outputs.json)
aiFoundryProjectEndpoint=$(jq -r '.aiFoundryProjectEndpoint.value' tmp_outputs.json)

# Delete the temporary file
rm tmp_outputs.json

# Get the keys from the resources
echo "Getting the keys from the resources..."
storageAccountKey=$(az storage account keys list --account-name $storageAccountName --resource-group $resourceGroupName --query "[0].value" -o tsv)
storageAccountConnectionString=$(az storage account show-connection-string --name $storageAccountName --resource-group $resourceGroupName --query connectionString -o tsv)

# Get AI Foundry endpoint and key
aiFoundryEndpoint=$(az cognitiveservices account show --name $aiFoundryHubName --resource-group $resourceGroupName --query properties.endpoint -o tsv)
aiFoundryKey=$(az cognitiveservices account keys list --name $aiFoundryHubName --resource-group $resourceGroupName --query key1 -o tsv)

# Get Search service key
searchServiceKey=$(az search admin-key show --resource-group $resourceGroupName --service-name $searchServiceName --query primaryKey -o tsv)

# Get Application Insights instrumentation key
appInsightsInstrumentationKey=$(az monitor app-insights component show --app $applicationInsightsName --resource-group $resourceGroupName --query instrumentationKey -o tsv)

# Overwrite the existing .env file
if [ -f ../.env ]; then
    rm ../.env
fi

# Store the keys and properties in a file
echo "Storing the keys and properties in '.env' file..."
echo "STORAGE_ACCOUNT_NAME=\"$storageAccountName\"" >> ../.env
echo "STORAGE_KEY=\"$storageAccountKey\"" >> ../.env
echo "STORAGE_CONNECTION_STRING=\"$storageAccountConnectionString\"" >> ../.env
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

# For backward compatibility, also set OpenAI-style variables pointing to AI Foundry
echo "AZURE_OPENAI_SERVICE_NAME=\"$aiFoundryHubName\"" >> ../.env
echo "AZURE_OPENAI_ENDPOINT=\"$aiFoundryEndpoint\"" >> ../.env
echo "AZURE_OPENAI_KEY=\"$aiFoundryKey\"" >> ../.env

echo "Keys and properties are stored in '.env' file successfully."
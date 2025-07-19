@allowed([
  'swedencentral'
])
@description('Azure location where resources should be deployed (e.g., swedencentral)')
param location string = 'swedencentral'

var prefix = 'msagthack'
var suffix = uniqueString(resourceGroup().id)

/*
  Create Storage Account
*/

var storageAccountName = replace('${prefix}-sa-${suffix}', '-', '')

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
}

/*
  Create Log Analytics Workspace
*/

var logAnalyticsWorkspaceName = '${prefix}-loganalytics-${suffix}'

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2021-06-01' = {
  name: logAnalyticsWorkspaceName
  location: location
  properties: {
    retentionInDays: 30
    features: {
      searchVersion: 1
    }
    sku: {
      name: 'PerGB2018'
    }
  }
}

/*
  Create Azure AI Search
*/

var searchServiceName = '${prefix}-search-${suffix}'

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: {
    name: 'basic'
  }
  properties: {
    hostingMode: 'default'
  }
}

/*
  Create Azure API Management
*/

var apimServiceName = '${prefix}-apim-${suffix}'

resource apiManagement 'Microsoft.ApiManagement/service@2023-05-01-preview' = {
  name: apimServiceName
  location: location
  sku: {
    name: 'Developer'
    capacity: 1
  }
  properties: {
    publisherEmail: 'admin@contoso.com'
    publisherName: 'Contoso'
    customProperties: {
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls10': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Tls11': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Protocols.Ssl30': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls10': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Tls11': 'False'
      'Microsoft.WindowsAzure.ApiManagement.Gateway.Security.Backend.Protocols.Ssl30': 'False'
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

/*
  Create Azure AI Foundry Hub
*/

var aiFoundryHubName = '${prefix}-aifoundry-hub-${suffix}'
var keyVaultName = '${prefix}-kv-${suffix}'
var applicationInsightsName = '${prefix}-appinsights-${suffix}'

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: applicationInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enableRbacAuthorization: true
  }
}

// Azure AI Foundry Hub
resource aiFoundryHub 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: aiFoundryHubName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  kind: 'Hub'
  properties: {
    friendlyName: aiFoundryHubName
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    hbiWorkspace: false
    managedNetwork: {
      isolationMode: 'Disabled'
    }
    v1LegacyMode: false
    publicNetworkAccess: 'Enabled'
    discoveryUrl: 'https://${location}.api.azureml.ms/discovery'
  }
}

/*
  Create Azure OpenAI Service for AI Foundry Hub
*/

var openAIServiceName = '${prefix}-openai-${suffix}'

resource openAIService 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAIServiceName
  location: location
  sku: {
    name: 'S0'
  }
  kind: 'OpenAI'
  properties: {
    apiProperties: {
      statisticsEnabled: false
    }
    customSubDomainName: openAIServiceName
  }
}

resource gpt4oMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  name: 'gpt-4o-mini'
  parent: openAIService
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: 50
  }
}


/*
  Return output values
*/

output storageAccountName string = storageAccountName
output logAnalyticsWorkspaceName string = logAnalyticsWorkspaceName
output searchServiceName string = searchServiceName
output apiManagementName string = apimServiceName
output aiFoundryHubName string = aiFoundryHubName
output keyVaultName string = keyVaultName

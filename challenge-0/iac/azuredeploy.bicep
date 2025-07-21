@allowed([
  'swedencentral'
])
@description('Azure location where resources should be deployed (e.g., swedencentral)')
param location string = 'swedencentral'

@description('Friendly name for your Azure AI Foundry hub resource')
param aiFoundryName string = 'aifoundry'

@description('Name for the AI project')
param aiProjectName string = 'my-ai-project'
param locationDocumentIntelligence string = 'westeurope' // West Europe hast the latest models needed for Document Intelligence

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
  properties: {
    allowBlobPublicAccess: false
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
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
  Create Azure Document Intelligence
*/

var documentIntelligenceName = '${prefix}-di-${suffix}'

resource documentIntelligence 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: documentIntelligenceName
  location: locationDocumentIntelligence
  sku: {
    name: 'S0'
  }
  kind: 'FormRecognizer'
  properties: {
    customSubDomainName: documentIntelligenceName
    apiProperties: {
      statisticsEnabled: false
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
    replicaCount: 1
    partitionCount: 1
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
  Create Container Registry
*/

var containerRegistryName = replace('${prefix}cr${suffix}', '-', '')

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

/*
  Create Application Insights and Key Vault
*/

var keyVaultName = '${prefix}kv${suffix}'  // Shortened to fit 24 char limit
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

/*
  An AI Foundry resources is a variant of a CognitiveServices/account resource type
*/ 
resource aiFoundry 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: aiFoundryName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  kind: 'AIServices'
  properties: {
    // required to work in AI Foundry
    allowProjectManagement: true 

    // Defines developer API endpoint subdomain
    customSubDomainName: aiFoundryName

    disableLocalAuth: false
  }
}

/*
  Developer APIs are exposed via a project, which groups in- and outputs that relate to one use case, including files.
  Its advisable to create one project right away, so development teams can directly get started.
  Projects may be granted individual RBAC permissions and identities on top of what account provides.
*/ 
resource aiProject 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  name: aiProjectName
  parent: aiFoundry
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {}
}

/*
  Optionally deploy a model to use in playground, agents and other tools.
*/
resource gpt4MiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01'= {
  parent: aiFoundry
  name: 'gpt-4o-mini'
  sku : {
    capacity: 1
    name: 'GlobalStandard'
  }
  properties: {
    model:{
      name: 'gpt-4o-mini'
      format: 'OpenAI'
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01'= {
  parent: aiFoundry
  name: 'text-embedding-ada-002'
  sku : {
    capacity: 1
    name: 'GlobalStandard'
  }
  properties: {
    model:{
      name: 'text-embedding-ada-002'
      format: 'OpenAI'
    }
  }
}

/*
  Create RBAC assignments for the AI Hub and Project managed identities
*/

// Get the Cognitive Services User role definition
var cognitiveServicesUserRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')

// Grant the AI Project access to the AI Foundry service
resource projectAIFoundryRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(aiFoundry.id, aiProject.id, cognitiveServicesUserRoleId)
  scope: aiFoundry
  properties: {
    roleDefinitionId: cognitiveServicesUserRoleId
    principalId: aiProject.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Get the Search Service Contributor role definition
var searchServiceContributorRoleId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7ca78c08-252a-4471-8644-bb5ff32d4ba0')

// Grant the AI Foundry access to the Search service
resource aiFoundrySearchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, aiFoundry.id, searchServiceContributorRoleId)
  scope: searchService
  properties: {
    roleDefinitionId: searchServiceContributorRoleId
    principalId: aiFoundry.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant the AI Project access to the Search service
resource projectSearchRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, aiProject.id, searchServiceContributorRoleId)
  scope: searchService
  properties: {
    roleDefinitionId: searchServiceContributorRoleId
    principalId: aiProject.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

/*
  Return output values
*/

output storageAccountName string = storageAccountName
output logAnalyticsWorkspaceName string = logAnalyticsWorkspaceName
output searchServiceName string = searchServiceName
output apiManagementName string = apimServiceName
output aiFoundryHubName string = aiFoundryName
output aiFoundryProjectName string = aiProjectName
output keyVaultName string = keyVaultName
output containerRegistryName string = containerRegistryName
output applicationInsightsName string = applicationInsightsName
output documentIntelligenceName string = documentIntelligenceName


// Output important endpoints and connection information
output searchServiceEndpoint string = 'https://${searchServiceName}.search.windows.net/'
output aiFoundryHubEndpoint string = 'https://ml.azure.com/home?wsid=${aiFoundry.id}'
output aiFoundryProjectEndpoint string = 'https://ai.azure.com/build/overview?wsid=${aiProject.id}'

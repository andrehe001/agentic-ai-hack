#!/bin/bash

# Helper script to get the service principal object ID for Azure deployment
# This is needed for the optional service principal permissions in Challenge 5

echo "🔍 Getting service principal Object ID for Azure deployment..."
echo ""

# Check if user is logged in to Azure CLI
if ! az account show &> /dev/null; then
    echo "❌ You are not logged in to Azure CLI"
    echo "Please run: az login"
    exit 1
fi

# The service principal client ID used in this hackathon
CLIENT_ID="3b442f77-79d6-4e66-8ba0-aaa486723d51"

# Get the object ID
OBJECT_ID=$(az ad sp show --id "$CLIENT_ID" --query "id" --output tsv 2>/dev/null)

if [[ -z "$OBJECT_ID" ]]; then
    echo "❌ Error: Could not find service principal with client ID: $CLIENT_ID"
    echo "   This might mean:"
    echo "   • You don't have permission to view this service principal"
    echo "   • The service principal doesn't exist in your tenant"
    echo "   • You're connected to the wrong Azure tenant"
    echo ""
    echo "💡 If you don't need service principal permissions for Challenge 5,"
    echo "   you can deploy without this parameter (leave it empty)."
    exit 1
fi

echo "✅ Service Principal Object ID: $OBJECT_ID"
echo ""
echo "📋 Next steps:"
echo "1. Copy the Object ID above"
echo "2. Use the 'Deploy to Azure' button in the README"
echo "3. Paste this Object ID into the 'servicePrincipalObjectId' parameter"
echo "4. Complete the deployment"
echo ""
echo "💡 Or leave the parameter empty if you don't need service principal permissions."

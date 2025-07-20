# Challenge 1 - Your very first MCP Server

Welcome to Challenge 1! In this challenge, you'll build your first intelligent agent using Azure AI Agent Service. This agent will be capable of searching through insurance policies using advanced AI capabilities including OCR, vectorization, and semantic search.

By the end of this challenge, you will have:
- Set up Azure Blob Storage to store policy and claims data
- Implemented OCR capabilities for processing image-based documents
- Created a vectorized search index using Azure AI Search
- Built your first Azure AI Agent that can intelligently search through insurance policies
- Learned to integrate multiple Azure AI services in a cohesive solution

## 📁 Data Structure

Your challenge includes the following data:

### Claims Data (`data/claims/`)
- `crash1.jpg` - Vehicle accident documentation
- `crash2.jpg` - Additional accident imagery  
- `invoice.png` - Repair invoice document

### Policy Data (`data/policies/`)
- `commercial_auto_policy.md` - Commercial vehicle coverage
- `comprehensive_auto_policy.md` - Full coverage policy
- `high_value_vehicle_policy.md` - Luxury vehicle coverage
- `liability_only_policy.md` - Basic liability coverage
- `motorcycle_policy.md` - Motorcycle-specific policy

## 🎉 Success Criteria

Your Challenge 1 solution should demonstrate:

✅ **Data Upload**: All policy and claims data successfully uploaded to Azure Blob Storage  
✅ **OCR Processing**: Text successfully extracted from claim images  
✅ **Vector Search**: Policy documents vectorized and searchable in Azure AI Search  
✅ **Agent Creation**: Functional Azure AI Agent that can search policies  
✅ **Query Handling**: Agent responds accurately to policy-related questions  


## 🔍 Sample Interactions

```
User: "I need insurance for my motorcycle. What options do I have?"
Agent: "Based on our motorcycle policy, we offer comprehensive coverage including collision, theft protection, and liability coverage. The motorcycle policy covers..."

User: "What's the difference between comprehensive and liability-only?"
Agent: "Great question! Here are the key differences:
- Comprehensive coverage includes collision, theft, vandalism, and natural disasters
- Liability-only covers damages to others but not your own vehicle
- Comprehensive has higher premiums but provides broader protection..."
```
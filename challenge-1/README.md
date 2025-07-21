# Challenge 1 - Document Processing and Vectorized Search

**Expected Duration:** 90 minutes

## Introduction

Welcome to Challenge 1! In this challenge, you'll build a comprehensive document processing and search system using Azure AI services. This challenge demonstrates how to process insurance documents (both text and images), create vectorized search capabilities, and prepare data for intelligent AI agents.

By the end of this challenge, you will have:
- Set up Azure Blob Storage for document management
- Implemented multimodal document processing using Azure OpenAI GPT-4o
- Created OCR capabilities for extracting text from insurance claim images
- Built a vectorized search index using Azure AI Search with integrated vectorization
- Processed and indexed real insurance data for semantic search capabilities

## 📁 Data Structure

Your challenge includes comprehensive insurance data across three categories:

### Claims Data (`data/claims/`)
- `crash1.jpg` - Vehicle accident scene documentation
- `crash2.jpg` - Additional accident imagery  
- `crash3.jpg` - Multi-vehicle collision photos
- `crash4.jpeg` - Damage assessment imagery
- `crash5.jpg` - Comprehensive accident documentation
- `invoice.png` - Repair invoice and billing document

### Policy Data (`data/policies/`)
- `commercial_auto_policy.md` - Commercial vehicle coverage guidelines
- `comprehensive_auto_policy.md` - Full coverage policy details
- `high_value_vehicle_policy.md` - Luxury and high-value vehicle coverage
- `liability_only_policy.md` - Basic liability coverage terms
- `motorcycle_policy.md` - Motorcycle-specific insurance policy

### Claim Statements (`data/statements/`)
- `crash1.md` through `crash5.md` - Detailed written statements corresponding to each claim

## 🛠️ Implementation Components

### 1. Document Processing Notebook ([`1.document-processing.ipynb`](challenge-1/1.document-processing.ipynb))
- **Document Upload**: Uploads all documents to Azure Blob Storage
- **Text Processing**: Processes markdown policy files using GPT-4o
- **OCR Processing**: Extracts text from claim images using GPT-4o vision capabilities
- **Text Enhancement**: Cleans and prepares documents for vectorization
- **Multimodal AI**: Leverages GPT-4o for both text and image processing

### 2. Document Vectorization Notebook ([`2.document-vectorizing.ipynb`](challenge-1/2.document-vectorizing.ipynb))
- **Search Index Creation**: Sets up Azure AI Search with integrated vectorization
- **Document Indexing**: Uploads processed documents to search index
- **Automatic Vectorization**: Uses Azure AI Search's integrated vectorization with text-embedding-ada-002
- **Semantic Search**: Enables hybrid search capabilities (keyword + vector + semantic)
- **Testing Interface**: Provides search testing and validation


## 🎯 Success Criteria

Your Challenge 1 solution demonstrates:

✅ **Azure Blob Storage Integration**: All documents uploaded and organized  
✅ **Multimodal Processing**: Text and image documents processed with GPT-4o  
✅ **OCR Capabilities**: Text successfully extracted from insurance claim images  
✅ **Vector Search Index**: Documents vectorized using Azure AI Search integrated vectorization  
✅ **Semantic Search**: Hybrid search capabilities (keyword + vector + semantic)  
✅ **Real-time Processing**: Automatic vectorization and indexing pipeline  
✅ **Search Testing**: Validated search functionality across document types  

## 🔍 Search Capabilities

The completed system supports:
- **Semantic Search**: Natural language queries across all documents
- **Vector Search**: Similarity-based document retrieval
- **Hybrid Search**: Combined keyword and semantic search
- **Multi-category Search**: Query across policies, claims, and statements
- **Real-time Indexing**: Automatic processing of new documents

## 📋 Sample Search Queries

```
"What motorcycle coverage options are available?"
→ Returns relevant sections from motorcycle_policy.md

"Show me accident damage reports"
→ Returns OCR-processed text from crash images

"Compare comprehensive vs liability coverage"
→ Returns comparative information from relevant policies

"Find claims related to vehicle collisions"
→ Returns both image-based claims and written statements
```

## 🚀 Next Steps

This foundation prepares you for:
- **Challenge 2**: Building multi-agent systems using this search infrastructure
- **Challenge 3**: Advanced agent orchestration and workflow management
- **Challenge 4**: Production deployment and scaling
- **Challenge 5**: End-to-end AI agent solutions

The vectorized search index and processed documents created in this challenge serve as the knowledge base for all subsequent AI agent implementations.
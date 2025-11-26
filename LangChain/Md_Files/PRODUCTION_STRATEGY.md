# Production Strategy: Handling Large Data with LLM Token Limits

## 🎯 Problem Statement

**Real-Time Production Challenges:**
- SAP returns 1M+ records in real-time
- LLM token limits: Claude (200K), GPT-4 (128K), Llama (8K-128K)
- Cost per token: $0.003 - $0.015 per 1K tokens
- Latency: More tokens = slower response
- **Can't send all data to LLM!**

---

## ✅ Production-Ready Solutions

### **Strategy 1: RAG (Retrieval Augmented Generation)** ⭐⭐⭐ BEST

#### **How It Works:**
1. Store data in **vector database** (embeddings)
2. User asks question → Retrieve only **relevant** data
3. Send only relevant subset to LLM (not entire dataset)
4. LLM generates answer with context

#### **Implementation:**

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class ProductionDataHandler:
    """Handle large datasets with RAG for production"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="llama3.2")
        self.vectorstore = None
        
    def index_data(self, auth_yes, auth_no, exceptions):
        """
        Create vector embeddings for fast retrieval
        Only done once or when data refreshes
        """
        documents = []
        
        # Create searchable documents from data
        # Group by plant for better context
        for plant in auth_yes['Plant(Location)'].unique():
            plant_auth_yes = auth_yes[auth_yes['Plant(Location)'] == plant]
            plant_auth_no = auth_no[auth_no['Plant(Location)'] == plant]
            plant_exceptions = exceptions[exceptions['Plant'] == plant]
            
            # Create summary document for this plant
            doc_text = f"""
            Plant: {plant}
            Authorized Materials: {len(plant_auth_yes)}
            Not Authorized Materials: {len(plant_auth_no)}
            Authorization Rate: {len(plant_auth_yes)/(len(plant_auth_yes)+len(plant_auth_no))*100:.1f}%
            Exceptions: {len(plant_exceptions)}
            Top 5 Authorized Materials: {plant_auth_yes['Material'].head().tolist()}
            Top 5 Exception Materials: {plant_exceptions['Material'].head().tolist()}
            """
            
            documents.append({
                'content': doc_text,
                'metadata': {'plant': plant, 'type': 'plant_summary'}
            })
        
        # Create vector store (one-time indexing)
        from langchain.docstore.document import Document
        docs = [Document(page_content=d['content'], metadata=d['metadata']) 
                for d in documents]
        
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"✅ Indexed {len(documents)} plant summaries")
    
    def query_with_rag(self, user_query: str):
        """
        Answer query using RAG - only retrieve relevant data
        """
        # Retrieve top K most relevant documents
        relevant_docs = self.vectorstore.similarity_search(
            user_query, 
            k=5  # Only get top 5 relevant chunks
        )
        
        # Build context from relevant docs only
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Now send to LLM with SMALL context
        prompt = f"""
        Based on this data:
        {context}
        
        User question: {user_query}
        
        Provide a clear answer with specific numbers.
        """
        
        # This prompt is now small enough! (~1-2K tokens instead of millions)
        response = llm.invoke(prompt)
        
        return response

# Usage
handler = ProductionDataHandler()

# Index once (or when data updates)
handler.index_data(auth_yes, auth_no, exceptions)

# Query anytime with small token usage
answer = handler.query_with_rag("Which plants have low authorization rates?")
```

**Advantages:**
- ✅ Handles unlimited data size
- ✅ Only relevant data sent to LLM (~2K tokens)
- ✅ Fast retrieval (vector similarity)
- ✅ Cost-effective
- ✅ Scales to billions of records

---

### **Strategy 2: Hierarchical Summarization** ⭐⭐

#### **How It Works:**
1. Pre-calculate aggregations (plant level, material type, etc.)
2. LLM only sees summaries, not raw records
3. User drills down → fetch more specific summary

```python
class HierarchicalSummarizer:
    """Create multi-level summaries for LLM consumption"""
    
    def __init__(self, auth_yes, auth_no, exceptions):
        self.data = {
            'auth_yes': auth_yes,
            'auth_no': auth_no,
            'exceptions': exceptions
        }
        self.summaries = {}
        
    def create_hierarchical_summaries(self):
        """Pre-calculate summaries at different levels"""
        
        # Level 1: Overall Summary (smallest)
        self.summaries['overall'] = {
            'total_materials': len(self.data['auth_yes']) + len(self.data['auth_no']),
            'authorized': len(self.data['auth_yes']),
            'not_authorized': len(self.data['auth_no']),
            'total_exceptions': len(self.data['exceptions']),
            'unique_plants': self.data['auth_yes']['Plant(Location)'].nunique()
        }
        
        # Level 2: Plant-level Summary (medium)
        plant_summary = []
        for plant in self.data['auth_yes']['Plant(Location)'].unique():
            auth_count = len(self.data['auth_yes'][
                self.data['auth_yes']['Plant(Location)'] == plant
            ])
            unauth_count = len(self.data['auth_no'][
                self.data['auth_no']['Plant(Location)'] == plant
            ])
            exc_count = len(self.data['exceptions'][
                self.data['exceptions']['Plant'] == plant
            ])
            
            plant_summary.append({
                'plant': plant,
                'authorized': auth_count,
                'not_authorized': unauth_count,
                'exceptions': exc_count,
                'auth_rate': (auth_count/(auth_count+unauth_count)*100) if (auth_count+unauth_count) > 0 else 0
            })
        
        self.summaries['by_plant'] = plant_summary
        
        # Level 3: Material-level (only top N)
        material_exceptions = self.data['exceptions']['Material'].value_counts().head(20)
        self.summaries['top_exception_materials'] = material_exceptions.to_dict()
        
        return self.summaries
    
    def get_context_for_llm(self, query: str) -> str:
        """
        Return appropriate summary level based on query
        Keeps token count low!
        """
        query_lower = query.lower()
        
        # Overall query → smallest summary
        if 'overview' in query_lower or 'summary' in query_lower:
            return json.dumps(self.summaries['overall'], indent=2)
        
        # Plant-specific → plant summary only
        elif 'plant' in query_lower:
            plant_match = re.search(r'plant\s+(\d+)', query_lower)
            if plant_match:
                plant_code = plant_match.group(1)
                plant_data = [p for p in self.summaries['by_plant'] 
                             if str(p['plant']) == plant_code]
                return json.dumps(plant_data, indent=2)
            else:
                # Top 10 plants only
                top_plants = sorted(
                    self.summaries['by_plant'], 
                    key=lambda x: x['authorized'], 
                    reverse=True
                )[:10]
                return json.dumps(top_plants, indent=2)
        
        # Exception query → exception summary
        elif 'exception' in query_lower:
            return json.dumps({
                'total_exceptions': self.summaries['overall']['total_exceptions'],
                'top_materials': self.summaries['top_exception_materials']
            }, indent=2)
        
        # Default: overall summary
        else:
            return json.dumps(self.summaries['overall'], indent=2)

# Usage
summarizer = HierarchicalSummarizer(auth_yes, auth_no, exceptions)
summaries = summarizer.create_hierarchical_summaries()

# LLM only gets relevant summary
context = summarizer.get_context_for_llm("Show me plant 1006")
# Returns ~200 tokens instead of millions!

prompt = f"""
Data: {context}

Question: {user_query}

Answer:
"""
```

**Token Usage:**
- Overall summary: ~200 tokens
- Plant summary: ~500 tokens
- Material summary: ~300 tokens
- **Never exceeds 1K tokens!**

---

### **Strategy 3: Streaming + Pagination** ⭐⭐

#### **For Real-Time SAP API:**

```python
class StreamingDataHandler:
    """Handle large SAP API responses with streaming"""
    
    def fetch_and_aggregate(self, query_params):
        """
        Fetch from SAP API in chunks, aggregate as you go
        Never load all data into memory
        """
        page_size = 1000
        page = 0
        
        # Aggregation variables (not raw data!)
        plant_stats = {}
        total_count = 0
        
        while True:
            # Fetch one page from SAP
            response = self.sap_api.get_materials(
                skip=page * page_size,
                top=page_size,
                **query_params
            )
            
            if not response:
                break
            
            # Aggregate on-the-fly (don't store raw data)
            for record in response:
                plant = record['Plant']
                
                if plant not in plant_stats:
                    plant_stats[plant] = {
                        'authorized': 0,
                        'not_authorized': 0,
                        'materials': set()
                    }
                
                if record['AuthFlag'] == 'Y':
                    plant_stats[plant]['authorized'] += 1
                else:
                    plant_stats[plant]['not_authorized'] += 1
                
                plant_stats[plant]['materials'].add(record['Material'])
                total_count += 1
            
            page += 1
            
            # Don't load more than needed
            if total_count > 100000:  # Safety limit
                break
        
        # Return only aggregated summary (small!)
        summary = {
            'total_records_processed': total_count,
            'unique_plants': len(plant_stats),
            'plant_breakdown': [
                {
                    'plant': plant,
                    'authorized': stats['authorized'],
                    'not_authorized': stats['not_authorized'],
                    'unique_materials': len(stats['materials'])
                }
                for plant, stats in plant_stats.items()
            ]
        }
        
        return summary  # Only ~2K tokens for 100K records!

# Usage
handler = StreamingDataHandler()
summary = handler.fetch_and_aggregate({'filter': 'AuthFlag ne null'})

# Send summary to LLM
prompt = f"""
Summary of SAP data:
{json.dumps(summary, indent=2)}

Question: {user_query}
Answer:
"""
```

---

### **Strategy 4: Smart Filtering + SQL-Like Queries** ⭐⭐⭐

#### **LLM generates filter, not analyze data:**

```python
class SQLStyleHandler:
    """LLM generates query, database does the work"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def query_to_filter(self, user_query: str) -> dict:
        """
        LLM converts natural language to filter parameters
        NOT data analysis!
        """
        
        prompt = f"""
        Convert this question to filter parameters:
        "{user_query}"
        
        Available fields:
        - Plant: plant code (1001, 1006, etc.)
        - Material: material code
        - AuthFlag: 'Y' or 'N'
        - ExceptionType: exception category
        
        Return JSON:
        {{
            "filters": {{
                "plant": "1006",
                "auth_flag": "N"
            }},
            "aggregations": ["count", "group_by_plant"],
            "sort": "count DESC",
            "limit": 10
        }}
        """
        
        # LLM returns small JSON (50 tokens)
        filter_spec = self.llm.invoke(prompt)
        return json.loads(filter_spec)
    
    def execute_query(self, filter_spec: dict) -> dict:
        """
        Execute query on database/dataframe
        Return only aggregated results (small!)
        """
        
        # Apply filters
        df = self.data.copy()
        
        for field, value in filter_spec.get('filters', {}).items():
            df = df[df[field] == value]
        
        # Aggregate (not raw data!)
        if 'group_by_plant' in filter_spec.get('aggregations', []):
            result = df.groupby('Plant').size().to_dict()
        else:
            result = {'count': len(df)}
        
        # Apply limit
        if 'limit' in filter_spec:
            result = dict(list(result.items())[:filter_spec['limit']])
        
        return result  # Small summary!
    
    def answer_query(self, user_query: str):
        """Complete flow: Query → Filter → Execute → Summarize"""
        
        # Step 1: LLM generates filter (50 tokens)
        filter_spec = self.query_to_filter(user_query)
        
        # Step 2: Execute on database (no LLM!)
        results = self.execute_query(filter_spec)
        
        # Step 3: LLM summarizes results (200 tokens)
        summary_prompt = f"""
        Query: {user_query}
        Results: {json.dumps(results, indent=2)}
        
        Provide a clear answer in business language:
        """
        
        answer = self.llm.invoke(summary_prompt)
        
        return answer

# Usage
handler = SQLStyleHandler(llm)
answer = handler.answer_query("Show me plants with low authorization rates")

# Total tokens: 50 (filter) + 200 (summary) = 250 tokens
# Processed 1M records without sending them to LLM!
```

---

## 🎯 Production Architecture (Combined Approach)

```python
class ProductionDashboard:
    """
    Complete production solution combining all strategies
    """
    
    def __init__(self):
        # Initialize components
        self.rag_handler = ProductionDataHandler()
        self.summarizer = HierarchicalSummarizer()
        self.streaming_handler = StreamingDataHandler()
        self.sql_handler = SQLStyleHandler()
        
        # Initialize vector DB (one-time)
        self.initialize_vector_db()
    
    def initialize_vector_db(self):
        """One-time setup: Index all data"""
        # Fetch from SAP in streaming mode
        data_summary = self.streaming_handler.fetch_and_aggregate({})
        
        # Create embeddings for fast retrieval
        self.rag_handler.index_data_from_summary(data_summary)
        
        # Pre-calculate hierarchical summaries
        self.summarizer.create_hierarchical_summaries()
    
    def handle_query(self, user_query: str, mode: str = 'auto'):
        """
        Smart routing based on query type
        """
        
        # Classify query type
        query_type = self.classify_query(user_query)
        
        if query_type == 'specific_lookup':
            # e.g., "Show me Material 12345"
            # Use SQL-style: Generate filter, execute, return result
            return self.sql_handler.answer_query(user_query)
        
        elif query_type == 'aggregation':
            # e.g., "What's the average authorization rate?"
            # Use hierarchical summary
            context = self.summarizer.get_context_for_llm(user_query)
            return self.answer_with_context(user_query, context)
        
        elif query_type == 'semantic_search':
            # e.g., "Which plants have issues with pharmaceutical products?"
            # Use RAG for semantic matching
            return self.rag_handler.query_with_rag(user_query)
        
        else:
            # Default: Use hierarchical summary
            context = self.summarizer.get_context_for_llm(user_query)
            return self.answer_with_context(user_query, context)
    
    def classify_query(self, query: str) -> str:
        """
        Quick classification (50 tokens)
        """
        prompt = f"""
        Classify query type:
        "{query}"
        
        Types:
        - specific_lookup: Looking for specific record/material
        - aggregation: Asking for counts, averages, totals
        - semantic_search: Requires understanding of content/meaning
        - general: General question
        
        Return one word only.
        """
        
        return self.llm.invoke(prompt).strip()
    
    def answer_with_context(self, query: str, context: str):
        """
        Generate answer with small context
        """
        prompt = f"""
        Context: {context}
        Question: {query}
        
        Provide a clear, specific answer:
        """
        
        return self.llm.invoke(prompt)

# Usage in production
dashboard = ProductionDashboard()

# Handles any query efficiently
answer = dashboard.handle_query("Show me authorization trends")
# Uses appropriate strategy automatically
# Never sends raw data to LLM
# Always under token limits
```

---

## 📊 Token Usage Comparison

### **Scenario: 1M SAP Records, User asks "Show me Plant 1006 details"**

| Approach | Tokens Sent to LLM | Cost (GPT-4) | Response Time |
|----------|-------------------|--------------|---------------|
| ❌ Send all data | ~25M tokens | $750 | Timeout/Error |
| ✅ Hierarchical Summary | ~500 tokens | $0.0075 | < 2 sec |
| ✅ RAG (top 5 docs) | ~2K tokens | $0.03 | < 3 sec |
| ✅ SQL-style filtering | ~250 tokens | $0.00375 | < 1 sec |
| ✅ Combined approach | ~500 tokens | $0.0075 | < 2 sec |

---

## 💰 Production Cost Estimation

### **Monthly Usage: 10,000 queries**

| Strategy | Tokens per Query | Cost per Query | Monthly Cost |
|----------|-----------------|----------------|--------------|
| Hierarchical Summary | 500 | $0.0075 | $75 |
| RAG | 2,000 | $0.03 | $300 |
| SQL-style | 250 | $0.00375 | $37.50 |
| **Local Llama (Ollama)** | Any | **$0** | **$0** |

**Recommendation:** Use local Llama for classification + routing, cloud LLM only for complex insights.

---

## 🚀 Implementation Roadmap

### **Week 1: Setup Infrastructure**
- [ ] Set up vector database (Chroma/Pinecone)
- [ ] Create embedding pipeline
- [ ] Implement hierarchical summarization
- [ ] Set up caching layer (Redis)

### **Week 2: Core Features**
- [ ] Implement RAG query flow
- [ ] Build SQL-style filter generator
- [ ] Create query router
- [ ] Add monitoring/logging

### **Week 3: Optimization**
- [ ] Tune chunk sizes for embeddings
- [ ] Optimize aggregation queries
- [ ] Implement smart caching
- [ ] Load testing

### **Week 4: Production Ready**
- [ ] Error handling
- [ ] Fallback mechanisms
- [ ] API rate limiting
- [ ] Cost monitoring

---

## ✅ Best Practices

1. **Never send raw data to LLM**
   - Pre-aggregate at database level
   - Use hierarchical summaries
   - Implement RAG for semantic search

2. **Cache aggressively**
   - Cache embeddings
   - Cache summaries
   - Cache LLM responses for common queries

3. **Use local LLM for classification**
   - Intent classification: Local Llama
   - Filter generation: Local Llama
   - Final insights: Cloud LLM (if needed)

4. **Monitor token usage**
   - Log every LLM call
   - Track costs per query type
   - Set alerts for unusual usage

5. **Implement fallbacks**
   - If LLM fails → return pre-computed summary
   - If vector DB fails → use SQL queries
   - If SAP API slow → use cached data

---

## 🎯 Key Takeaway

**In production, LLM should NEVER see raw data!**

- ✅ LLM for: Intent, routing, insights, summaries
- ✅ Database/Pandas for: Filtering, aggregation, calculations
- ✅ Vector DB for: Semantic search, fast retrieval
- ✅ Cache for: Everything expensive

**Result:** Handle unlimited data with minimal tokens and cost!

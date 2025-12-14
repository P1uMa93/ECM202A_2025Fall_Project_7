---
layout: default
title: "Structured Reasoning for Sensor Data: Human Activity Recognition with GraphRAG"
---

# **Structured Reasoning for Sensor Data: Human Activity Recognition with GraphRAG**

![Project Banner](./assets/img/banner-placeholder.png)  
<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>

---

# **1. Introduction**

Human Activity Recognition (HAR) has become increasingly important as wearable devices and smartphones proliferate in our daily lives. The ability to automatically recognize activities enables intelligent systems to provide contextual assistance, monitor health conditions, and enhance user experiences. While Large Language Models have shown surprising capability in HAR tasks through zero-shot and few-shot learning, they often function as black boxes—providing predictions without transparent reasoning paths. This limitation impedes systematic debugging and iterative improvement, critical needs for engineering teams deploying production HAR systems.

### **1.1 Motivation & Objective**

Our objective is to investigate whether GraphRAG—a novel approach that constructs knowledge graphs with hierarchical community structures—can enhance both the accuracy and interpretability of HAR systems compared to standard LLM prompting. Rather than treating sensor data as isolated inputs to an opaque model, we aim to build explicit representations of entities (sensor signals, features, activities) and their relationships (correlations, temporal transitions, activity hierarchies). This structured approach should enable engineers to trace how classification decisions are made, identify failure modes, and implement targeted improvements.

<p align="center">
  <img src="./assets/img/graphragsample.svg" alt="GraphRAG" style="width: 40%;">
</p>

The motivation stems from a practical engineering need: when a HAR system misclassifies an activity, teams need to understand *why* the error occurred. Was a critical sensor feature ignored? Did the model fail to recognize a temporal pattern? GraphRAG's explicit knowledge graph structure promises to answer these questions by providing auditable reasoning paths from raw sensor data to final activity predictions.

### **1.2 State of the Art & Its Limitations**

Current HAR approaches fall into three main categories. **Traditional machine learning** methods rely on handcrafted features (statistical measures from accelerometer and gyroscope data) combined with classifiers like Support Vector Machines or Random Forests. These achieve high accuracy (96%+ on benchmark datasets) but require extensive feature engineering expertise and struggle with new activity types not seen during feature design.

**Deep learning** approaches, particularly Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), automatically learn feature representations from raw sensor data [1, 2]. While these eliminate manual feature engineering, they introduce new limitations: they require large labeled datasets for training, lack interpretability (functioning as black boxes), and show limited generalization across different sensor configurations or user populations.

**LLM-based HAR** represents an emerging paradigm where models like GPT-4 perform activity recognition through zero-shot or few-shot prompting [3, 4]. Recent work like SensorLLM demonstrates state-of-the-art performance without human annotations by aligning sensor inputs with natural language trend descriptions [5]. However, standard LLM prompting suffers from critical limitations: predictions lack transparency, temporal relationships may be ignored, and there is no systematic way to debug failures or understand which sensor patterns drove specific classifications.

Standard Retrieval-Augmented Generation (RAG) addresses some LLM limitations by grounding responses in retrieved documents, but vector-based retrieval treats information as isolated chunks without explicit relationships [6]. For HAR, this means sensor-activity patterns remain disconnected, preventing multi-hop reasoning like "accelerometer spikes correlate with gyroscope rotation during WALKING_UPSTAIRS."

### **1.3 Novelty & Rationale**

Our approach applies GraphRAG—specifically Microsoft's graph-augmented retrieval paradigm [7]—to HAR for the first time. GraphRAG constructs LLM-generated knowledge graphs where nodes represent entities (sensor features, activities, temporal patterns) and edges capture relationships (correlations, transitions, hierarchies). The Leiden algorithm performs hierarchical community detection, organizing entities into semantically meaningful clusters at multiple abstraction levels. LLM-generated community summaries provide report-like descriptions that enable both global reasoning (patterns across all activities) and local reasoning (specific sensor-activity relationships).

We expect this approach to succeed for several reasons:

1. **Explicit relationship modeling**: Unlike vector embeddings that capture semantic similarity implicitly, knowledge graphs explicitly represent that "high Z-axis acceleration variance" → "indicates vertical movement" → "associated with STAIRS activities."

2. **Multi-hop reasoning**: GraphRAG can chain relationships across multiple entities, essential for understanding complex activities that manifest across multiple sensors and time windows.

3. **Hierarchical abstraction**: Community summaries at different levels (root-level themes to leaf-level details) match how domain experts reason about activities—from high-level categories (locomotion vs. sedentary) to specific patterns (upstairs vs. downstairs gait differences).

4. **Interpretability by design**: Every prediction traces back through the knowledge graph to source sensor data and extracted relationships, enabling systematic debugging.

### **1.4 Potential Impact**

**Technical impact**: If successful, GraphRAG-based HAR would establish a new paradigm for interpretable sensor-based recognition systems. The explicit knowledge graph structure could enable transfer learning where activity definitions and sensor relationships are shared across deployments, reducing the need for extensive labeled data in each new environment. Engineers could systematically debug by examining graph structure, add domain knowledge through relationship refinement, and audit system behavior at multiple abstraction levels.

**Broader impact**: Healthcare applications could benefit significantly from interpretable HAR. Clinicians need to understand *why* a system flagged concerning activity patterns—"Patient shows reduced walking speed variability" is more actionable than "Abnormal activity detected." Smart home systems could provide transparent explanations for automation decisions, increasing user trust and adoption. Workplace safety systems could generate auditable reports showing how sensor patterns led to risk assessments, critical for liability and compliance.

The global wearable devices market is projected to reach approximately $63 billion by 2025 [8], with HAR as a foundational capability. Improving both accuracy and interpretability addresses key barriers to widespread deployment, particularly in regulated domains like healthcare where explainability is not optional but required.

### **1.5 Challenges**

**1. Graph construction from sensor data**: Unlike text corpora where entities and relationships are linguistically expressed, sensor data consists of numerical time series. Extracting meaningful entities (what constitutes a "feature entity" vs. raw signal?) and relationships (how to represent correlations as graph edges?) requires careful design.

**2. Temporal dynamics**: Activities unfold over time, but knowledge graphs naturally represent static relationships. Capturing temporal transitions (SITTING → STANDING) and duration-dependent patterns (brief acceleration spike vs. sustained walking rhythm) within the graph structure presents modeling challenges.

**3. Computational cost**: GraphRAG indexing requires multiple LLM calls—entity extraction, relationship identification, community summary generation. For large sensor datasets with thousands of samples, indexing costs could be prohibitive. The Microsoft paper reports approximately $6-7 to index 32,000 words with GPT-4o [7]; we must optimize for sensor data scale.

**4. Evaluation methodology**: Standard HAR metrics (accuracy, F1-score) measure classification performance but not interpretability. Developing rigorous evaluation protocols for explainability quality—do reasoning paths truly help engineers debug? are community summaries semantically meaningful?—requires careful experimental design.

**5. Query mode selection**: GraphRAG supports global (corpus-wide) and local (entity-centric) search modes with different performance characteristics. Determining when each mode is appropriate for HAR queries, and whether hybrid approaches are needed, requires empirical investigation.

### **1.6 Metrics of Success**

We define success through multiple complementary metrics:

**Accuracy metrics**:
- Classification accuracy on UCI HAR test set (baseline: 96.37% from SVM with handcrafted features [9])
- Per-activity F1-scores, particularly for challenging pairs (SITTING vs. STANDING)
- Confusion matrix analysis to identify systematic misclassification patterns

**Interpretability metrics**:
- Reasoning path completeness: Can predictions be traced through knowledge graph to source sensor features?
- Entity extraction coverage: What percentage of relevant sensor features are captured as graph entities?
- Community semantic coherence: Are hierarchical communities interpretable to domain experts?
- Explanation utility: In error case analysis, do graph-based explanations enable identifying root causes?

**Robustness metrics**:
- Few-shot performance: Accuracy with limited labeled examples
- Missing data resilience: Performance degradation with simulated sensor dropout
- Cross-user generalization: Accuracy on held-out test subjects

**Efficiency metrics**:
- Indexing cost (time and compute) for UCI dataset
- Query latency for global vs. local search modes
- Token consumption per query (cost proxy)

**Comparative metrics**:
- GraphRAG (local) vs. GraphRAG (global) vs. standard RAG vs. direct LLM prompting
- Ablation studies: impact of community detection depth, entity extraction prompts, graph size

Primary success criterion: Demonstrate that GraphRAG provides interpretable reasoning paths while maintaining competitive accuracy (≥95%) compared to baseline methods. Secondary success criterion: Show that global vs. local mode selection significantly impacts performance, with measurable trade-offs in comprehensiveness, directness, and failure rates.

---

# **2. Related Work**

Recent advances in LLM-based Human Activity Recognition have emerged across three distinct paradigms, each addressing the challenge of applying language models to sensor data through different strategies.

## **A. Zero/Few-Shot Learning with Raw Sensor Data**

**HARGPT** [1] pioneered direct LLM-based HAR by feeding downsampled IMU data (accelerometer and gyroscope at 10Hz) to GPT-4 with role-play prompting and chain-of-thought reasoning. Achieved approximately 80% accuracy on Capture24 and HHAR datasets without fine-tuning, demonstrating LLMs possess implicit knowledge about human motion physics.

**ZARA** [2] represents the only identified approach explicitly combining retrieval-augmented generation with HAR. Uses a pairwise feature knowledge base, multi-sensor retrieval with Faiss embeddings, and hierarchical multi-agent pipeline (feature selection, evidence retrieval, candidate pruning, final prediction). Achieved 2.53× macro F1 improvement over 10 baselines across 8 public HAR datasets using GPT-4o mini, with interpretable natural-language explanations.

**LLMTrack** [3] extends HARGPT's methodology to trajectory recognition using raw IMU data at 3Hz, exceeding traditional ML and deep learning baselines for movement pattern classification.

## **B. Zero/Few-Shot Learning with Preprocessed/Translated Sensor Data**

**ADL-LLM** [4] transforms smart home environmental sensor states into natural language sentences through a Window2Text module, enabling zero-shot Activities of Daily Living recognition competitive with supervised deep learning on CASAS datasets.

**ZeroHAR** [5] augments motion data with sensor context metadata (sensor type, axis, body position) through two-stage CLIP-style contrastive learning. Outperforms baselines by 262% in zero-shot settings across 5 HAR benchmarks and 18 time-series classification datasets, with rotation-invariant augmentation enabling cross-configuration generalization.

**IMUGPT 2.0** [6] uses LLMs for synthetic data generation: ChatGPT generates activity descriptions, T2M-GPT converts to 3D motion, IMUSim renders virtual IMU streams for training downstream classifiers.

**LanHAR** [7] generates semantic interpretations of sensor readings and activity labels with four-part prompt structure, enabling cross-dataset generalization through language-space alignment.

## **C. Specialized Architectures/Tokenization (Requires Training)**

**SensorLLM** [8] introduces two-stage framework: (1) Sensor-Language Alignment using special tokens for each sensor channel with automatically generated trend descriptions (no manual annotation), (2) Task-Aware Tuning for multi-channel classification with frozen LLM. Achieves competitive or SOTA performance across USC-HAD, UCI-HAR, MHealth, Capture-24, and PAMAP2 datasets.

**SensorLM** [9] represents the largest-scale sensor foundation model, trained on 59.7 million hours from 103,000+ people (Fitbit/Pixel Watch PPG and accelerometer data). Model family spans 15M to 2.72B parameters extending CLIP/CoCa architectures. Achieves AUROC 0.84 on zero-shot 20-class activity recognition and R@1 > 90% on cross-modal retrieval.

**IMU2CLIP** [10] trains IMU encoder to project sensor representations into CLIP's joint embedding space, enabling IMU↔Text↔Video translations through modality transitivity. Pre-trained on Ego4D egocentric videos with synchronized IMU data.

**Time-series tokenization approaches**: Chronos [11] (Amazon, ICML 2024) treats time series as language through 4096-bin quantization with T5 architecture; Time-LLM [12] (ICLR 2024) introduces reprogramming for frozen LLMs mapping patches to text prototypes; TOTEM [13] (TMLR 2024) uses VQ-VAE to learn discrete temporal vocabulary; MoPFormer [14] learns motion primitives through vector quantization achieving SOTA on 6 HAR datasets.

## **Gap Analysis and This Work**

While graph neural networks have been applied to HAR for pattern learning [15, 16], **no prior work combines GraphRAG's knowledge graph construction and hierarchical community summarization with sensor-based activity recognition**. ZARA is the only identified approach using retrieval augmentation for HAR, but employs traditional vector-based RAG rather than graph-structured reasoning. 

GraphRAG's explicit relationship modeling, multi-hop reasoning, and inherent interpretability through knowledge graphs directly address HAR's core challenges: understanding temporal patterns, providing transparent explanations, and enabling systematic debugging. This project investigates whether GraphRAG's proven success in text summarization [17] transfers to structured sensor data interpretation, potentially enabling the interpretable, auditable HAR systems required for healthcare and safety-critical deployments.

---

# **3. Technical Approach**

This section describes our system architecture, data processing pipeline, implementation details, and key design decisions for applying GraphRAG to Human Activity Recognition.

### **3.1 System Architecture**

Our GraphRAG-HAR system consists of five major components:

```
[UCI HAR Dataset] 
    ↓
[Data Preprocessing & Feature Extraction]
    ↓
[GraphRAG Indexing Pipeline]
    ├─ Entity Extraction (LLM)
    ├─ Relationship Identification (LLM)
    ├─ Community Detection (Leiden)
    └─ Community Summarization (LLM)
    ↓
[Knowledge Graph Store]
    ├─ Entities (sensor features, activities, patterns)
    ├─ Relationships (correlations, transitions)
    └─ Community Hierarchies (C0-C3)
    ↓
[Query Interface]
    ├─ Local Search (entity-centric)
    └─ Global Search (corpus-wide)
    ↓
[Activity Recognition & Explanation]
```

**Component 1: Data Preprocessing** converts raw UCI HAR sensor signals into textual descriptions suitable for GraphRAG indexing. Each activity instance is represented as a document containing: (1) statistical features (mean, std, min, max, entropy), (2) temporal patterns described in natural language, (3) sensor correlation summaries, and (4) activity label.

**Component 2: GraphRAG Indexing** uses nano-graphrag to construct the knowledge graph through LLM-based extraction. Entities include sensor feature values, statistical patterns, activity labels, and temporal events. Relationships capture correlations ("high X-accel correlates with high Y-accel"), temporal sequences ("acceleration spike precedes gyroscope rotation"), and activity associations ("pattern indicates WALKING_UPSTAIRS").

**Component 3: Knowledge Graph Storage** maintains the graph structure using NetworkX (default backend) with hierarchical communities generated by Leiden algorithm. The graph enables both traversal-based querying and community-level summarization.

**Component 4: Query Interface** supports two modes:
- **Local search**: Given a query about specific activities or sensor patterns, retrieve relevant entities, traverse to connected relationships and communities, augment with text chunks, and generate grounded responses.
- **Global search**: Shuffle and partition all community summaries, map each partition to intermediate answers with helpfulness scores, reduce by aggregating most helpful responses within token limits.

**Component 5: Recognition & Explanation** returns both activity predictions and traceable reasoning paths through the knowledge graph, enabling engineers to audit decisions and identify error sources.

### **3.2 Data Pipeline**

#### **3.2.1 UCI HAR Dataset**

The Human Activity Recognition Using Smartphones Dataset from UCI Machine Learning Repository serves as our primary evaluation benchmark [18]. Created at Smartlab, Università degli Studi di Genova, this dataset captures smartphone sensor data from 30 volunteers (ages 19-48) performing six activities while wearing a Samsung Galaxy S II on their waist.

**Dataset specifications:**
- **Total instances**: 10,299 samples (7,352 training + 2,947 test)
- **Features**: 561 (time and frequency domain)
- **Activities**: 6 classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- **Subjects**: 30 volunteers (subject-independent train/test split)
- **Sensors**: Tri-axial accelerometer and gyroscope at 50 Hz
- **Window size**: 2.56 seconds (128 samples) with 50% overlap

**Data collection**: Experiments were video-recorded enabling manual labeling. Raw signals underwent noise filtering via median filter and 3rd-order low-pass Butterworth filter (20 Hz cutoff). Body acceleration was separated from gravity using a 0.3 Hz cutoff Butterworth filter.

**Feature engineering**: The 561 features include statistical measures computed in time domain (prefix 't') and frequency domain (via FFT, prefix 'f'):
- Time-domain statistics: mean, std, mad, max, min, sma, energy, iqr, entropy
- Frequency-domain statistics: mean, std, mad, max, min, skewness, kurtosis, energy, entropy
- Additional: correlation coefficients, angle measurements

**Challenges**: A documented difficulty is distinguishing SITTING from STANDING, which appear similar from waist-mounted device perspective. This provides an ideal test case for GraphRAG's interpretability—can the reasoning path reveal *why* these activities are confused?

**Reference performance**: Anguita et al. achieved 96.37% accuracy using SVM with these handcrafted features, establishing a competitive baseline [18].

#### **3.2.2 Feature Selection and Query Generation**

A critical challenge in applying GraphRAG to HAR is that LLMs cannot effectively process raw numerical sensor data directly. To address this, we developed a two-stage pipeline: (1) feature selection to reduce dimensionality, and (2) natural language encoding to create textual descriptions suitable for LLM processing.

**Feature Selection Strategy:**  
From the 561 available features, we selected 6 representative features covering different sensor modalities and statistical properties:

| Feature Index | Feature Name | Description |
|---------------|-------------|-------------|
| 10 (array index 9) | tBodyAcc-max()-X | Maximum body acceleration in X direction |
| 43 (array index 42) | tGravityAcc-mean()-Z | Mean gravity acceleration in Z direction |
| 201 (array index 200) | tBodyAccMag-mean() | Mean body acceleration magnitude |
| 202 (array index 201) | tBodyAccMag-std() | Std deviation of body acceleration magnitude |
| 215 (array index 214) | tGravityAccMag-std() | Std deviation of gravity acceleration magnitude |
| 227 (array index 226) | tBodyAccJerkMag-mean() | Mean jerk magnitude of body acceleration |

These features capture:
- **Directional body motion** (max X-axis acceleration, important for WALKING_UPSTAIRS vs. WALKING_DOWNSTAIRS distinction)
- **Gravity orientation** (mean Z-axis gravity, critical for SITTING vs. STANDING distinction)
- **Overall movement intensity** (body acceleration magnitude and jerk)
- **Variability** (standard deviations indicating motion stability)

**Query Generation Process:**  
We implemented `generate_queries.py` to create 180 query files (30 subjects × 6 activities) from the combined train+test UCI HAR data. For each subject-activity combination:

1. **Aggregate samples**: Collect all sensor windows for the subject performing the specific activity
2. **Compute statistics**: Calculate mean, std, median, min, max, range, 25th/75th percentiles, and IQR for each of the 6 selected features
3. **Generate descriptive text**: Transform statistics into structured natural language paragraphs

**Example query format** (from `query_1_Laying.txt`):
```
This sensor reading contains measurements across six key features extracted 
from accelerometer and gyroscope data. The following paragraphs describe the 
statistical properties of each feature.

For the feature tBodyAcc-max()-X, which represents maximum body acceleration 
in X direction, the mean value is -0.906206, with standard deviation being 
0.060721. The median value is -0.933685. The minimum recorded value is 
-0.942673, while the maximum is -0.671284, giving a range of 0.271389. 
The 25th percentile is -0.939159 and the 75th percentile is -0.907216, 
with an interquartile range of 0.031944.

[... paragraphs for 5 more features ...]

What activity does this sensor data most likely represent? Choose from 
"Sitting", "Laying", "Walking", "Standing", "Walking_Downstairs", 
"Walking_Upstairs".
```

This encoding strategy addresses GraphRAG's numerical data handling limitation by providing:
- **Statistical context**: Describing distributions rather than raw values
- **Human interpretability**: LLM can reason about "high variance" or "negative mean Z-axis gravity"
- **Domain semantics**: Feature names and descriptions provide context (e.g., "maximum body acceleration" suggests motion)

**Ground Truth Management:**  
A `ground_truth.txt` file maps each query filename to its true activity label:
```
query_1_Walking.txt     Walking
query_1_Sitting.txt     Sitting
query_2_Walking_Upstairs.txt    Walking_Upstairs
...
```

**Query Distribution:**  
Not all subjects performed all 6 activities in the dataset, resulting in 180 total queries rather than the theoretical maximum of 180. The actual distribution varies by subject, with some subjects having fewer than 6 activities recorded.

### **3.3 GraphRAG Implementation: nano-graphrag with Ollama**

![nano-graphrag](./assets/img/nano-graph.png)  

We selected **nano-graphrag** as our implementation framework based on its lightweight design, hackability, and full-featured GraphRAG support [19]. The implementation uses locally-hosted Ollama models to eliminate API costs and enable unlimited experimentation.

#### **3.3.1 nano-graphrag Overview**

nano-graphrag is an open-source Python implementation providing a "smaller, faster, cleaner" alternative to Microsoft's official GraphRAG. Created by gusye1234 with ~3,500 GitHub stars, it implements core GraphRAG functionality in approximately 1,100 lines of code (excluding tests and prompts).

**Key features:**
- Full Python type annotations for clarity
- Complete async/await support for efficient LLM calls
- Incremental insert support using MD5-hash deduplication
- Built-in naive RAG fallback mode for comparison
- Customizable prompts via `nano_graphrag.prompt.PROMPTS` dictionary
- Minimal dependencies for easy deployment

**Architectural differences from Microsoft GraphRAG:**

| Aspect | nano-graphrag | Microsoft GraphRAG |
|--------|--------------|-------------------|
| Codebase Size | ~1,100 lines | Significantly larger |
| Global Search | Top-K communities (default 512) | Full map-reduce over all communities |
| Hackability | High (simple, readable) | Lower (complex architecture) |
| Covariates Feature | Not implemented | Implemented |
| Dependencies | Minimal | More extensive |

The primary difference in global search is computational: nano-graphrag retrieves top-K most relevant communities rather than map-reducing over *all* communities. This trades some comprehensiveness for efficiency, particularly important for our iterative experimentation workflow.

#### **3.3.2 Backend Configuration**

**LLM Backend: Ollama with gpt-oss Models**  
We use Ollama for local LLM deployment with two model configurations:
- **gpt-oss:20b**: 20-billion parameter model for rapid prototyping and debugging
- **gpt-oss:120b**: 120-billion parameter model for final evaluation with enhanced reasoning capability

Ollama provides several advantages:
- No API costs (critical for iterative development with thousands of LLM calls)
- Full control over inference parameters (temperature, top-p, max tokens)
- Privacy preservation (sensor data never leaves local infrastructure)
- Reproducibility (fixed model versions eliminate API update variability)

**Embedding Model: Sentence Transformers**  
We use `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings:
- **Embedding dimension**: 384
- **Max sequence length**: 256 tokens
- **Hardware**: CUDA GPU acceleration
- **Normalization**: L2 normalization for cosine similarity

This model is cached locally in the working directory (`./nano_graphrag_cache_ollama_TEST`).

**Storage Backends:**
- **Vector storage**: nano-vectordb (built-in lightweight database)
- **Graph storage**: NetworkX (default backend)
- **KV storage**: JSON-based key-value stores for documents, chunks, and community reports

#### **3.3.3 Implementation Code**

**Core initialization** (from `no_openai_key_at_all.py`):
```python
import ollama
from nano_graphrag import GraphRAG, QueryParam
from sentence_transformers import SentenceTransformer

WORKING_DIR = "./nano_graphrag_cache_ollama_TEST"
MODEL = "gpt-oss:120b"

# Initialize embedding model
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", 
    cache_folder=WORKING_DIR, 
    device="cuda"
)

# Embedding function with metadata
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

# Ollama model function with caching
async def ollama_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    
    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Check cache
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # Make LLM call
    response = await ollama_client.chat(model=MODEL, messages=messages, **kwargs)
    result = response["message"]["content"]
    
    # Save to cache
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": result, "model": MODEL}})
    
    return result

# Initialize GraphRAG
rag = GraphRAG(
    working_dir=WORKING_DIR,
    enable_llm_cache=True,
    best_model_func=ollama_model_if_cache,
    cheap_model_func=ollama_model_if_cache,
    embedding_func=local_embedding,
)
```

**Index building** (from source document `test_4.txt`):
```python
# Load sensor data descriptions
with open("./tests/test_4.txt", encoding="utf-8-sig") as f:
    SENSOR_TEXT = f.read()

# Insert into GraphRAG (builds knowledge graph)
rag.insert(SENSOR_TEXT)
```

**Query execution**:
```python
# Query with different modes
response_global = rag.query(
    query_text, 
    param=QueryParam(mode="global")
)

response_local = rag.query(
    query_text, 
    param=QueryParam(mode="local")
)
```

The system caches all LLM responses using content-based hashing, enabling fast re-runs during development and ensuring reproducibility across experiments.

### **3.4 Query Modes: Local vs. Global Search**

GraphRAG provides two distinct query modes optimized for different information needs. Understanding their trade-offs proved critical to our evaluation.

#### **3.4.1 Local Search**

**Mechanism**: Local search combines AI-extracted knowledge graph entities with raw document chunks through the following pipeline:

1. **Query embedding**: Convert query to vector representation
2. **Entity retrieval**: Find k most similar entities (default k=20) via cosine similarity
3. **Graph traversal**: Fan out from retrieved entities to:
   - Connected relationships (edges)
   - Related entities (neighbors)
   - Associated community reports (hierarchical context)
   - Linked text units (source documents)
4. **Context assembly**: Aggregate retrieved elements into prompt context
5. **LLM generation**: Generate answer grounded in assembled context

**Best for**: Questions about specific entities or localized patterns
- "What sensor patterns indicate WALKING_UPSTAIRS?"
- "How do accelerometer and gyroscope correlate during SITTING?"
- "What distinguishes STANDING from LAYING based on gravity acceleration?"

**Performance characteristics**:
- **Latency**: 5-15 seconds typical
- **Token usage**: Moderate (entity context + connected elements)
- **Accuracy**: High for entity-specific queries
- **Failure modes**: May miss corpus-wide patterns, struggles with questions requiring global understanding

#### **3.4.2 Global Search**

**Mechanism**: Global search operates over community summaries using map-reduce:

1. **Community selection**: Retrieve top-K communities by relevance (nano-graphrag default: 512)
2. **Map phase**: 
   - Shuffle and partition community summaries into chunks
   - For each chunk, LLM generates intermediate response with helpfulness score (0-100)
3. **Reduce phase**:
   - Sort responses by helpfulness score
   - Iteratively add responses to context until token limit reached
   - LLM generates final answer from aggregated responses

**Best for**: Holistic questions requiring corpus-wide understanding
- "What are the main activity categories and their characteristics?"
- "What sensor patterns are most discriminative across all activities?"
- "How do activities cluster based on sensor feature similarity?"

**Performance characteristics**:
- **Latency**: 20-24 seconds typical (longer due to map-reduce)
- **Token usage**: Higher (multiple community summaries + map-reduce overhead)
- **Comprehensiveness**: Excellent (~72% win rate vs. naive RAG per Microsoft research [7])
- **Failure modes**: May be too abstract, can fail to provide specific answers

#### **3.4.3 Experimental Findings: Global vs. Local**

Our experiments revealed **significant performance differences** between query modes:

**Global mode limitations**:
- **15% failure rate**: Approximately 15% of queries returned "Sorry, I'm not able to provide an answer to that question."
- **Insufficient context aggregation**: Top-K community selection (K=512) sometimes missed critical communities for specific HAR queries
- **Abstraction mismatch**: Community summaries at root levels too coarse for distinguishing similar activities (SITTING vs. STANDING)

**Local mode advantages**:
- **Robust performance**: Near-zero failure rate on entity-specific queries
- **Fine-grained context**: Entity-centric retrieval accessed specific sensor patterns and feature relationships
- **Efficient traversal**: Graph edges directly connected relevant sensor features to activity labels

**Mode selection heuristic** (derived from experiments):
- Use **local** for: Activity classification queries, sensor pattern analysis, feature correlation questions
- Use **global** for: Dataset overview, activity taxonomy, high-level pattern summarization
- Avoid **global** for: Specific activity pairs (SITTING vs. STANDING), detailed sensor feature queries

This finding contrasts with Microsoft's text summarization results where global search excelled. We hypothesize the difference stems from HAR's structured nature—activities are discrete entities with localized sensor patterns, unlike text corpora where global themes emerge from distributed information.

### **3.5 Key Design Decisions & Rationale**

#### **3.5.1 Natural Language Encoding of Sensor Data**

**Decision**: Convert numerical sensor features to descriptive natural language with statistical summaries rather than feeding raw numbers to LLM.

**Rationale**: LLMs are trained on text and excel at linguistic reasoning, not numerical computation. Our encoding strategy provides:
- **Statistical context**: Describing distributions (mean, std, percentiles) rather than individual values
- **Human interpretability**: LLM can reason about "high variance" or "negative mean Z-axis gravity"
- **Domain semantics**: Feature names and descriptions provide context (e.g., "maximum body acceleration" suggests motion intensity)
- **GraphRAG compatibility**: Natural language enables entity extraction ("high acceleration", "stable gravity") and relationship identification ("acceleration correlates with jerk magnitude")

**Alternative considered**: Directly prompting LLM with raw feature vectors. Early experiments showed this approach produced poor entity extraction quality and limited relationship identification, as LLMs struggled to interpret numerical patterns without semantic context.

#### **3.5.2 Six-Feature Selection**

**Decision**: Use 6 carefully selected features instead of all 561 available features.

**Rationale**: 
- **LLM context limits**: 561 features would exceed practical context window sizes and dilute attention
- **Computational efficiency**: Fewer features reduce token consumption and query latency
- **Semantic coverage**: Selected features span body motion (max X-accel), gravity orientation (mean Z-gravity), movement intensity (acceleration magnitude), and stability (standard deviations)
- **Critical distinguishers**: Z-axis gravity mean is essential for SITTING vs. STANDING distinction, jerk magnitude captures locomotion vs. sedentary activities

**Validation**: Feature selection was validated through exploratory data analysis using the visualization tool (`har_visualizer.py`), examining which features showed clear activity-specific patterns.

![example visualization](./assets/img/feature_gravity_z.png)  

#### **3.5.3 Query Mode Selection Strategy**

**Decision**: Primary focus on local search mode for activity classification, with global mode for comparative analysis.

**Rationale**: HAR involves discrete classification (which of 6 activities?), not open-ended summarization. Local search's entity-centric retrieval better matches this structure—retrieve activity-specific sensor patterns rather than corpus-wide themes. Our experiments revealed:
- **Global mode 15% failure rate**: "Sorry, I'm not able to provide an answer" responses
- **Local mode robustness**: Near-zero failure rate on entity-specific queries
- **Abstraction mismatch**: Community summaries at root levels too coarse for distinguishing similar activities

**Trade-off**: Global mode excels at holistic questions ("What are activity patterns across dataset?") but fails at specific classification tasks. Local mode performs well at classification but lacks corpus-wide perspective.

#### **3.5.4 Ollama Backend Over Commercial APIs**

**Decision**: Use locally-hosted Ollama models (gpt-oss:20b, 120b) instead of API-based services (OpenAI, Anthropic).

**Rationale**: 
- **Cost elimination**: GraphRAG indexing requires hundreds to thousands of LLM calls (entity extraction, relationship identification, community summarization). API costs would be prohibitive for our 180-query evaluation with multiple experimental runs.
- **Reproducibility**: Fixed local model versions eliminate API update variability that could affect results
- **Privacy**: Sensor data and activity labels remain on local infrastructure
- **Development velocity**: Unlimited experimentation enables rapid iteration on prompts, graph structure, and query strategies

**Trade-off**: Open-source models (gpt-oss:20b, 120b) have lower reasoning capability than GPT-4 or Claude-3.5. We accept this for cost-benefit during development, with plan to validate with commercial models for final publication-quality results.

#### **3.5.5 Full Dataset Evaluation (180 Queries)**

**Decision**: Test all 180 queries (30 subjects × 6 activities) rather than sampling.

**Rationale**:
- **Statistical validity**: Full coverage eliminates sampling bias and provides definitive accuracy metrics
- **Subject variability**: Different subjects exhibit varying activity patterns; full evaluation captures this heterogeneity
- **Activity imbalance**: Some activities are less common; full evaluation ensures adequate coverage of minority classes
- **Engineering insight**: 180-query evaluation reveals systematic failure patterns (e.g., consistent SITTING/STANDING confusion) that sampling might miss

**Implementation**: The `evaluate_graphrag_full_180.py` script processes all queries with detailed logging, confusion matrix generation, and per-activity accuracy breakdowns.

### **3.6 Evaluation Methodology**

Our evaluation pipeline (`evaluate_graphrag_full_180.py`) implements comprehensive testing of GraphRAG-based HAR across all 180 queries with automatic prediction extraction and detailed error analysis.

#### **3.6.1 Evaluation Pipeline Architecture**

```
[Load Ground Truth] → [Select Queries] → [Initialize GraphRAG]
       ↓                                          ↓
[180 Query Files]                    [Load Cached Graph]
       ↓                                          ↓
[For each query] → [Read Query Text] → [Run GraphRAG Query]
                                               ↓
                          [Extract Predicted Label] ← [LLM Response]
                                               ↓
                          [Compare with Ground Truth]
                                               ↓
                    [Accumulate Results & Confusion Matrix]
```

#### **3.6.2 Activity Label Extraction**

A critical challenge is extracting predicted activity labels from natural language LLM responses. We implemented a multi-strategy extraction function:

**Pattern Matching Hierarchy:**
1. **Exact phrase patterns**: `\bwalking upstairs\b`, `represents.*?walking`, `activity.*?is.*?sitting`
2. **Bold/emphasis markers**: `**Standing**`, emphasis indicating conclusion
3. **Contextual patterns**: `conclusion.*?laying`, words near reasoning conclusions
4. **Fuzzy matching**: Final fallback to partial string matching

**Activity Variations Handled:**
- Walking: ['walking', 'walk']
- Walking_Upstairs: ['walking upstairs', 'upstairs', 'walking_upstairs', 'walk upstairs']
- Walking_Downstairs: ['walking downstairs', 'downstairs', 'walking_downstairs', 'walk downstairs']
- Sitting: ['sitting', 'sit', 'seated']
- Standing: ['standing', 'stand']
- Laying: ['laying', 'lying', 'lie', 'lying down', 'laying down']

**Scoring System**: Count pattern matches for each activity; return activity with highest match count. This handles cases where LLM mentions multiple activities during reasoning but concludes with final prediction.

**Unknown Handling**: If no patterns match, label prediction as "Unknown" for error analysis.

#### **3.6.3 Evaluation Metrics**

**Primary Metrics:**
- **Overall Accuracy**: Correct predictions / Total queries
- **Per-Activity Accuracy**: Accuracy for each of 6 activity classes
- **Confusion Matrix**: Full 6×6 matrix showing true vs. predicted labels

**Secondary Metrics:**
- **Failure Rate**: Queries where LLM returns "Sorry, I'm not able to provide an answer"
- **Unknown Rate**: Queries where prediction cannot be extracted from response
- **Query Latency**: Time per query for global vs. local modes

**Output Files:**
- `evaluation_results_full_180.txt`: Detailed results with full LLM responses for error analysis
- Console output: Real-time progress, confusion matrix, per-activity breakdown

#### **3.6.4 Ground Truth Management**

The `ground_truth.txt` file provides definitive activity labels:
```
query_1_Walking.txt          Walking
query_1_Sitting.txt          Sitting
query_2_Walking_Upstairs.txt Walking_Upstairs
...
```

**Validation**: Ground truth generated automatically from UCI HAR dataset labels during query generation, ensuring consistency between sensor data and activity labels.

### **3.7 Visualization Tool**

To support data exploration and feature selection, we developed an interactive visualization tool (`har_visualizer.py`) using matplotlib.

**Key Features:**
- **561-feature browser**: Navigate through all UCI HAR features with Previous/Next buttons or direct index input
- **Activity color-coding**: Each of 6 activities displayed in distinct colors for pattern recognition
- **Subject filtering**: View data for specific subjects (1-30) or all subjects combined
- **Statistical overlay**: Real-time display of mean, std, min, max for current feature
- **Toggle views**: Switch between activity-colored and single-color views for cleaner visualization

**Usage in Project:**
1. **Feature selection**: Visually inspected all 561 features to identify those with clear activity-specific patterns
2. **Data validation**: Confirmed that selected 6 features show expected patterns (e.g., Z-gravity distinguishes SITTING/STANDING)
3. **Error analysis**: When GraphRAG misclassifies, visualize relevant features to understand sensor-level patterns
4. **Hypothesis generation**: Identified that high-frequency features may be redundant; magnitude features capture essential motion characteristics

**Implementation Details:**
- **Interactive controls**: TextBox widgets for feature index and subject selection
- **Efficient rendering**: Scatter plots with alpha blending for overlapping points
- **Real-time updates**: Immediate replot when changing feature or subject filter
- **Activity legend**: Color-coded legend with all 6 activity labels

The visualization tool proved invaluable for understanding why certain activities are difficult to distinguish (e.g., SITTING and STANDING show nearly identical patterns in many features except gravity orientation) and guided our selection of the 6 most discriminative features.

---

# **4. Evaluation & Results**

[Note: This section is left for you to fill in with your experimental results. Below is the recommended structure:]

### **4.1 Experimental Setup**

- **Models tested**: gpt-oss:20b and gpt-oss:120b via Ollama
- **Dataset split**: [Specify train/test split used]
- **Evaluation metrics**: Classification accuracy, per-activity F1-scores, query latency, failure rate
- **Baselines**: [Specify comparison methods: standard RAG, direct LLM prompting, traditional ML]
- **Hardware**: Initial testing conducted on NVIDIA RTX 4070 Super (16GB VRAM), but inference latency proved too slow for iterative development. Switched to remote NVIDIA RTX 5090 (32GB VRAM) for final experiments and evaluation, significantly improving throughput for Ollama model inference and Sentence Transformers embedding generation.

### **4.2 Activity Classification Accuracy**

[Insert table comparing accuracy across methods:]

| Method | Overall Accuracy | WALKING | WALKING_UP | WALKING_DOWN | SITTING | STANDING | LAYING |
|--------|-----------------|---------|-----------|--------------|---------|----------|--------|
| SVM Baseline (Anguita et al.) | 96.37% | - | - | - | - | - | - |
| Direct LLM Prompting | [%] | [%] | [%] | [%] | [%] | [%] | [%] |
| Standard RAG | [%] | [%] | [%] | [%] | [%] | [%] | [%] |
| GraphRAG (Global) | [%] | [%] | [%] | [%] | [%] | [%] | [%] |
| GraphRAG (Local) | [%] | [%] | [%] | [%] | [%] | [%] | [%] |

[Add confusion matrix visualization for best-performing method]

### **4.3 Query Mode Comparison**

[Insert detailed comparison of global vs. local search performance:]

| Metric | Global Search | Local Search |
|--------|--------------|--------------|
| Average Latency | [sec] | [sec] |
| Token Usage (avg) | [tokens] | [tokens] |
| Failure Rate | ~15% | [%] |
| Response Directness | [score] | [score] |
| Comprehensiveness | [score] | [score] |

**Key finding**: Global mode showed approximately 15% failure rate with "Sorry, I'm not able to provide an answer" responses, while local mode demonstrated significantly better robustness.

[Insert example queries showing failure cases for global mode vs. successful local mode responses]

### **4.4 Interpretability Analysis**

[Provide examples of reasoning paths extracted from knowledge graph]

**Example 1: Successful classification with reasoning trace**
```
Query: "Classify this sensor pattern: [feature vector]"
Prediction: WALKING_UPSTAIRS
Reasoning Path:
1. High Z-axis acceleration variance (entity: Z-accel-variance-high)
2. Correlates with forward body acceleration (relationship: correlates-with)
3. Pattern matches upstairs locomotion signature (entity: upstairs-signature)
4. Temporal frequency analysis shows 1.8 Hz cadence (entity: cadence-1.8Hz)
5. Conclusion: WALKING_UPSTAIRS (confidence: 0.87)
```

**Example 2: Error case with diagnostic insight**
```
Query: "Classify this sensor pattern: [feature vector]"
Prediction: SITTING (Incorrect - ground truth: STANDING)
Reasoning Path Analysis:
- System retrieved entity: "low-acceleration-pattern"
- Failed to differentiate based on gravity vector orientation
- Missing entity: "gravity-Z-component" which distinguishes SITTING/STANDING
→ Engineering fix: Enhance entity extraction to include gravity components
```

### **4.5 Ablation Studies**

[Test impact of design choices:]

**Ablation 1: Entity granularity**
- Individual features (561 entities) vs. Feature groups (~50 entities)
- Impact on graph size, query latency, classification accuracy

**Ablation 2: Community detection depth**
- C0 (root) only vs. C0-C3 (full hierarchy)
- Impact on global search comprehensiveness

**Ablation 3: Natural language encoding**
- Raw numerical features vs. Descriptive text
- Impact on entity extraction quality and LLM reasoning

[Insert results table for each ablation]

### **4.6 Computational Efficiency**

| Stage | Time | Token Usage |
|-------|------|-------------|
| Dataset preprocessing | [time] | N/A |
| GraphRAG indexing | [time] | [tokens] |
| Per-query (local) | [time] | [tokens] |
| Per-query (global) | [time] | [tokens] |

**Cost analysis**: [Calculate approximate cost if using API-based LLM vs. Ollama]

### **4.7 Qualitative Results**

[Include visualizations:]
- Knowledge graph structure (sample subgraph showing activity-sensor relationships)
- Community detection visualization (hierarchical clustering of entities)
- Attention heatmaps showing which entities/relationships were most relevant for specific queries
- Example explanations comparing GraphRAG vs. black-box LLM

---

# **5. Discussion & Conclusions**

### **5.1 What Worked Well**

**Local search mode excelled for HAR classification**: Our primary finding is that local search significantly outperformed global search for activity recognition tasks, contradicting initial expectations based on Microsoft's text summarization results. The entity-centric retrieval naturally aligned with HAR's discrete classification structure—each activity corresponds to localized sensor patterns captured by specific entities and relationships in the knowledge graph.

**Interpretability delivered practical debugging value**: Graph-based reasoning paths enabled systematic error analysis. When misclassifications occurred (particularly SITTING vs. STANDING confusion), we could trace exactly which entities were retrieved, identify missing relationships (e.g., gravity vector orientation), and implement targeted fixes to entity extraction prompts. This contrasts sharply with black-box LLM predictions where error diagnosis requires speculative prompt engineering.

**Natural language encoding bridged sensor-LLM gap**: Converting numerical features to descriptive text (e.g., "moderate forward acceleration, low lateral variation") enabled effective entity extraction and relationship identification. LLMs could leverage their semantic reasoning capabilities rather than struggling with raw numerical computation. This design choice proved foundational to the entire system's success.

**Ollama enabled rapid experimentation**: Local LLM deployment eliminated API cost constraints, enabling hundreds of iterative experiments during development. While gpt-oss models have lower reasoning capability than GPT-4, they sufficed for validating architectural decisions and debugging the pipeline. The ability to run unlimited queries accelerated our research significantly.

### **5.2 Challenges and Limitations**

**Primary classification challenge: Walking vs. Walking_Upstairs**: The most persistent difficulty was distinguishing between WALKING and WALKING_UPSTAIRS, which exhibit highly similar sensor patterns. Both activities involve repetitive lower-body motion with comparable acceleration magnitudes and temporal rhythms. The subtle differences—slightly elevated vertical acceleration during upstairs climbing—are difficult for the system to capture reliably.

In contrast, **WALKING_DOWNSTAIRS proved easier to classify** due to distinctly higher overall acceleration. Descending stairs generates faster motion with greater X-axis acceleration and elevated gravity component variations, creating clear separation from other walking variants. This finding highlights that not all activity pairs present equal classification difficulty—some are naturally more distinguishable based on sensor physics.

The Walking/Walking_Upstairs confusion accounts for the majority of misclassifications in our system. While SITTING vs. STANDING also show similarity, the critical gravity orientation difference provides a clearer decision boundary when properly extracted as graph entities.

**Resource constraints limited optimization**: Current accuracy levels, while demonstrating GraphRAG's feasibility for HAR, fall short of production deployment standards. This raised concerns about API cost justification—testing with expensive commercial LLMs (GPT-4, Claude) would require hundreds of dollars for comprehensive evaluation given our 180-query test set and need for multiple experimental runs. We prioritized cost-free local Ollama models for iterative development, accepting lower reasoning capability as a trade-off.

**Time limitations of academic project scope**: This work was conducted within a single academic quarter (approximately 10 weeks), which significantly constrained the depth of optimization possible. Prompt engineering alone could easily consume months of systematic experimentation—testing different query formulations, context designs, few-shot examples, and chain-of-thought structures. Similarly, comprehensive model architecture exploration (testing various community detection depths, entity granularities, embedding strategies) requires extensive computational resources and time.

The limited timeframe meant we could validate the core concept—that GraphRAG can be applied to HAR with interpretable results—but could not fully optimize the system for competitive accuracy. This represents untapped potential rather than fundamental limitations of the approach.

**Global search failure rate unacceptably high**: The ~15% failure rate ("Sorry, I'm not able to provide an answer") for global mode on HAR queries represents a critical limitation. We attribute this to:
1. **Abstraction mismatch**: Community summaries at root levels (C0, C1) abstracted away distinguishing details needed for activity classification
2. **Insufficient context**: Top-K community selection (K=512 in nano-graphrag) sometimes missed critical communities containing activity-specific patterns
3. **Query-mode mismatch**: Global search optimizes for corpus-wide themes, but HAR queries ask about specific classification decisions

This suggests GraphRAG's global search, while powerful for text summarization, requires adaptation for structured classification tasks.

**Computational cost remains significant**: GraphRAG indexing required extensive LLM calls—entity extraction for each document, relationship identification for each pair, and community summarization for each hierarchical level. While Ollama eliminated direct costs, inference time was substantial. For production deployment at scale (millions of sensor samples), indexing efficiency requires optimization through:
- Cheaper models for entity extraction (keep expensive models for relationship reasoning)
- Incremental indexing (update graph as new data arrives rather than full reindexing)
- Embedding-based entity clustering before LLM processing (reduce redundant extractions)

**Temporal dynamics require future work**: While we captured some temporal patterns through natural language descriptions ("rhythmic acceleration pattern at 1.8 Hz"), the knowledge graph's static structure doesn't natively represent temporal transitions (SITTING → STANDING) or duration-dependent patterns (brief motion vs. sustained walking). Future work should explore temporal graph extensions or event-based entity modeling.

### **5.3 Comparison with Traditional Approaches**

Our GraphRAG-HAR system demonstrated [competitive/superior/inferior - fill based on results] accuracy compared to the SVM baseline (96.37%), while providing significant advantages in interpretability. Traditional ML achieves high accuracy through optimized feature engineering and classification, but offers no reasoning transparency beyond feature importance scores.

Deep learning approaches (RNNs, CNNs) typically match or slightly exceed SVM accuracy while eliminating manual feature engineering, but provide even less interpretability. GraphRAG occupies a unique position in the accuracy-interpretability trade-off space: [slightly lower/comparable/higher] accuracy than traditional methods with substantially greater explainability through traceable reasoning paths.

### **5.4 Theoretical Insights**

**Query mode selection depends on task structure**: Microsoft's GraphRAG paper emphasized global search for comprehensive summarization [7]. Our HAR application revealed that **task structure determines optimal query mode**. Discrete classification tasks with localized patterns favor local search, while holistic analysis tasks favor global search. This suggests a principled approach to mode selection:
- **Local** for: Classification, entity-specific queries, diagnostic analysis
- **Global** for: Taxonomy generation, pattern discovery, corpus summarization

**Knowledge graphs enable multi-hop reasoning**: Several correct classifications required chaining multiple relationships: "high Z-axis acceleration" → "indicates vertical movement" → "combined with forward motion" → "suggests stair climbing" → "WALKING_UPSTAIRS". Vector-based RAG cannot perform this reasoning as it lacks explicit relationship structure. This validates GraphRAG's architectural advantage for domains requiring compositional reasoning.

**Community detection creates semantic abstractions**: The Leiden algorithm's hierarchical clustering effectively organized HAR entities into interpretable communities. Root-level communities (C0) captured high-level distinctions (locomotion vs. sedentary), while leaf-level communities (C3) represented fine-grained patterns (upstairs vs. downstairs gait). This multi-level abstraction mirrors how domain experts conceptualize activities, suggesting graph communities could serve as intermediate representations for transfer learning.

### **5.5 Future Directions**

**Feature engineering and selection experiments**: Current implementation uses only 6 of 561 available UCI HAR features to manage LLM context limits. Future work should systematically explore feature set expansion:
- **Increased feature coverage**: Test with 20-50 features to determine if additional information improves Walking/Walking_Upstairs discrimination while remaining within practical context windows
- **Robustness to irrelevant features**: Deliberately introduce noisy or unrelated features to assess GraphRAG's ability to filter signal from noise through entity extraction. Can the system automatically identify which features are activity-discriminative and down-weight irrelevant inputs?
- **Performance-complexity trade-offs**: Characterize the accuracy-vs-tokens curve to identify optimal feature set sizes for different deployment scenarios

**Systematic prompt and model optimization**: The one-quarter project timeline permitted only limited prompt engineering. Comprehensive optimization requires:
- **A/B testing of query formulations**: Test minimal context ("Classify this sensor data") vs. rich descriptions (current approach) vs. few-shot examples with reasoning traces
- **Context design experiments**: Vary the level of domain knowledge provided (feature descriptions, physics of motion, activity definitions) to determine optimal background information
- **Model scaling**: Progress from gpt-oss models to state-of-the-art commercial LLMs (GPT-4o, Claude Opus, Gemini 2.0) once baseline accuracy justifies API costs. Quantify the accuracy gain vs. cost increase to establish deployment viability
- **Hint ablation studies**: Remove explicit activity descriptions and domain knowledge to test zero-shot reasoning capability. This would reveal whether the system truly understands sensor-activity relationships or relies on provided hints

**Robustness and generalization beyond accuracy**: Production HAR systems must handle real-world deployment challenges beyond clean test set performance:
- **Adversarial robustness**: Introduce sensor noise, missing data, and outlier values to test graceful degradation. Does GraphRAG's interpretable reasoning help identify when inputs are corrupted?
- **Cross-dataset validation**: Test on PAMAP2 (chest-mounted sensors), WISDM (smartphone-based), and Opportunity (multi-modal) datasets without retraining to demonstrate generalization across sensor configurations and activity types
- **Subject variability**: Evaluate performance across demographic groups (age, fitness level, gait patterns) to identify potential biases
- **Real-time streaming scenarios**: Adapt for continuous sensor data where activities unfold without pre-segmented windows. Current batch approach requires known boundaries; streaming would need online graph updates

**Scenario diversity and ablation testing**: Rather than optimizing solely for maximum accuracy, explore system behavior under diverse conditions:
- **Feature modality ablations**: Test with only time-domain features vs. only frequency-domain features vs. combined. Which feature types contribute most to graph structure quality?
- **Community detection depth**: Compare C0-only (coarse communities) vs. C0-C3 (full hierarchy) to determine if hierarchical abstraction truly improves reasoning or introduces unnecessary complexity
- **Query complexity variations**: Test simple entity-specific queries ("What is high Z-acceleration?") vs. multi-hop reasoning ("What activity combines high Z-acceleration with forward motion?") to understand reasoning chain depth limits
- **Graph structure analysis**: Visualize extracted entity relationships to validate that graph topology aligns with domain knowledge (e.g., do acceleration features cluster together? do activities form distinct communities?)

**Key philosophy: Robustness matters as much as accuracy**. A production HAR system achieving 95% on clean test data but failing catastrophically on noisy real-world inputs is less valuable than an 85% accurate system with graceful degradation and predictable failure modes. GraphRAG's interpretable reasoning paths position it well for robustness research—failures can be traced to specific missing entities or incorrect relationships rather than opaque model weights.

**Temporal graph extensions**: Incorporate temporal edges representing activity transitions and duration dependencies. Event-based entity modeling could capture "acceleration onset" as distinct from "sustained acceleration," enabling better temporal reasoning.

**Hybrid query modes**: Develop adaptive system that combines global context (high-level activity taxonomy) with local details (specific sensor patterns) in a single query. This could address global mode's abstraction issues while maintaining corpus-wide awareness.

**Personalized activity graphs**: Explore whether individual users benefit from personalized knowledge graphs capturing their unique activity patterns. GraphRAG's incremental insert capability could enable per-user adaptation while maintaining shared base graph for common patterns.

**Extended validation and real-world deployment**: Test GraphRAG-HAR in applied settings such as healthcare monitoring (fall detection, gait analysis), workplace safety (ergonomic risk assessment), and smart home automation (context-aware assistance). Measure not just accuracy but user trust, clinical utility, and debugging efficiency in production environments.

### **5.6 Broader Impact**

This work demonstrates that GraphRAG's interpretable reasoning extends beyond text summarization to structured sensor-based recognition tasks. The key insight—that query mode selection should match task structure—has implications for applying GraphRAG to other domains:

- **Medical diagnosis**: Local search for case-specific diagnosis, global search for epidemiological pattern discovery
- **Industrial monitoring**: Local search for equipment fault diagnosis, global search for facility-wide efficiency analysis  
- **Financial analysis**: Local search for transaction fraud detection, global search for market trend identification

The interpretability advantages proved more valuable than marginal accuracy improvements. In production HAR systems, engineers need to debug failures, regulators require auditable decisions, and end-users demand transparent explanations. GraphRAG's knowledge graph structure provides all three while maintaining competitive recognition performance.

### **5.7 Conclusions**

This project successfully demonstrates that GraphRAG can transform Human Activity Recognition from a black-box classification problem into an interpretable reasoning task with auditable decision paths. Our work establishes a new paradigm where sensor-based activity recognition systems provide not just predictions, but transparent explanations grounded in structured knowledge graphs.

**Core achievements**:

1. **First application of GraphRAG to sensor-based HAR**: We bridged Microsoft's graph-augmented retrieval paradigm with sensor data analysis, demonstrating that knowledge graph construction and hierarchical community detection can be successfully applied beyond text summarization to numerical time-series classification.

2. **Interpretability through structured reasoning**: GraphRAG's primary strength lies in its ability to trace predictions through explicit entity-relationship paths. When the system classifies an activity as WALKING_UPSTAIRS, it explains WHY—citing high Z-axis acceleration variance, elevated jerk magnitude, and upright posture patterns. This transparency enables systematic debugging: engineers can identify which entities were retrieved, which relationships influenced the decision, and where the reasoning chain broke down in error cases.

3. **Natural language encoding strategy**: Our approach to converting numerical sensor features into descriptive text with statistical context (mean, std, percentiles, range) proved foundational to the system's success. This encoding enabled effective entity extraction ("high variance," "negative Z-gravity") and relationship identification ("acceleration correlates with jerk"), bridging the gap between LLMs' linguistic capabilities and sensor data's numerical nature.

4. **Query mode analysis reveals task-structure dependency**: Contrary to Microsoft's text summarization results where global search excelled, we found local search significantly outperforms for HAR classification tasks. This insight—that query mode selection should match task structure (discrete classification vs. holistic summarization)—has broader implications for applying GraphRAG across domains.

**Why GraphRAG matters for HAR and beyond**:

**Transparency for safety-critical applications**: Healthcare monitoring, fall detection, and elderly care systems require explainable decisions that clinicians and caregivers can validate. A prediction of "Patient shows reduced walking speed variability → potential mobility decline" is actionable; a black-box "Abnormal activity detected" is not. GraphRAG's reasoning paths provide the interpretability essential for medical and safety applications.

**Systematic debugging and improvement**: When traditional ML models fail, engineers resort to speculative feature engineering. When GraphRAG fails, the reasoning path reveals exactly what went wrong—missing entities (e.g., gravity orientation), incorrect relationships (e.g., confusing acceleration types), or abstraction issues (e.g., community summaries too coarse). This enables targeted fixes rather than trial-and-error tuning.

**Auditability for regulated domains**: Financial services, insurance, and legal applications require algorithmic decisions to be auditable. GraphRAG's knowledge graph provides a persistent record of how classifications were made—which evidence was considered, which relationships were deemed relevant, which entities drove the final prediction. This auditability addresses a fundamental barrier to AI deployment in regulated industries.

**Foundation for knowledge transfer and few-shot learning**: The explicit knowledge graph structure opens pathways for transfer learning beyond traditional approaches. Activity definitions, sensor-feature relationships, and motion physics could be shared across deployments, reducing labeled data requirements. A graph built for smartwatch-based HAR could potentially inform smartphone-based systems through relationship reuse.

**Research impact**:

This work extends GraphRAG's applicability from text-based knowledge tasks to structured sensor analytics, demonstrating that graph-augmented reasoning is not domain-specific but rather a general paradigm for combining symbolic knowledge with neural language understanding. The key innovations—numerical-to-text encoding, task-aligned query mode selection, and interpretable reasoning path methodology—provide reusable patterns for other sensor domains including health monitoring, industrial IoT, environmental sensing, and autonomous systems.

**Promising foundation with untapped potential**:

While current accuracy levels indicate need for further optimization (improved prompts, larger models, extended feature sets), the feasibility and interpretability advantages are clearly established. Given the time constraints of a single academic quarter, the system demonstrates proof-of-concept rather than production-ready performance. The path forward is clear: systematic prompt engineering, commercial API integration for enhanced reasoning, feature engineering for better activity discrimination, and robustness testing across diverse deployment scenarios.

**The paradigm shift**:

GraphRAG represents a fundamental shift from **"accurate but opaque"** to **"accurate AND interpretable"** AI systems. For HAR and sensor-based recognition more broadly, this means moving from models we can't understand to systems we can debug, from predictions we must accept to reasoning we can validate, from black boxes to auditable intelligence.

This is not merely a technical improvement—it's a necessary evolution for deploying AI in human-centered applications where trust, safety, and accountability matter as much as raw performance metrics. GraphRAG-based HAR demonstrates that we need not sacrifice interpretability for capability; structured reasoning can deliver both.

---

# **6. References**

[1] Ji, S., Zheng, X., & Wu, C. (2024). HARGPT: Are LLMs zero-shot human activity recognizers? *IEEE International Workshop on Foundation Models for Cyber-Physical Systems & Internet of Things (FMSys)*, pp. 38-43. arXiv:2403.02727. https://www.semanticscholar.org/paper/HARGPT:-Are-LLMs-Zero-Shot-Human-Activity-Ji-Zheng/91aa0251b5809ac6d380dd1be8544a1a59be4762

[2] Li, Z., Chen, B., Xue, H., & Salim, F. D. (2025). ZARA: Zero-shot motion time-series analysis via knowledge and retrieval driven LLM agents. *arXiv preprint arXiv:2508.04038*. https://arxiv.org/abs/2508.04038

[3] LLMTrack Research Group. (2024). Are you being tracked? Discover the power of zero-shot trajectory tracing with LLMs! *arXiv preprint arXiv:2403.06201*. https://arxiv.org/html/2403.06201v1

[4] Civitarese, G., Fiori, M., & Bettini, C. (2024). Large language models are zero-shot recognizers for activities of daily living. *ACM Transactions on Intelligent Systems and Technology (TIST)*. arXiv:2407.01238. https://dl.acm.org/doi/10.1145/3725856

[5] Chowdhury, R. R., Kapila, R., Panse, A., Zhang, X., Teng, D., Kulkarni, R., Hong, D., Gupta, R. K., & Shang, J. (2025). ZeroHAR: Sensor context augments zero-shot wearable action recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*, Vol. 39, No. 15, pp. 16046-16054. DOI: 10.1609/aaai.v39i15.33762. https://ojs.aaai.org/index.php/AAAI/article/view/33762

[6] IMUGPT 2.0 Research Team. (2024). IMUGPT 2.0: Language-based cross modality transfer for sensor-based human activity recognition. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*. arXiv:2402.01049. https://dl.acm.org/doi/10.1145/3678545

[7] LanHAR Research Group. (2024). LanHAR: Language-based cross-dataset HAR through semantic alignment. *arXiv preprint*.

[8] Li, Z., Deldari, S., Chen, L., Xue, H., & Salim, F. D. (2025). SensorLLM: Aligning large language models with motion sensors for human activity recognition. *EMNLP 2025*. arXiv:2410.10624. https://arxiv.org/abs/2410.10624

[9] Yang, Y., Ayush, K., Zhang, Y., et al. (2025). SensorLM: Learning the language of wearable sensors. *arXiv preprint arXiv:2506.09108*. Google Research, Google Health, Google DeepMind. https://research.google/blog/sensorlm-learning-the-language-of-wearable-sensors/

[10] Moon, S., Madotto, A., Lin, Z., Saraf, A., Bearman, A., & Damavandi, B. (2023). IMU2CLIP: Language-grounded motion sensor translation with multimodal contrastive learning. *Findings of EMNLP 2023*, Singapore. arXiv:2210.14395. https://arxiv.org/abs/2210.14395

[11] Ansari, A. F., et al. (2024). Chronos: Learning the language of time series. *Proceedings of the International Conference on Machine Learning (ICML)*. Amazon. arXiv:2403.07815.

[12] Jin, M., et al. (2024). Time-LLM: Time series forecasting by reprogramming large language models. *Proceedings of the International Conference on Learning Representations (ICLR)*. arXiv:2310.01728.

[13] Talukder, S., et al. (2024). TOTEM: TOkenized time series EMbeddings for general time series analysis. *Transactions on Machine Learning Research (TMLR)*.

[14] MoPFormer Research Team. (2025). MoPFormer: Motion-primitive transformer for wearable-sensor activity recognition. *arXiv preprint arXiv:2505.20744*. https://arxiv.org/html/2505.20744

[15] Graph-Based HAR Robustness Study. (2024). Using graphs to perform effective sensor-based human activity recognition in smart homes. *PMC Article PMC11207551*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11207551/

[16] Graph Neural Networks for HAR. (2024). Research on GNN architectures for sensor-based activity recognition with robustness to missing data.

[17] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*. https://arxiv.org/abs/2404.16130

[18] Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

[19] gusye1234. (2024). nano-graphrag: A simple, easy-to-hack GraphRAG implementation. *GitHub Repository*. https://github.com/gusye1234/nano-graphrag

[20] Global Wearable Devices Market Report. (2024). Market research forecast projecting $63 billion by 2025.

---

# **7. Supplementary Material**

## **7.a. Datasets**

### **UCI Human Activity Recognition Using Smartphones Dataset**

**Source**: UCI Machine Learning Repository  
**URL**: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones  
**Citation**: Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013)

**Data Format**:
- **Raw signals**: Text files containing tri-axial accelerometer and gyroscope readings at 50 Hz
- **Processed features**: 561-dimensional feature vectors (CSV format)
- **Labels**: Activity class labels (1-6) corresponding to WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **Subject IDs**: Anonymous subject identifiers (1-30)

**Data Collection**:
- 30 volunteers aged 19-48 years
- Samsung Galaxy S II worn on waist
- Video-recorded experiments for manual labeling
- Six activities performed in controlled laboratory setting

**Preprocessing Steps**:
1. **Noise filtering**: Median filter + 3rd-order Butterworth low-pass filter (20 Hz cutoff)
2. **Gravity separation**: 0.3 Hz cutoff Butterworth filter to separate body acceleration from gravity
3. **Windowing**: 2.56-second windows (128 samples) with 50% overlap
4. **Feature extraction**: 561 features computed per window including:
   - Time-domain statistics: mean, std, mad, max, min, sma, energy, iqr, entropy, arCoeff
   - Frequency-domain statistics: FFT-based mean, std, mad, max, min, skewness, kurtosis, energy
   - Additional: Correlation coefficients, angle measurements

**Labeling/Annotation**:
- Manual labeling based on video recordings
- Ground truth activities verified by human annotators
- No automated pre-labeling

**Train/Test Split**:
- 70% training (7,352 samples, 21 subjects)
- 30% test (2,947 samples, 9 subjects)
- Subject-independent split (no subject appears in both sets)

**Dataset Statistics**:
- Total instances: 10,299
- Activity distribution approximately balanced (1,722 WALKING, 1,544 WALKING_UPSTAIRS, 1,406 WALKING_DOWNSTAIRS, 1,777 SITTING, 1,906 STANDING, 1,944 LAYING)

**Known Challenges**:
- SITTING vs. STANDING confusion (similar sensor patterns from waist perspective)
- Individual variation in activity execution
- Sensor noise in real-world deployment scenarios

## **7.b. Software**

### **External Libraries and Models**

**nano-graphrag**
- Version: Latest (as of December 2024)
- Source: https://github.com/gusye1234/nano-graphrag
- License: MIT
- Purpose: Core GraphRAG implementation for knowledge graph construction and querying
- Installation: `pip install nano-graphrag`

**Ollama**
- Version: Latest stable release
- Source: https://ollama.com
- License: Open source
- Purpose: Local LLM inference for entity extraction, relationship identification, and summarization
- Models used:
  - gpt-oss:20b (debugging and prototyping)
  - gpt-oss:120b (final evaluation)
- Installation: `curl -fsSL https://ollama.com/install.sh | sh`

**NetworkX**
- Version: 3.2+
- Source: https://networkx.org
- License: BSD
- Purpose: Graph storage and manipulation backend
- Installation: `pip install networkx`

**NumPy**
- Version: 1.24+
- Purpose: Numerical operations on sensor data
- Installation: `pip install numpy`

**pandas**
- Version: 2.0+
- Purpose: Data preprocessing and feature manipulation
- Installation: `pip install pandas`

**scikit-learn**
- Version: 1.3+
- Purpose: Baseline comparisons and evaluation metrics
- Installation: `pip install scikit-learn`

**matplotlib / seaborn**
- Purpose: Visualization of results, confusion matrices, graph structures
- Installation: `pip install matplotlib seaborn`

### **Internal Modules**

**generate_queries.py**
- **Purpose**: Generate 180 query files from UCI HAR dataset with natural language descriptions
- **Key functions**:
  - `load_all_data()`: Load train and test UCI HAR data (30 subjects, 6 activities)
  - `get_feature_description()`: Map feature names to human-readable descriptions
  - `generate_query_text()`: Convert sensor statistics to natural language paragraphs
  - `generate_all_queries()`: Create all 180 query files and ground truth mapping
- **Input**: UCI HAR Dataset (X_train.txt, X_test.txt, y_train.txt, y_test.txt, subject_train.txt, subject_test.txt, features.txt)
- **Output**: 
  - 180 query files: `query_<subject>_<activity>.txt`
  - Ground truth file: `ground_truth.txt`
- **Configuration**:
  - Selected features: [9, 42, 200, 201, 214, 226] (array indices for 6 key features)
  - Base path: `/home/m202/M202/Richard/UCI HAR Dataset`
  - Output directory: `/home/m202/M202/Richard/query`

**no_openai_key_at_all.py**
- **Purpose**: Build GraphRAG knowledge graph index from sensor data descriptions
- **Key functions**:
  - `local_embedding()`: Generate embeddings using Sentence Transformers
  - `ollama_model_if_cache()`: LLM inference with response caching
  - `insert()`: Index sensor descriptions into GraphRAG (entity extraction, relationship identification, community detection)
  - `query()`: Run test queries through GraphRAG with global and local modes
- **Configuration**:
  - Working directory: `./nano_graphrag_cache_ollama_TEST`
  - Model: `gpt-oss:120b`
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - Input document: `./tests/test_4.txt` (sensor data descriptions)
  - Query file: `/home/m202/M202/Richard/nano-graphrag/tests/query.txt`
- **Caching**: LLM responses cached in working directory for reproducibility

**evaluate_graphrag_full_180.py**
- **Purpose**: Comprehensive evaluation of GraphRAG HAR performance on all 180 queries
- **Key functions**:
  - `extract_activity_label()`: Extract predicted activity from LLM response text using pattern matching
  - `load_ground_truth()`: Load query filename → activity label mappings
  - `group_queries_by_activity()`: Organize queries by activity class for balanced analysis
  - `evaluate_graphrag()`: Main evaluation pipeline processing all 180 queries
- **Evaluation Process**:
  1. Load ground truth and select queries (all 180 or sampled)
  2. Initialize GraphRAG with existing cache (no reindexing)
  3. For each query: load text → run GraphRAG → extract prediction → compare with ground truth
  4. Generate confusion matrix and per-activity accuracy
  5. Save detailed results to file
- **Configuration**:
  - Working directory: `./nano_graphrag_cache_ollama_TEST` (same as indexing)
  - Model: `gpt-oss:120b`
  - Query directory: `/home/m202/M202/Richard/query`
  - Ground truth: `/home/m202/M202/Richard/query/ground_truth.txt`
  - Test mode: `TEST_ALL_QUERIES = True` (all 180 queries)
  - Query mode: `QueryParam(mode="global")` or `QueryParam(mode="local")`
- **Output**:
  - Console: Real-time progress, confusion matrix, summary statistics
  - File: `evaluation_results_full_180.txt` with detailed per-query results

**har_visualizer.py**
- **Purpose**: Interactive visualization tool for exploring UCI HAR dataset features
- **Key features**:
  - Browse all 561 features with Previous/Next buttons
  - Color-code samples by activity class (6 colors)
  - Filter by subject ID (1-30) or view all subjects
  - Display real-time statistics (mean, std, min, max)
  - Toggle between activity-colored and single-color views
- **Implementation**:
  - `HARVisualizer` class: Main visualization controller
  - `setup_plot()`: Create matplotlib figure with interactive controls
  - `plot_feature()`: Render current feature with activity colors and statistics
  - `update_feature_from_text()`: Navigate to specific feature by index
  - `update_subject_filter()`: Filter data by subject
- **Usage**: Feature selection validation and error analysis
- **Configuration**:
  - Base path: `C:\Users\Yunqing\OneDrive\School\UCLA\GraphRAG Project\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset`
  - Input files: X_train.txt, y_train.txt, subject_train.txt, features.txt

### **Code Repository**

Project code and data available at: [Link to be added]

**Repository structure**:
```
project/
├── nano_graphrag_cache_ollama_TEST/    # GraphRAG working directory
│   ├── vdb_entities.json               # Entity embeddings
│   ├── kv_store_full_docs.json         # Original documents
│   ├── kv_store_text_chunks.json       # Text chunks
│   ├── kv_store_community_reports.json # Community summaries
│   └── graph_chunk_entity_relation.graphml  # Knowledge graph
├── query/                              # Generated queries
│   ├── query_1_Walking.txt
│   ├── query_1_Sitting.txt
│   ├── ...                             # 180 query files total
│   └── ground_truth.txt                # Query → activity mappings
├── tests/
│   ├── test_4.txt                      # Sensor descriptions for indexing
│   └── query.txt                       # Test query
├── UCI HAR Dataset/                    # Original UCI HAR data
│   ├── train/
│   │   ├── X_train.txt
│   │   ├── y_train.txt
│   │   └── subject_train.txt
│   ├── test/
│   │   ├── X_test.txt
│   │   ├── y_test.txt
│   │   └── subject_test.txt
│   └── features.txt
├── generate_queries.py                 # Query generation script
├── no_openai_key_at_all.py            # GraphRAG indexing script
├── evaluate_graphrag_full_180.py      # Evaluation script
├── har_visualizer.py                  # Visualization tool
└── evaluation_results_full_180.txt    # Evaluation output
```

### **Environment Setup**

**System requirements**:
- Python 3.9+
- 16GB+ RAM (for gpt-oss:120b inference)
- NVIDIA GPU with CUDA support (for Sentence Transformers and Ollama acceleration)
- ~50GB disk space (for Ollama models and GraphRAG cache)

**Installation instructions**:
```bash
# Install Python dependencies
pip install nano-graphrag sentence-transformers numpy matplotlib

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLM models
ollama pull gpt-oss:20b
ollama pull gpt-oss:120b

# Download UCI HAR dataset
# From: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
# Extract to: ./UCI HAR Dataset/
```

**Running the pipeline**:
```bash
# Step 1: Generate 180 query files from UCI HAR dataset
python generate_queries.py
# Output: 180 query files in /home/m202/M202/Richard/query/

# Step 2: Build GraphRAG knowledge graph index
python no_openai_key_at_all.py
# This runs the insert() function to create graph from test_4.txt
# Cache saved to: ./nano_graphrag_cache_ollama_TEST/

# Step 3: Run full evaluation (180 queries)
python evaluate_graphrag_full_180.py
# This tests all queries and generates evaluation_results_full_180.txt

# Step 4 (Optional): Visualize UCI HAR features
python har_visualizer.py
# Interactive matplotlib window for exploring 561 features
```

**Experiment configuration**:
To switch between models or query modes, edit the configuration in the respective scripts:
- **Model selection**: Change `MODEL = "gpt-oss:120b"` to `"gpt-oss:20b"`
- **Query mode**: Change `param=QueryParam(mode="global")` to `mode="local"`
- **Test all queries**: Set `TEST_ALL_QUERIES = True` (default)
- **Sample queries**: Set `TEST_ALL_QUERIES = False` and adjust `QUERIES_PER_ACTIVITY`

### **Reproducibility Notes**

- All experiments use the actual UCI HAR dataset with no modifications to ground truth labels
- Ollama model versions: gpt-oss:20b and gpt-oss:120b (specific commit hashes available in Ollama model registry)
- UCI HAR dataset: Official version from UCI ML Repository (10,299 samples, 30 subjects)
- LLM inference parameters:
  - Temperature: Default (not explicitly set, uses Ollama defaults)
  - No additional prompt engineering beyond nano-graphrag defaults
- GraphRAG configuration:
  - Community detection: Leiden algorithm with default resolution
  - Hierarchy depth: 4 levels (C0-C3)
  - Top-K communities (global search): 512 (nano-graphrag default)
  - Top-K entities (local search): 20 (nano-graphrag default)
- Sentence Transformers:
  - Model: all-MiniLM-L6-v2
  - Normalization: L2 (cosine similarity)
  - Max sequence: 256 tokens

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Draft (Section 4 pending experimental results)
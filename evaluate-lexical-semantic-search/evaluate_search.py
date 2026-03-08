import json
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')   # Add this line

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# Load bi-encoder model (good balance of speed/accuracy)
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # [citation:8]

# Create document embeddings
def create_dense_index(documents):
    # Prepare texts
    texts = [doc['title'] + " " + doc['text'] for doc in documents]
    
    # Encode in batches to manage memory
    batch_size = 32
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = bi_encoder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    
    return index, embeddings

def dense_search(query, index, documents, bi_encoder, top_k=100):
    # Encode query
    query_embedding = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx != -1:  # Valid index
            results.append({
                'id': documents[idx]['id'],
                'text': documents[idx]['title'] + " " + documents[idx]['text'],
                'score': float(score)
            })
    return results



# Load and prepare documents
def load_documents(file_path="simplewiki-2020-11-01.jsonl", max_docs=10000):
    documents = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            data = json.loads(line)
            # Combine paragraphs into one text (they are a list)
            paragraphs = data.get('paragraphs', [])
            # Join paragraphs with a space, or keep as one block
            full_text = ' '.join(paragraphs) if paragraphs else ''
            documents.append({
                'id': i,                             # you can also use data['id'] if available
                'title': data.get('title', ''),
                'text': full_text,                    # now filled
                'url': data.get('url', '')
            })
    return documents

# Tokenize and build BM25 index
def build_bm25_index(documents):
    tokenized_corpus = []
    for doc in documents:
        # Combine title and text for better context
        full_text = doc['title'] + " " + doc['text']
        tokens = word_tokenize(full_text.lower())
        tokenized_corpus.append(tokens)
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

# Search function
def bm25_search(query, bm25, documents, top_k=100):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'id': documents[idx]['id'],
            'text': documents[idx]['title'] + " " + documents[idx]['text'],
            'score': scores[idx]
        })
    return results



def reciprocal_rank_fusion(bm25_results, dense_results, documents, k=60):
    """
    Reciprocal Rank Fusion (RRF) - combines rankings from multiple sources [citation:1]
    """
    fused_scores = {}
    
    # Add BM25 results with RRF scoring
    for rank, result in enumerate(bm25_results):
        doc_id = result['id']
        fused_scores[doc_id] = 1.0 / (k + rank + 1)
    
    # Add dense results with RRF scoring
    for rank, result in enumerate(dense_results):
        doc_id = result['id']
        if doc_id in fused_scores:
            fused_scores[doc_id] += 1.0 / (k + rank + 1)
        else:
            fused_scores[doc_id] = 1.0 / (k + rank + 1)
    
    # Sort by fused score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 100 candidates for re-ranking
    candidate_ids = [doc_id for doc_id, _ in sorted_docs[:100]]
    doc_map = {doc['id']: doc for doc in documents}
    candidates = [doc_map[doc_id] for doc_id in candidate_ids if doc_id in doc_map]
    return candidates       


from sentence_transformers import CrossEncoder
import torch

# Load cross-encoder model (trained on MS MARCO for relevance scoring) [citation:1][citation:3]
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_with_cross_encoder(query, candidates, top_k=10):
    """
    Re-rank candidates using cross-encoder for precise relevance scoring
    """
    # Prepare pairs
    pairs = [(query, candidate['text']) for candidate in candidates]
    
    # Get relevance scores
    with torch.no_grad():
        scores = cross_encoder.predict(pairs, convert_to_tensor=True)
    
    # Convert to numpy if needed
    if torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    
    # Combine with candidates
    results = []
    for i, candidate in enumerate(candidates):
        results.append({
            'id': candidate['id'],
            'text': candidate['text'],
            'score': float(scores[i])
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]


def hybrid_search_pipeline(query, documents, bm25, dense_index, 
                          bi_encoder, cross_encoder, top_k_final=10):
    """
    Complete end-to-end hybrid search pipeline
    """
    # Stage 1: BM25 retrieval
    print(f"🔍 Stage 1: BM25 retrieval for '{query}'")
    bm25_results = bm25_search(query, bm25, documents, top_k=100)
    print(f"   Retrieved {len(bm25_results)} documents")
    
    # Stage 2: Dense retrieval
    print(f"🔍 Stage 2: Dense retrieval")
    dense_results = dense_search(query, dense_index, documents, bi_encoder, top_k=100)
    print(f"   Retrieved {len(dense_results)} documents")
    
    # Stage 3: Hybrid fusion
    print(f"🔄 Stage 3: Hybrid fusion (RRF)")
    candidates = reciprocal_rank_fusion(bm25_results, dense_results, documents)
    print(f"   Created {len(candidates)} candidates for re-ranking")
    
    # Stage 4: Cross-encoder re-ranking
    print(f"⚡ Stage 4: Cross-encoder re-ranking")
    final_results = rerank_with_cross_encoder(query, candidates, top_k=top_k_final)
    print(f"   Final top-{top_k_final} results")
    
    return final_results

# Example usage
def run_poc():
    # Load documents (increase max_docs to include all IDs)
    print("Loading documents...")
    documents = load_documents(max_docs=100000)  # or a number > 84853

    # Build BM25 index
    print("Building BM25 index...")
    bm25, _ = build_bm25_index(documents)

    # Build dense index
    print("Creating dense embeddings (this may take a few minutes)...")
    dense_index, _ = create_dense_index(documents)

    # --- Original demo queries (optional) ---
    demo_queries = [
        "What is machine learning?",
        "History of the Roman Empire",
        "How does photosynthesis work?",
        "Python programming language",
        "Famous paintings in the Louvre"
    ]
    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        results = hybrid_search_pipeline(query, documents, bm25, dense_index,
                                         bi_encoder, cross_encoder)
        print(f"\n📋 Final Top Results:")
        for i, result in enumerate(results, 1):
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            print(f"{i:2d}. Score: {result['score']:.4f} | {text_preview}")

    # --- Evaluation with custom queries ---
    print("\n" + "="*60)
    print("🔬 Starting Evaluation with Custom Queries")
    print("="*60)

    # Build a lookup from id to document
    doc_by_id = {doc['id']: doc for doc in documents}

    # Define the two custom queries
    custom_queries = [
        ("serial killer executed in Florida", [9824]),   # keyword-rich
        ("celebration of the lunar new year", [84801])   # semantic-rich
    ]

    # Verify that the relevant documents exist
    for _, rel_ids in custom_queries:
        for rid in rel_ids:
            if rid not in doc_by_id:
                print(f"Warning: Document with id {rid} not found. Increase max_docs in load_documents.")

    # Run evaluation
    metrics = evaluate_pipeline(
        documents=documents,
        bm25=bm25,
        dense_index=dense_index,
        bi_encoder=bi_encoder,
        cross_encoder=cross_encoder,
        test_queries_with_relevance=custom_queries,
        k=10
    )

    # Print results
    print("\n📊 Evaluation Results (Precision@10 / Recall@10):")
    for method, scores in metrics.items():
        print(f"{method:15} P@10: {scores['precision_avg']:.3f}  R@10: {scores['recall_avg']:.3f}")

    # Optionally print per‑query details
    print("\n📋 Per-Query Details:")
    for i, (query, rel_ids) in enumerate(custom_queries):
        print(f"\nQuery {i+1}: {query}")
        # We can recompute for each method or reuse stored values – for brevity, we skip here.
        # You could extend evaluate_pipeline to return per‑query lists.



def evaluate_pipeline(documents, bm25, dense_index, bi_encoder, cross_encoder,
                      test_queries_with_relevance):
    """
    Evaluates BM25, dense, and hybrid+rerank pipelines.
    Returns average Precision@10 and Recall@10 for each method.
    """
    # Accumulators
    metrics = {
        'bm25_only': {'precision_sum': 0, 'recall_sum': 0, 'count': 0},
        'dense_only': {'precision_sum': 0, 'recall_sum': 0, 'count': 0},
        'hybrid+rerank': {'precision_sum': 0, 'recall_sum': 0, 'count': 0}
    }
    
    for query, relevant_ids in test_queries_with_relevance:
        if not relevant_ids:
            continue   # skip queries with no relevance judgments
        
        relevant_set = set(relevant_ids)
        total_relevant = len(relevant_set)
        
        # --- BM25 only (top 10) ---
        bm25_results = bm25_search(query, bm25, documents, top_k=10)
        bm25_ids = [r['id'] for r in bm25_results]
        bm25_relevant = len(set(bm25_ids) & relevant_set)
        bm25_precision = bm25_relevant / 10.0
        bm25_recall = bm25_relevant / total_relevant
        
        metrics['bm25_only']['precision_sum'] += bm25_precision
        metrics['bm25_only']['recall_sum'] += bm25_recall
        metrics['bm25_only']['count'] += 1
        
        # --- Dense only (top 10) ---
        dense_results = dense_search(query, dense_index, documents, bi_encoder, top_k=10)
        dense_ids = [r['id'] for r in dense_results]
        dense_relevant = len(set(dense_ids) & relevant_set)
        dense_precision = dense_relevant / 10.0
        dense_recall = dense_relevant / total_relevant
        
        metrics['dense_only']['precision_sum'] += dense_precision
        metrics['dense_only']['recall_sum'] += dense_recall
        metrics['dense_only']['count'] += 1
        
        # --- Hybrid + rerank (top 10) ---
        # Retrieve top 100 from each
        bm25_100 = bm25_search(query, bm25, documents, top_k=100)
        dense_100 = dense_search(query, dense_index, documents, bi_encoder, top_k=100)
        # Fuse with RRF (must pass documents)
        candidates = reciprocal_rank_fusion(bm25_100, dense_100, documents)
        # Re‑rank with cross‑encoder and take top 10
        reranked = rerank_with_cross_encoder(query, candidates, top_k=10)
        reranked_ids = [r['id'] for r in reranked]
        hybrid_relevant = len(set(reranked_ids) & relevant_set)
        hybrid_precision = hybrid_relevant / 10.0
        hybrid_recall = hybrid_relevant / total_relevant
        
        metrics['hybrid+rerank']['precision_sum'] += hybrid_precision
        metrics['hybrid+rerank']['recall_sum'] += hybrid_recall
        metrics['hybrid+rerank']['count'] += 1
    
    # Average the results
    results = {}
    for method, data in metrics.items():
        if data['count'] > 0:
            results[method] = {
                'precision@10': data['precision_sum'] / data['count'],
                'recall@10': data['recall_sum'] / data['count']
            }
        else:
            results[method] = {'precision@10': 0, 'recall@10': 0}
    
    return results

if __name__ == "__main__":
    run_poc()



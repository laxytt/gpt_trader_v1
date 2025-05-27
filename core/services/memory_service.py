"""
Memory service for RAG-based trade case storage and retrieval.
Manages historical trade data for pattern recognition and learning.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Sentence transformers or FAISS not available. Memory service disabled.")

from core.domain.models import Trade, TradeCase, TradeResult, create_case_id
from core.domain.exceptions import MemoryError, ErrorContext
from core.infrastructure.database.repositories import MemoryCaseRepository
from config.settings import DatabaseSettings


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for trade context using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise MemoryError("Sentence transformers library not available")
        
        self.model_name = model_name
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise MemoryError(f"Failed to initialize embedding model: {str(e)}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with embedding vector
        """
        if not self._model:
            raise MemoryError("Embedding model not initialized")
        
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise MemoryError(f"Failed to generate embedding: {str(e)}")
    
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array with shape (len(texts), embedding_dim)
        """
        if not self._model:
            raise MemoryError("Embedding model not initialized")
        
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise MemoryError(f"Failed to generate batch embeddings: {str(e)}")


class SimilaritySearchEngine:
    """FAISS-based similarity search for trade cases"""
    
    def __init__(self, embedding_dim: int = 384):
        if not EMBEDDINGS_AVAILABLE:
            raise MemoryError("FAISS library not available")
        
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        self.case_ids = []  # Track case IDs corresponding to index positions
        
    def add_embeddings(self, embeddings: np.ndarray, case_ids: List[str]):
        """
        Add embeddings to the search index.
        
        Args:
            embeddings: Array of embeddings with shape (n, embedding_dim)
            case_ids: List of case IDs corresponding to embeddings
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise MemoryError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        try:
            self.index.add(embeddings)
            self.case_ids.extend(case_ids)
            logger.debug(f"Added {len(case_ids)} embeddings to search index")
        except Exception as e:
            logger.error(f"Failed to add embeddings to index: {e}")
            raise MemoryError(f"Failed to add embeddings: {str(e)}")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        """
        Search for similar cases.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar cases to return
            
        Returns:
            List of (distance, case_id) tuples
        """
        if len(self.case_ids) == 0:
            return []
        
        try:
            # Ensure query is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_embedding, min(k, len(self.case_ids)))
            
            # Return results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.case_ids):
                    results.append((float(distance), self.case_ids[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise MemoryError(f"Similarity search failed: {str(e)}")
    
    def get_index_size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal
    
    def clear_index(self):
        """Clear the search index"""
        self.index.reset()
        self.case_ids.clear()


class MemoryService:
    """
    Service for managing trade memory using RAG (Retrieval-Augmented Generation).
    Stores trade cases with embeddings for similarity-based retrieval.
    """
    
    def __init__(
        self,
        repository: MemoryCaseRepository,
        database_config: DatabaseSettings
    ):
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Memory service initialized without embedding support")
            self._embeddings_enabled = False
        else:
            self._embeddings_enabled = True
        
        self.repository = repository
        self.database_config = database_config
        self.max_cases = database_config.max_memory_cases
        
        # Initialize components if embeddings available
        if self._embeddings_enabled:
            self.embedding_generator = EmbeddingGenerator(database_config.embedding_model)
            self.search_engine = SimilaritySearchEngine(self.embedding_generator.embedding_dim)
            self._load_existing_cases()
        else:
            self.embedding_generator = None
            self.search_engine = None
    
    def _load_existing_cases(self):
        """Load existing cases into the search index"""
        if not self._embeddings_enabled:
            return
        
        try:
            cases_data = self.repository.get_all_cases_with_embeddings()
            
            if not cases_data:
                logger.info("No existing cases found in database")
                return
            
            # Extract embeddings and case IDs
            embeddings = []
            case_ids = []
            
            for case_data in cases_data:
                if case_data['embedding']:
                    embedding = np.frombuffer(case_data['embedding'], dtype=np.float32)
                    embeddings.append(embedding)
                    case_ids.append(case_data['id'])
            
            if embeddings:
                embeddings_array = np.stack(embeddings)
                self.search_engine.add_embeddings(embeddings_array, case_ids)
                logger.info(f"Loaded {len(embeddings)} existing cases into memory")
            
        except Exception as e:
            logger.error(f"Failed to load existing cases: {e}")
            # Continue without existing cases rather than fail
    
    async def add_trade_case(self, trade: Trade) -> bool:
        """
        Add a completed trade to memory.
        
        Args:
            trade: Completed trade to add
            
        Returns:
            True if successfully added
        """
        if not trade.original_signal or not trade.result:
            logger.warning(f"Cannot add incomplete trade to memory: {trade.id}")
            return False
        
        with ErrorContext("Add trade case", symbol=trade.symbol) as ctx:
            ctx.add_detail("trade_id", trade.id)
            
            try:
                # Create trade case
                case = self._create_trade_case(trade)
                
                # Generate embedding if enabled
                embedding_bytes = None
                if self._embeddings_enabled:
                    embedding = self.embedding_generator.generate_embedding(case.context)
                    embedding_bytes = embedding.tobytes()
                    
                    # Add to search index
                    self.search_engine.add_embeddings(
                        embedding.reshape(1, -1), 
                        [case.id]
                    )
                else:
                    embedding_bytes = b''  # Empty bytes for disabled embeddings
                
                # Save to database
                self.repository.save_case(case, embedding_bytes)
                
                # Cleanup old cases if needed
                await self._cleanup_old_cases()
                
                logger.info(f"Added trade case to memory: {case.id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add trade case: {e}")
                return False
    
    def _create_trade_case(self, trade: Trade) -> TradeCase:
        """Create TradeCase from Trade"""
        # Create context description
        signal = trade.original_signal
        latest_candle = None
        
        # Try to extract market context from signal
        market_context = signal.market_context if signal else {}
        
        context_parts = [
            f"Symbol: {trade.symbol}",
            f"Side: {trade.side.value}",
            f"Entry: {trade.entry_price:.5f}",
            f"Duration: {trade.duration_minutes:.1f}min" if trade.duration_minutes else "Duration: unknown"
        ]
        
        # Add market context if available
        if market_context:
            session = market_context.get('session', 'unknown')
            volatility = market_context.get('volatility', 'unknown')
            context_parts.extend([
                f"Session: {session}",
                f"Volatility: {volatility}"
            ])
        
        # Add technical indicators from signal reason if available
        if signal and signal.reason:
            context_parts.append(f"Reason: {signal.reason}")
        
        context = ", ".join(context_parts)
        
        case_id = create_case_id(trade.symbol, trade.timestamp, trade.entry_price)
        
        return TradeCase(
            id=case_id,
            symbol=trade.symbol,
            context=context,
            signal=trade.side,
            entry_price=trade.entry_price,
            risk_reward=trade.risk_reward_ratio or 1.0,
            result=trade.result,
            reason=signal.reason if signal else "No signal data",
            timestamp=trade.timestamp
        )
    
    async def find_similar_cases(
        self,
        context: str,
        symbol: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical trade cases.
        
        Args:
            context: Context description to search for
            symbol: Trading symbol to filter by
            limit: Maximum number of cases to return
            
        Returns:
            List of similar case dictionaries
        """
        if not self._embeddings_enabled:
            logger.debug("Embeddings disabled, using simple query")
            return await self._simple_case_query(symbol, limit)
        
        with ErrorContext("Find similar cases", symbol=symbol) as ctx:
            ctx.add_detail("context_length", len(context))
            ctx.add_detail("limit", limit)
            
            try:
                # Generate query embedding
                query_embedding = self.embedding_generator.generate_embedding(context)
                
                # Search for similar cases (oversample to allow filtering)
                similar_results = self.search_engine.search_similar(
                    query_embedding, 
                    k=limit * 3  # Oversample
                )
                
                if not similar_results:
                    logger.debug(f"No similar cases found for {symbol}")
                    return []
                
                # Get case details from database
                case_details = []
                for distance, case_id in similar_results:
                    cases = self.repository.get_cases_for_symbol(symbol, limit=50)
                    
                    # Find matching case
                    for case_data in cases:
                        if case_data['id'] == case_id:
                            case_dict = {
                                'id': case_id,
                                'symbol': case_data['symbol'],
                                'context': case_data['context'],
                                'signal': case_data['signal'],
                                'entry_price': case_data['entry_price'],
                                'risk_reward': case_data['risk_reward'],
                                'result': case_data['result'],
                                'reason': case_data['reason'],
                                'similarity_distance': distance
                            }
                            case_details.append(case_dict)
                            break
                    
                    if len(case_details) >= limit:
                        break
                
                logger.debug(f"Found {len(case_details)} similar cases for {symbol}")
                return case_details
                
            except Exception as e:
                logger.error(f"Similar case search failed: {e}")
                return await self._simple_case_query(symbol, limit)
    
    async def _simple_case_query(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Simple case query without embeddings"""
        try:
            cases = self.repository.get_cases_for_symbol(symbol, limit)
            return [
                {
                    'id': case['id'],
                    'symbol': case['symbol'],
                    'context': case['context'],
                    'signal': case['signal'],
                    'entry_price': case['entry_price'],
                    'risk_reward': case['risk_reward'],
                    'result': case['result'],
                    'reason': case['reason']
                }
                for case in cases
            ]
        except Exception as e:
            logger.error(f"Simple case query failed: {e}")
            return []
    
    def get_performance_stats(self, symbol: str, sample_size: int = 20) -> Dict[str, Any]:
        """
        Get performance statistics for a symbol.
        
        Args:
            symbol: Trading symbol
            sample_size: Number of recent trades to analyze
            
        Returns:
            Dictionary with performance statistics
        """
        try:
            cases = self.repository.get_cases_for_symbol(symbol, sample_size)
            
            if not cases:
                return {
                    'streak_type': 'unknown',
                    'streak_length': 0,
                    'win_rate': 0.0,
                    'sample_size': 0
                }
            
            # Analyze results
            results = [case['result'] for case in cases]
            win_count = sum(1 for result in results if 'win' in result.lower())
            win_rate = win_count / len(results)
            
            # Calculate current streak
            if results:
                last_result = results[0]  # Most recent (cases are ordered DESC)
                streak_type = 'win' if 'win' in last_result.lower() else 'loss'
                streak_length = 1
                
                for result in results[1:]:
                    current_type = 'win' if 'win' in result.lower() else 'loss'
                    if current_type == streak_type:
                        streak_length += 1
                    else:
                        break
            else:
                streak_type = 'unknown'
                streak_length = 0
            
            return {
                'streak_type': streak_type,
                'streak_length': streak_length,
                'win_rate': win_rate,
                'sample_size': len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance stats for {symbol}: {e}")
            return {
                'streak_type': 'unknown',
                'streak_length': 0,
                'win_rate': 0.0,
                'sample_size': 0
            }
    
    async def _cleanup_old_cases(self):
        """Remove old cases if we exceed the maximum"""
        try:
            removed_count = self.repository.cleanup_old_cases(self.max_cases)
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old cases")
                
                # Reload search index after cleanup
                if self._embeddings_enabled:
                    self.search_engine.clear_index()
                    self._load_existing_cases()
        except Exception as e:
            logger.error(f"Failed to cleanup old cases: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory service statistics"""
        stats = {
            'embeddings_enabled': self._embeddings_enabled,
            'max_cases': self.max_cases
        }
        
        if self._embeddings_enabled and self.search_engine:
            stats.update({
                'index_size': self.search_engine.get_index_size(),
                'embedding_model': self.embedding_generator.model_name,
                'embedding_dimension': self.embedding_generator.embedding_dim
            })
        
        return stats


# Export main service
__all__ = ['MemoryService', 'EmbeddingGenerator', 'SimilaritySearchEngine']
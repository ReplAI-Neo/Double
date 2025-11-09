#!/usr/bin/env python3
"""
Conversation retrieval system using Neo4j GraphRAG
Supports semantic search via embeddings and tag-based filtering
"""

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import argparse
from typing import List, Dict, Any, Optional


class ConversationRetriever:
    """Retrieve conversations using vector similarity and graph filtering"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password123",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"Loading embedding model '{embedding_model}'...")
        self.embedder = SentenceTransformer(embedding_model)
        print("✓ Retriever initialized")
    
    def close(self):
        self.driver.close()
    
    def vector_search(
        self,
        query: str,
        limit: int = 10,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant conversations using vector similarity.
        
        Args:
            query: Search query
            limit: Number of results to return
            tags: Optional filters like {'purpose': 'debugging', 'tone': 'serious'}
        """
        # Embed the query
        print(f"\nSearching for: '{query}'")
        query_embedding = self.embedder.encode(query).tolist()
        
        with self.driver.session() as session:
            # Build tag filter clause
            tag_filter = ""
            if tags:
                tag_conditions = []
                for category, value in tags.items():
                    tag_conditions.append(
                        f"EXISTS {{ MATCH (c)-[:HAS_TAG]->(t:Tag {{category: '{category}', name: '{value}'}}) }}"
                    )
                tag_filter = "WHERE " + " AND ".join(tag_conditions)
            
            # Vector similarity search on messages
            query = f"""
                CALL db.index.vector.queryNodes('message_embedding_index', $limit * 10, $embedding)
                YIELD node as m, score
                MATCH (c:Conversation)-[:HAS_MESSAGE]->(m)
                WHERE c.total_messages < 500
                {tag_filter if tag_filter else ''}
                WITH c, m, score, 
                     [(c)-[:HAS_TAG]->(t:Tag) WHERE t.category IN 
                      ['purpose', 'tone', 'complexity', 'resolution', 'domain'] | 
                      t.category + ':' + t.name] as tags
                ORDER BY score DESC
                LIMIT $limit
                RETURN c.conversation_id as conversation_id,
                       c.total_messages as total_messages,
                       m.message_id as matching_message_id,
                       m.content as matching_message,
                       score,
                       tags
            """
            
            result = session.run(query, embedding=query_embedding, limit=limit)
            results = []
            
            for record in result:
                results.append({
                    'conversation_id': record['conversation_id'],
                    'total_messages': record['total_messages'],
                    'matching_message_id': record['matching_message_id'],
                    'matching_message': record['matching_message'],
                    'score': record['score'],
                    'tags': record['tags'],
                })
            
            return results
    
    def get_conversation_context(
        self, 
        conversation_id: str,
        matching_message_id: str = None,
        context_window: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation context around a matching message"""
        with self.driver.session() as session:
            if matching_message_id:
                # Get messages around the matching message
                result = session.run("""
                    MATCH (c:Conversation {conversation_id: $conv_id})-[:HAS_MESSAGE]->(match:Message {message_id: $match_id})
                    MATCH (c)-[:HAS_MESSAGE]->(m:Message)
                    WITH match, m, c
                    ORDER BY m.timestamp
                    WITH match, collect(m) as all_messages
                    
                    // Find index of matching message
                    UNWIND range(0, size(all_messages)-1) as idx
                    WITH match, all_messages, idx
                    WHERE all_messages[idx].message_id = match.message_id
                    
                    // Get window around it
                    WITH all_messages, idx, 
                         CASE WHEN idx - $before < 0 THEN 0 ELSE idx - $before END as start_idx,
                         CASE WHEN idx + $after >= size(all_messages) THEN size(all_messages)-1 ELSE idx + $after END as end_idx
                    
                    UNWIND range(start_idx, end_idx) as i
                    WITH all_messages[i] as m, i, start_idx + ($before) as match_idx
                    RETURN m.content as content,
                           m.author as author,
                           m.role as role,
                           m.timestamp as timestamp,
                           i = match_idx as is_match
                    ORDER BY i
                """, conv_id=conversation_id, match_id=matching_message_id, 
                     before=context_window//2, after=context_window//2)
            else:
                # Fallback: get first messages
                result = session.run("""
                    MATCH (c:Conversation {conversation_id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
                    RETURN m.content as content,
                           m.author as author,
                           m.role as role,
                           m.timestamp as timestamp,
                           false as is_match
                    ORDER BY m.timestamp
                    LIMIT $limit
                """, conv_id=conversation_id, limit=context_window)
            
            return [dict(record) for record in result]
    
    def search_and_display(
        self,
        query: str,
        limit: int = 5,
        tags: Optional[Dict[str, str]] = None,
        show_context: bool = True,
    ):
        """Search and display results in a readable format"""
        results = self.vector_search(query, limit=limit, tags=tags)
        
        if not results:
            print("\nNo results found!")
            return
        
        print(f"\nFound {len(results)} relevant conversations")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─'*80}")
            print(f"RESULT {i} (similarity: {result['score']:.3f})")
            print(f"{'─'*80}")
            print(f"Conversation ID: {result['conversation_id'][:40]}...")
            print(f"Total Messages: {result['total_messages']}")
            print(f"Tags: {', '.join(result['tags'][:5])}")
            
            print(f"\nMatching message:")
            content = result['matching_message'] or '[No content]'
            if len(content) > 300:
                content = content[:300] + "..."
            for line in content.split('\n'):
                print(f"  {line}")
            
            if show_context:
                print(f"\nConversation context (around matching message):")
                messages = self.get_conversation_context(
                    result['conversation_id'],
                    matching_message_id=result.get('matching_message_id'),
                    context_window=10
                )
                for j, msg in enumerate(messages, 1):
                    role = msg['role'] or 'user'
                    author = msg['author'] or 'Unknown'
                    msg_content = msg['content'] or '[No content]'
                    is_match = msg.get('is_match', False)
                    
                    if len(msg_content) > 150:
                        msg_content = msg_content[:150] + "..."
                    
                    emoji = "🤖" if role == "assistant" else "👤"
                    match_indicator = " ⭐ [MATCH]" if is_match else ""
                    print(f"\n  {emoji} Message {j} ({author}){match_indicator}:")
                    for line in msg_content.split('\n')[:3]:
                        print(f"     {line}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Retrieve conversations using GraphRAG")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--purpose", help="Filter by purpose (debugging, learning, etc.)")
    parser.add_argument("--tone", help="Filter by tone (humorous, serious, neutral)")
    parser.add_argument("--complexity", help="Filter by complexity (beginner, intermediate, advanced)")
    parser.add_argument("--domain", help="Filter by domain (programming, web_dev, etc.)")
    parser.add_argument("--no-context", action="store_true", help="Don't show conversation context")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    retriever = ConversationRetriever()
    
    try:
        if args.interactive or not args.query:
            # Interactive mode
            print("\nInteractive conversation search")
            print("="*80)
            print("Type your questions to search. Type 'exit' to quit.\n")
            
            while True:
                query = input("Query: ").strip()
                
                if not query or query.lower() == 'exit':
                    break
                
                # Ask for optional filters
                tags = {}
                
                purpose = input("  Filter by purpose (or press Enter to skip): ").strip()
                if purpose:
                    tags['purpose'] = purpose
                
                tone = input("  Filter by tone (or press Enter to skip): ").strip()
                if tone:
                    tags['tone'] = tone
                
                retriever.search_and_display(
                    query,
                    limit=args.limit,
                    tags=tags if tags else None,
                    show_context=not args.no_context,
                )
        else:
            # Command-line mode
            tags = {}
            if args.purpose:
                tags['purpose'] = args.purpose
            if args.tone:
                tags['tone'] = args.tone
            if args.complexity:
                tags['complexity'] = args.complexity
            if args.domain:
                tags['domain'] = args.domain
            
            retriever.search_and_display(
                args.query,
                limit=args.limit,
                tags=tags if tags else None,
                show_context=not args.no_context,
            )
    
    finally:
        retriever.close()
        print("\n👋 Goodbye!\n")


if __name__ == '__main__':
    main()


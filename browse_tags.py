#!/usr/bin/env python3
"""
Interactive tag browser for Neo4j
"""

import os
from neo4j import GraphDatabase

# Load credentials from environment variables with defaults
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def browse_categories():
    """Show all tag categories"""
    with driver.session() as session:
        result = session.run("""
            MATCH (t:Tag)
            RETURN DISTINCT t.category as category
            ORDER BY category
        """)
        categories = [r["category"] for r in result]

        print("\n" + "=" * 80)
        print("TAG CATEGORIES")
        print("=" * 80)
        for i, cat in enumerate(categories, 1):
            print(f"  {i:2d}. {cat}")
        print("=" * 80)
        return categories


def browse_tags_in_category(category):
    """Show all tags in a specific category"""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (c:Conversation)-[:HAS_TAG]->(t:Tag {category: $cat})
            RETURN t.name as tag, count(c) as count
            ORDER BY count DESC
        """,
            cat=category,
        )

        print(f"\n{'=' * 80}")
        print(f"TAGS IN CATEGORY: {category.upper()}")
        print("=" * 80)

        tags = []
        for r in result:
            pct = (r["count"] / 6748) * 100
            print(f"  • {r['tag']:30s} → {r['count']:5,} conversations ({pct:5.1f}%)")
            tags.append(r["tag"])

        print("=" * 80)
        return tags


def browse_conversations_with_tag(category, tag):
    """Show sample conversations with a specific tag"""
    with driver.session() as session:
        # First get conversations with tags
        result = session.run(
            """
            MATCH (c:Conversation)-[:HAS_TAG]->(t:Tag {category: $cat, name: $tag})
            MATCH (c)-[:HAS_TAG]->(all_tags:Tag)
            WITH c, collect(all_tags.category + ':' + all_tags.name) as tags
            RETURN c.conversation_id as id, 
                   c.total_messages as msgs,
                   tags
            LIMIT 5
        """,
            cat=category,
            tag=tag,
        )

        conversations = list(result)

        print(f"\n{'=' * 80}")
        print(f"SAMPLE CONVERSATIONS: {category}:{tag}")
        print("=" * 80)

        for i, r in enumerate(conversations, 1):
            print(f"\n{'─' * 80}")
            print(f"  CONVERSATION {i}:")
            print(f"  ID: {r['id'][:40]}...")
            print(f"  Total Messages: {r['msgs']}")

            # Group tags by category
            tag_dict = {}
            for t in r["tags"]:
                cat_name, value = t.split(":", 1)
                if cat_name not in tag_dict:
                    tag_dict[cat_name] = []
                tag_dict[cat_name].append(value)

            print("\n  🏷️  Tags:")
            for cat_name in sorted(tag_dict.keys()):
                if cat_name in [
                    "formality",
                    "tone",
                    "sentiment",
                    "friendliness",
                    "purpose",
                    "complexity",
                    "resolution",
                    "engagement",
                    "domain",
                    "urgency",
                ]:
                    print(f"    • {cat_name}: {', '.join(tag_dict[cat_name])}")

            # Fetch messages for this conversation
            msg_result = session.run(
                """
                MATCH (c:Conversation {conversation_id: $conv_id})-[:HAS_MESSAGE]->(m:Message)
                RETURN m.content as content, 
                       m.author as author, 
                       m.timestamp as timestamp,
                       m.role as role
                ORDER BY m.timestamp
                LIMIT 10
            """,
                conv_id=r["id"],
            )

            messages = list(msg_result)

            print(f"\n  💬 Messages (showing first {len(messages)}):")
            print(f"  {'─' * 76}")

            for j, msg in enumerate(messages, 1):
                author = msg["author"] or "Unknown"
                msg["role"] or "user"
                content = msg["content"] or "[No content]"

                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."

                print(f"\nMessage {j} ({author}):")

                # Print message with proper indentation
                for line in content.split("\n"):
                    print(f"     {line}")

            print(f"\n  {'─' * 76}")

        print("\n" + "=" * 80)


def main():
    print("\n🏷️  NEO4J TAG BROWSER")
    print("=" * 80)

    while True:
        print("\n1. Browse all tag categories")
        print("2. Browse tags in a specific category")
        print("3. View conversations with a specific tag")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            browse_categories()

        elif choice == "2":
            categories = browse_categories()
            cat_num = input(f"\nEnter category number (1-{len(categories)}): ").strip()
            try:
                category = categories[int(cat_num) - 1]
                browse_tags_in_category(category)
            except (ValueError, IndexError):
                print("Invalid choice!")

        elif choice == "3":
            categories = browse_categories()
            cat_num = input(f"\nEnter category number (1-{len(categories)}): ").strip()
            try:
                category = categories[int(cat_num) - 1]
                tags = browse_tags_in_category(category)
                tag_name = input(
                    f"\nEnter tag name (or number 1-{len(tags)}): "
                ).strip()

                # Allow entering by number or name
                try:
                    tag = tags[int(tag_name) - 1]
                except (ValueError, IndexError):
                    tag = tag_name

                browse_conversations_with_tag(category, tag)
            except (ValueError, IndexError):
                print("Invalid choice!")

        elif choice == "4":
            break

        else:
            print("Invalid choice!")

    driver.close()
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()

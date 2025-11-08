# Parser Output Schema

Complete schema for the parsed conversation data for discord, imessage, and instagram.

## Top-Level Structure

The output is a JSON array where each element represents one conversation

## Conversation Object

```typescript
{
  // OpenAI-compatible messages (content only, role-based)
  openai_messages: Array<OpenAIMessage>,
  
  // Full Discord metadata for each message
  full_metadata_messages: Array<FullMetadataMessage>,
  
  // Timestamps
  first_message_timestamp: string,  // ISO 8601 format
  last_message_timestamp: string,   // ISO 8601 format
  
  // Participants
  recipients: Array<string>,        // List of recipient names (excluding self)
  num_participants: number,         // Total unique participants (including self)
  
  // Message counts
  total_messages: number,           // Total messages in this conversation
  
  // Source and conversation info
  source: string,                   // "discord", "imessage", "instagram"
  chat_type: string,             // "direct", "group"
}
```

---

## OpenAIMessage Object

Simple message format compatible with OpenAI's chat API.

```typescript
{
  role: "user" | "assistant",  // "assistant" = you, "user" = others
  content: string              // Message text content
}
```

**Note**: Only messages with content are included. Empty messages (system messages, etc.) are filtered out.

---

## FullMetadataMessage Object

Complete message data with all metadata preserved.

```typescript
{
  // Message identifiers
  message_id: string,              // message ID
  
  // Timestamps
  timestamp: string,               // ISO 8601 format timestamp
  
  // Content
  content: string,                 // Message text content
  
  // Author info
  author: string,
}
```

## Key Points

### Message Roles

- **"assistant"**: Messages sent by you
- **"user"**: Messages sent by others

### Empty Values

- Arrays are empty `[]` if no items present
- Optional fields are `null` if not present
- Empty messages are excluded from `openai_messages` but present in `full_metadata_messages`

### Timestamps

- All timestamps in ISO 8601 format with timezone: `YYYY-MM-DDTHH:MM:SS.sss+00:00`
- Timezone is typically UTC (`+00:00`)

### Chat Types

- **direct**: 1-on-1 direct message
- **group**: Group chat with 3+ people
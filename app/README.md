# Beeper Message Agent

Send messages and run an AI agent using the Beeper Desktop API.

## Setup

1. Create a `.env` file in this directory:

```bash
BEEPER_ACCESS_TOKEN="your_token_here"
OPENAI_API_KEY="your_openai_key_here"
TARGET_PHONE="+1 123-456-7890"
```

2. Get your Beeper access token:

   - Make sure Beeper Desktop API is enabled and running
   - Open Settings → Developers in Beeper Desktop
   - Click the "+" button next to "Approved connections"
   - Follow the instructions to create your token
   - Copy the token and paste it into your `.env` file

3. Add your OpenAI API key to the `.env` file

4. Install dependencies:

```bash
npm install
```

## Usage

### Send a single message

Run the script to send a test message:

```bash
npm run send
```

### Run the AI agent

Start the conversational agent that monitors and responds to messages:

```bash
npm run agent-v3
```

The agent will:

- Monitor messages across all iMessage chats
- Generate responses using GPT-5-nano
- Send responses automatically
- Keep a conversation history for context
- Run continuously until you press Ctrl+C

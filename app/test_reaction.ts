import BeeperDesktop from "@beeper/desktop-api";
import dotenv from "dotenv";

dotenv.config();

async function testReaction() {
  const accessToken = process.env["BEEPER_ACCESS_TOKEN"];
  const targetPhone = process.env["TARGET_PHONE"];

  if (!accessToken || !targetPhone) {
    console.error("❌ Error: Missing environment variables");
    process.exit(1);
  }

  const client = new BeeperDesktop({ accessToken });

  try {
    // Find the chat
    console.log(`Searching for ${targetPhone}...`);
    const chatResults = await client.chats.search({
      query: targetPhone,
      limit: 1,
      type: "single",
    });

    if (chatResults.items.length === 0) {
      console.error(`Could not find chat`);
      process.exit(1);
    }

    const chatId = chatResults.items[0].id;
    console.log(`Found chat: ${chatId}`);

    // Get the most recent message
    const encodedChatId = encodeURIComponent(chatId);
    const messagesResponse = (await client.get(
      `/v1/chats/${encodedChatId}/messages`,
      { query: { limit: 1 } }
    )) as any;

    const lastMessage = messagesResponse.items?.[0];
    if (!lastMessage) {
      console.error("No messages found");
      process.exit(1);
    }

    console.log(`Last message ID: ${lastMessage.id}`);
    console.log(`Last message text: "${lastMessage.text}"`);

    // Try to add a reaction
    console.log(`\nAttempting to add ❤️ reaction...`);

    // Try method 1: Using a potential messages.addReaction method
    try {
      if ((client.messages as any).addReaction) {
        await (client.messages as any).addReaction({
          chatId: chatId,
          messageId: lastMessage.id,
          reactionKey: "❤️",
        });
        console.log("✅ Reaction added via addReaction method!");
        return;
      }
    } catch (e) {
      console.log("addReaction method failed or doesn't exist");
    }

    // Try method 2: Direct POST to potential reactions endpoint
    try {
      const response = await client.post(
        `/v1/chats/${encodedChatId}/messages/${encodeURIComponent(
          lastMessage.id
        )}/reactions`,
        {
          body: {
            reactionKey: "❤️",
          },
        }
      );
      console.log("✅ Reaction added via POST endpoint!");
      console.log("Response:", response);
      return;
    } catch (e: any) {
      console.log(`POST /reactions failed: ${e.status || e.message}`);
    }

    // Try method 3: PUT endpoint
    try {
      const response = await client.put(
        `/v1/chats/${encodedChatId}/messages/${encodeURIComponent(
          lastMessage.id
        )}/reactions/❤️`,
        {}
      );
      console.log("✅ Reaction added via PUT endpoint!");
      console.log("Response:", response);
      return;
    } catch (e: any) {
      console.log(`PUT /reactions failed: ${e.status || e.message}`);
    }

    console.log("\n❌ Unable to find working reaction endpoint");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

testReaction();

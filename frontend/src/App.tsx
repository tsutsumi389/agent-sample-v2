import { useCallback, useEffect, useState } from "react";
import ChatPanel from "./components/ChatPanel";
import MemoryPanel from "./components/MemoryPanel";
import { fetchMemories } from "./api";
import type { MemoryItem } from "./types";

function newThreadId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const [userId, setUserId] = useState("alice");
  const [threadId, setThreadId] = useState(newThreadId);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [loadingMem, setLoadingMem] = useState(false);

  const refreshMemories = useCallback(async () => {
    setLoadingMem(true);
    try {
      setMemories(await fetchMemories(userId));
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingMem(false);
    }
  }, [userId]);

  useEffect(() => {
    refreshMemories();
  }, [refreshMemories]);

  const newConversation = () => setThreadId(newThreadId());

  return (
    <div className="app">
      <header className="header">
        <h1>🧠 長期記憶エージェント デモ</h1>
        <div className="controls">
          <label>
            user_id:
            <input
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </label>
          <span className="thread">thread: {threadId.slice(0, 8)}…</span>
          <button onClick={newConversation}>新しい会話</button>
        </div>
      </header>
      <main className="layout">
        <ChatPanel
          userId={userId}
          threadId={threadId}
          onAfterReply={refreshMemories}
        />
        <MemoryPanel
          userId={userId}
          memories={memories}
          loading={loadingMem}
          onRefresh={refreshMemories}
        />
      </main>
    </div>
  );
}

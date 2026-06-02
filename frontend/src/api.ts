import type { ChatResponse, MemoryItem } from "./types";

export async function sendChat(
  userId: string,
  threadId: string,
  message: string,
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, thread_id: threadId, message }),
  });
  if (!res.ok) throw new Error(`chat failed: ${res.status}`);
  return res.json();
}

export async function fetchMemories(userId: string): Promise<MemoryItem[]> {
  const res = await fetch(`/api/memories?user_id=${encodeURIComponent(userId)}`);
  if (!res.ok) throw new Error(`memories failed: ${res.status}`);
  return res.json();
}

export async function deleteMemory(userId: string, key: string): Promise<void> {
  const params = new URLSearchParams({ user_id: userId, key });
  const res = await fetch(`/api/memories?${params.toString()}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`delete failed: ${res.status}`);
}

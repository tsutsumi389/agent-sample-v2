import type { MemoryItem } from "./types";

/**
 * チャット応答を SSE でストリーミング受信する。
 * トークンが届くたびに onToken を呼び出す。POST のため EventSource は使えず、
 * fetch の ReadableStream を手動で SSE パースする。
 */
export async function streamChat(
  userId: string,
  threadId: string,
  message: string,
  onToken: (text: string) => void,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, thread_id: threadId, message }),
  });
  if (!res.ok || !res.body) throw new Error(`chat failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE はイベント間を空行（\n\n）で区切る
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const lines = part.split("\n");
      const event = lines.find((l) => l.startsWith("event:"))?.slice(6).trim();
      const dataLine = lines.find((l) => l.startsWith("data:"))?.slice(5).trim();
      if (!dataLine) continue;
      const data = JSON.parse(dataLine);
      if (event === "token") onToken(data.text);
      else if (event === "error") throw new Error(data.message);
      // "done" は読み切りで自然終了
    }
  }
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

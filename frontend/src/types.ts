export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface MemoryItem {
  key: string;
  value: unknown;
  created_at: string | null;
  updated_at: string | null;
}

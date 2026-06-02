import { deleteMemory } from "../api";
import type { MemoryItem } from "../types";

interface Props {
  userId: string;
  memories: MemoryItem[];
  loading: boolean;
  onRefresh: () => void;
}

function renderValue(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    // LangMem は {"content": "..."} 形式で保存することが多い
    if (typeof obj.content === "string") return obj.content;
    return JSON.stringify(value);
  }
  return String(value);
}

export default function MemoryPanel({
  userId,
  memories,
  loading,
  onRefresh,
}: Props) {
  const remove = async (key: string) => {
    await deleteMemory(userId, key);
    onRefresh();
  };

  return (
    <section className="panel memory">
      <div className="memory-head">
        <h2>長期記憶（{userId}）</h2>
        <button onClick={onRefresh} disabled={loading}>
          {loading ? "更新中…" : "更新"}
        </button>
      </div>
      {memories.length === 0 ? (
        <p className="hint">まだ記憶はありません。会話すると蓄積されます。</p>
      ) : (
        <ul className="memory-list">
          {memories.map((m) => (
            <li key={m.key} className="memory-item">
              <div className="memory-value">{renderValue(m.value)}</div>
              <div className="memory-meta">
                <span>{m.updated_at?.slice(0, 19).replace("T", " ")}</span>
                <button className="del" onClick={() => remove(m.key)}>
                  削除
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

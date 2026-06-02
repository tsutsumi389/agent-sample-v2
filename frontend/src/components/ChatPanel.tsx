import { useState } from "react";
import { sendChat } from "../api";
import type { ChatMessage } from "../types";

interface Props {
  userId: string;
  threadId: string;
  onAfterReply: () => void;
}

export default function ChatPanel({ userId, threadId, onAfterReply }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    setBusy(true);
    try {
      const res = await sendChat(userId, threadId, text);
      setMessages((m) => [...m, { role: "assistant", content: res.reply }]);
      // 背景抽出には少し遅延があるため、応答直後と数秒後の両方で更新する
      onAfterReply();
      setTimeout(onAfterReply, 3500);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ エラー: ${String(err)}` },
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel chat">
      <h2>チャット</h2>
      <div className="messages">
        {messages.length === 0 && (
          <p className="hint">
            例: 「私の名前は田中で、コーヒーが好きです」と話しかけ、
            別の会話で「私の好きな飲み物は？」と聞いてみてください。
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="role">{m.role === "user" ? "🧑" : "🤖"}</span>
            <span className="content">{m.content}</span>
          </div>
        ))}
        {busy && <div className="msg assistant">🤖 考え中…</div>}
      </div>
      <form className="composer" onSubmit={submit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="メッセージを入力…"
          disabled={busy}
        />
        <button type="submit" disabled={busy}>
          送信
        </button>
      </form>
    </section>
  );
}

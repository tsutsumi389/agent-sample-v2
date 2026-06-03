import { useState } from "react";
import { streamChat } from "../api";
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

    // ユーザー発話と、ストリーミング先の空 assistant メッセージを用意する
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setBusy(true);
    try {
      await streamChat(userId, threadId, text, (token) => {
        // 末尾の assistant メッセージにトークンを追記していく
        setMessages((m) => {
          const next = [...m];
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, content: last.content + token };
          return next;
        });
      });
      // 背景抽出には少し遅延があるため、応答直後と数秒後の両方で更新する
      onAfterReply();
      setTimeout(onAfterReply, 3500);
    } catch (err) {
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = {
          role: "assistant",
          content: `⚠️ エラー: ${String(err)}`,
        };
        return next;
      });
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
        {messages.map((m, i) => {
          // ストリーミング待ちの空 assistant バブルは「考え中…」表示に置き換える
          const isPendingAssistant =
            busy &&
            m.role === "assistant" &&
            m.content === "" &&
            i === messages.length - 1;
          return (
            <div key={i} className={`msg ${m.role}`}>
              <span className="role">{m.role === "user" ? "🧑" : "🤖"}</span>
              <span className="content">
                {isPendingAssistant ? "考え中…" : m.content}
              </span>
            </div>
          );
        })}
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

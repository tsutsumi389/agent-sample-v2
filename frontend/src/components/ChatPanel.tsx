import { useState } from "react";
import { streamChat } from "../api";
import type { ChatMessage, PhaseEvent, ProgressEntry } from "../types";

interface Props {
  userId: string;
  threadId: string;
  onAfterReply: () => void;
}

/** 現在のフェーズをユーザー向けのステータス文言にする */
function phaseLabel(phase: PhaseEvent): string {
  const labels = {
    planner: "計画中…",
    executor: "回答生成中…",
    evaluator: "評価中…",
  } as const;
  return `${labels[phase.node]}（試行 ${phase.iteration}）`;
}

/** 進捗エントリ1件の表示 */
function ProgressLine({ entry }: { entry: ProgressEntry }) {
  if (entry.kind === "plan") {
    return (
      <div className="progress-line plan-box">
        <span className="progress-tag">📝 計画（試行 {entry.iteration}）</span>
        <pre>{entry.text}</pre>
      </div>
    );
  }
  if (entry.kind === "evaluation") {
    const ok = entry.verdict === "ok";
    return (
      <div className={`progress-line ${ok ? "eval-ok" : "eval-ng"}`}>
        <span className="progress-tag">
          {ok ? "✅ 評価: 合格" : "❌ 評価: 不合格"}（試行 {entry.iteration}
          {entry.score != null ? ` / score ${entry.score}` : ""}）
        </span>
        {entry.feedback && <pre>{entry.feedback}</pre>}
      </div>
    );
  }
  return (
    <div className="progress-line retry-note">
      <span className="progress-tag">🔄 再計画します（試行 {entry.iteration}）</span>
    </div>
  );
}

export default function ChatPanel({ userId, threadId, onAfterReply }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<PhaseEvent | null>(null);

  // 末尾（ストリーミング中の assistant）メッセージを不変更新するヘルパー
  const updateLast = (fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((m) => {
      const next = [...m];
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  };

  const appendProgress = (entry: ProgressEntry) =>
    updateLast((m) => ({ ...m, progress: [...(m.progress ?? []), entry] }));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;

    // ユーザー発話と、ストリーミング先の空 assistant メッセージを用意する
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "", progress: [] },
    ]);
    setInput("");
    setBusy(true);
    setPhase(null);
    try {
      await streamChat(userId, threadId, text, {
        // Executor（最終回答）のトークンを末尾の assistant メッセージへ追記
        onToken: (token) =>
          updateLast((m) => ({ ...m, content: m.content + token })),
        onPhase: (p) => {
          setPhase(p);
          // 再試行の回答生成が始まる直前に不合格ドラフトを破棄し、
          // 最後に合格（または採用）された回答だけが残るようにする
          if (p.node === "executor" && p.iteration > 1) {
            updateLast((m) => (m.content ? { ...m, content: "" } : m));
          }
        },
        onPlan: (p) =>
          appendProgress({ kind: "plan", iteration: p.iteration, text: p.text }),
        onEvaluation: (ev) =>
          appendProgress({
            kind: "evaluation",
            iteration: ev.iteration,
            verdict: ev.verdict,
            score: ev.score,
            feedback: ev.feedback,
          }),
        onRetry: (r) => appendProgress({ kind: "retry", iteration: r.iteration }),
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
      setPhase(null);
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
          // ストリーミング待ちの空 assistant バブルは現在フェーズの表示に置き換える
          const isPendingAssistant =
            busy &&
            m.role === "assistant" &&
            m.content === "" &&
            i === messages.length - 1;
          const hasProgress = (m.progress?.length ?? 0) > 0;
          return (
            <div key={i} className={`msg ${m.role}`}>
              <span className="role">{m.role === "user" ? "🧑" : "🤖"}</span>
              <span className="content">
                {hasProgress && (
                  <details className="progress">
                    <summary>エージェントの動き</summary>
                    {m.progress!.map((entry, j) => (
                      <ProgressLine key={j} entry={entry} />
                    ))}
                  </details>
                )}
                {isPendingAssistant
                  ? phase
                    ? phaseLabel(phase)
                    : "考え中…"
                  : m.content}
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

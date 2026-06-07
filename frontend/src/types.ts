export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** assistant メッセージ生成時のエージェント進捗（計画・評価・リトライ） */
  progress?: ProgressEntry[];
}

/** マルチエージェント（Planner→Executor→Evaluator）のフェーズ名 */
export type AgentPhase = "planner" | "executor" | "evaluator";

/** SSE "phase" イベント: 各エージェントの処理開始通知 */
export interface PhaseEvent {
  node: AgentPhase;
  iteration: number;
}

/** SSE "plan" イベント: Planner が確定した計画 */
export interface PlanEvent {
  text: string;
  iteration: number;
}

/** SSE "evaluation" イベント: Evaluator の評価結果 */
export interface EvaluationEvent {
  verdict: "ok" | "ng";
  score: number | null;
  feedback: string;
  iteration: number;
}

/** SSE "retry" イベント: 不合格による再計画の通知 */
export interface RetryEvent {
  iteration: number;
  reason: string;
}

/** チャット1往復分の進捗履歴（計画・評価・リトライを時系列に保持） */
export interface ProgressEntry {
  kind: "plan" | "evaluation" | "retry";
  iteration: number;
  text?: string;
  verdict?: "ok" | "ng";
  score?: number | null;
  feedback?: string;
}

export interface MemoryItem {
  key: string;
  value: unknown;
  created_at: string | null;
  updated_at: string | null;
}

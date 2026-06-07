import json
import re
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from app.config import settings
from app.llm import get_chat_model

# 記憶ツールの使い方ガイド。Planner / Executor のシステムプロンプトで共有する
# （gemma4 のようなローカルモデルでもツール呼び出しを行いやすくするため明示する）。
# プロアクティブ想起（recall ノード）が毎ターン記憶を注入するため、search_memory は
# フォールバック用途。
MEMORY_GUIDE = """記憶の扱い方:
- ユーザーに関する新しい事実（名前・好み・所属・予定など）を知ったら、
  manage_memory ツールで保存してください。
- <user_profile> タグ内は確定済みのユーザー基本情報（JSON）です。常にこれを
  踏まえてパーソナライズしてください。
- <related_memories> タグ内は今の話題に関連して過去の会話から想起した記憶
  （JSON、1行1件）です。関連する場合のみ活用し、情報が不足する場合のみ
  search_memory で補ってください。"""

PLANNER_SYSTEM_PROMPT = f"""あなたは回答計画の立案担当（プランナー）です。
ユーザーの最新の要求を達成するための回答方針を設計してください。

ルール:
- 出力は計画のみ。ユーザーへの回答そのものは書かないでください。
- 計画は3〜6項目程度の簡潔な日本語の箇条書きで、回答に含めるべき内容・
  トーン・構成を示してください。
- 挨拶や雑談のような単純な要求でも、短い計画（1〜2項目）を出してください。

{MEMORY_GUIDE}"""

EXECUTOR_SYSTEM_PROMPT = f"""あなたは回答の実行担当（エグゼキューター）で、長期記憶を持つ
親切な日本語アシスタントです。<plan> タグ内の計画に従い、ユーザーへの最終回答を
作成してください。この出力がそのままユーザーに表示されます。

ルール:
- 計画の項目を満たしつつ、自然な日本語で回答してください。計画自体は出力しないでください。
- 応答スタイルの指定があれば必ず従ってください。

{MEMORY_GUIDE}"""

EVALUATOR_SYSTEM_PROMPT = """あなたは回答の品質評価担当（エヴァリエーター）です。
会話の文脈・<plan> タグ内の計画・<draft> タグ内の回答ドラフトを読み、ドラフトが
ユーザーの最新の要求を満たしているか評価してください。

評価基準:
- ユーザーの要求への適合性、日本語の自然さ、応答スタイル指定の遵守。
- 挨拶や雑談などの単純な要求では、自然で適切な応答であれば合格（ok）にして
  ください。過度に厳しく評価しないこと。

出力形式（厳守）: 次の JSON のみを出力してください。前後に説明文やコードフェンスを
付けないでください。
{"verdict": "ok" または "ng", "score": 0から100の整数, "feedback": "ngの場合の改善点を日本語で。okなら空文字"}"""


class AgentState(MessagesState):
    """Planner→Executor→Evaluator ループのグラフ state。

    messages（会話履歴 = checkpointer 永続対象）には最終回答のみを追加し、
    計画・ドラフト・評価などのループ内部情報は専用フィールドに分離して履歴を
    汚さない。Planner / Executor の ReAct ツール往復も scratch フィールド上で
    行い、本体 messages とは混ぜない。
    """

    recalled_profile: str
    recalled_memories: str
    plan: str  # Planner の最新計画
    draft: str  # Executor の最新ドラフト回答
    evaluation: dict  # {"verdict": "ok"|"ng", "score": int|None, "feedback": str}
    feedback: str  # NG 時に Planner へ渡す改善指示
    iteration: int  # 試行回数（Planner が計画を確定するたびに +1）
    planner_scratch: Annotated[list[AnyMessage], add_messages]
    executor_scratch: Annotated[list[AnyMessage], add_messages]


def _format_items(items) -> str:
    """store の検索結果アイテム列を、1行1件の JSON（JSONL）に整形する。

    langmem の保存形式（{"kind", "content"} エンベロープ等）をそのまま出すことで、
    形状判定による剥がしロジックを持たない。日本語が \\uXXXX にエスケープされない
    よう ensure_ascii=False を指定する。
    """
    return "\n".join(
        json.dumps(item.value, ensure_ascii=False) for item in items if item.value
    )


def _build_system_text(base: str, recalled_profile: str, recalled_memories: str) -> str:
    """想起済みのプロフィール・記憶ブロックをシステムプロンプトへ合成する。

    プロフィール（常時注入の確定情報）と話題依存の記憶を XML タグで区切り、
    LLM にデータの境界と確度の違いを伝える。
    """
    sys_text = base
    if recalled_profile:
        sys_text += f"\n\n<user_profile>\n{recalled_profile}\n</user_profile>"
    if recalled_memories:
        sys_text += f"\n\n<related_memories>\n{recalled_memories}\n</related_memories>"
    return sys_text


def _message_text(message) -> str:
    """メッセージからテキスト部分を取り出す（マルチパート content にも対応）。"""
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


def _normalize_verdict(raw: str) -> str:
    """Evaluator が返す verdict の表記ゆれを ok / ng に正規化する（不明は空文字）。"""
    v = raw.strip().lower()
    if v in {"ok", "pass", "passed", "yes", "good", "合格", "良い"}:
        return "ok"
    if v in {"ng", "fail", "failed", "no", "bad", "不合格"}:
        return "ng"
    return ""


def parse_evaluation(text: str) -> dict:
    """Evaluator の出力テキストから合否判定を堅牢に抽出する。

    ローカルモデル（gemma4）の structured output は信頼できないため、段階的に
    フォールバックする:
    1. コードフェンスを除去し、最初の { 〜 最後の } を JSON としてパース
    2. verdict の表記ゆれ（yes/合格 等）を正規化
    3. JSON が壊れていればキーワード判定（「不合格」を「合格」より先に判定）
    4. 完全に判定不能なら ng（リトライ上限 MAX_PLAN_ITERATIONS が安全弁）
    """
    cleaned = re.sub(r"```[a-zA-Z]*", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            verdict = _normalize_verdict(str(data.get("verdict", "")))
            if verdict:
                score = data.get("score")
                # bool は int のサブクラスのため明示的に除外する（true → 1 を防ぐ）
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    score = int(score)
                else:
                    score = None
                return {
                    "verdict": verdict,
                    "score": score,
                    "feedback": str(data.get("feedback") or ""),
                }

    # キーワードフォールバック。「不合格」は「合格」を含むため NG 側を先に判定する。
    ng_words = ("不合格", '"ng"', "fail", "やり直し", "修正が必要", "改善が必要")
    ok_words = ("合格", "問題ありません", "問題なし", '"ok"', "pass")
    lowered = text.lower()
    if any(w in lowered for w in ng_words):
        return {"verdict": "ng", "score": None, "feedback": text.strip()}
    if any(w in lowered for w in ok_words):
        return {"verdict": "ok", "score": None, "feedback": ""}
    return {"verdict": "ng", "score": None, "feedback": text.strip()}


def route_after_eval(state, max_iterations: int | None = None) -> str:
    """Evaluator の合否と試行回数から次ノードを決める条件エッジ。

    - ok → finalize（ドラフトを最終回答として確定）
    - ng & 上限到達 → finalize（無限ループ防止: 最後のドラフトを最良として採用）
    - ng & 上限未満 → planner_agent（フィードバック付きで再計画）
    """
    limit = max_iterations if max_iterations is not None else settings.MAX_PLAN_ITERATIONS
    verdict = (state.get("evaluation") or {}).get("verdict")
    if verdict == "ok":
        return "finalize"
    if state.get("iteration", 0) >= limit:
        return "finalize"
    return "planner_agent"


def _scratch_route(state, key: str) -> str:
    """scratch の末尾メッセージにツール呼び出しがあれば tools、なければ next。

    tools_condition は空の messages で例外を送出するため、scratch（計画確定時に
    空へリセットされる）に対しては使わず独自に判定する。
    """
    scratch = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
    last = scratch[-1] if scratch else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "next"


def planner_route(state) -> str:
    """Planner の ReAct 分岐: ツール実行か Executor への遷移かを決める。"""
    return _scratch_route(state, "planner_scratch")


def executor_route(state) -> str:
    """Executor の ReAct 分岐: ツール実行か Evaluator への遷移かを決める。"""
    return _scratch_route(state, "executor_scratch")


def finalize_node(state) -> dict:
    """合格（または上限到達）したドラフトを最終回答として会話履歴に確定する。

    ループ中の計画・評価・途中ドラフトは messages に入れないため、checkpointer に
    残る履歴と reflection（バックグラウンド記憶抽出）への入力は「ユーザー発話 +
    最終回答」だけの綺麗な会話になる。

    ドラフトが空（モデルが内容を返せなかった等）の場合は、空のバブルを
    ユーザーに見せないようフォールバック文言を返す。
    """
    draft = state.get("draft", "")
    if not draft:
        draft = "申し訳ありません。うまく回答を生成できませんでした。もう一度お試しください。"
    return {"messages": [AIMessage(content=draft)]}


async def recall_node(
    state: AgentState, *, store: BaseStore, config: RunnableConfig
) -> dict:
    """ターン冒頭に1回だけ長期記憶を取得し、結果を state に載せるノード。

    LangGraph がノード関数の署名（引数名+型注釈）から store / config を実行時に
    注入する。検索はターンあたり1回で、下流の Planner / Executor は state を読む
    だけにする（ReAct ループ中に再検索しない）。

    - プロフィール: 基本情報なので話題に関係なく常に取得する
    - 一般記憶: 直近のユーザー発話で意味検索し、話題に関連するものを厳選する

    あわせてループ制御フィールド（iteration 等）と、前ターンが異常終了した場合に
    残り得る scratch を初期化する。
    """
    # ループ制御の初期化。scratch は add_messages reducer のため REMOVE_ALL で消す。
    loop_init = {
        "plan": "",
        "draft": "",
        "evaluation": {},
        "feedback": "",
        "iteration": 0,
        "planner_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        "executor_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }

    user_id = config["configurable"].get("user_id")
    if not user_id:
        return {"recalled_profile": "", "recalled_memories": "", **loop_init}

    # 直近のユーザー発話を意味検索クエリにする（無ければ通常検索にフォールバック）
    last_user = next(
        (m for m in reversed(state["messages"]) if getattr(m, "type", None) == "human"),
        None,
    )
    # マルチパート content（list 形式）にも対応するため _message_text で文字列化する
    query = _message_text(last_user) if last_user else None

    # プロアクティブ想起は付加機能。検索失敗（DB 一時断など）で会話全体を
    # 落とさず、記憶なしとして応答を継続する。
    try:
        # プロフィール = 常時注入: query なし（類似度ランキング非依存）で必ず取得する。
        # 単一の集約オブジェクト前提のため limit は安全弁（複数件できた場合の注入上限）。
        profile_items = await store.asearch(("profile", user_id), limit=5)
        # 一般記憶 = 話題依存: 意味検索で上位5件に厳選する
        memory_items = await store.asearch(("memories", user_id), query=query, limit=5)
    except Exception:  # noqa: BLE001
        return {"recalled_profile": "", "recalled_memories": "", **loop_init}

    return {
        "recalled_profile": _format_items(profile_items),
        "recalled_memories": _format_items(memory_items),
        **loop_init,
    }


def build_agent(store, checkpointer):
    """Planner→Executor→Evaluator ループ構成のマルチエージェントグラフを構築する。

    - store: 長期記憶（pgvector 意味検索付き）
    - checkpointer: 会話履歴（thread_id 単位の短期記憶）

    グラフ構成:
        START → recall → planner_agent ⇄ planner_tools
                              ↓
                         executor_agent ⇄ executor_tools
                              ↓
                         evaluator ─ ng（上限未満）→ planner_agent（再計画）
                              └─ ok / 上限到達 → finalize → END

    - recall: ターン冒頭に1回だけ記憶を取得し state に載せる（プロアクティブ想起）
    - planner_agent: 回答計画を立てる。記憶ツール使用可（planner_scratch 上で ReAct）
    - executor_agent: 計画に従い最終回答ドラフトを生成。記憶ツール使用可。
      SSE でトークンを配信する唯一のノード
    - evaluator: ドラフトを評価し、不合格ならフィードバック付きで Planner へ戻す
    - finalize: 確定ドラフトのみを messages（会話履歴）に追加する
    """
    tools = [
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ]
    llm_with_tools = get_chat_model().bind_tools(tools)
    llm_plain = get_chat_model()

    def _pick_model(scratch):
        """ツール往復が上限に達したら、ツールなしモデルで強制的に最終出力させる。

        ローカルモデルがツールを呼び続ける暴走の安全弁。AIMessage の数 = これまでの
        ツール往復回数。
        """
        tool_turns = sum(1 for m in scratch if isinstance(m, AIMessage))
        return llm_with_tools if tool_turns < settings.MAX_TOOL_TURNS else llm_plain

    async def planner_agent(state: AgentState) -> dict:
        """ユーザー要求と記憶から回答計画を立てるノード（scratch 上で ReAct）。"""
        scratch = state.get("planner_scratch") or []
        sys_text = _build_system_text(
            PLANNER_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        feedback = state.get("feedback", "")
        if feedback:
            sys_text += (
                f"\n\n<evaluator_feedback>\n{feedback}\n</evaluator_feedback>\n"
                "前回の計画に基づく回答は上記の点で不十分と評価されました。"
                "フィードバックを反映した改善計画を立ててください。"
            )
        prompt = [SystemMessage(content=sys_text)] + state["messages"] + list(scratch)
        response = await _pick_model(scratch).ainvoke(prompt)
        if getattr(response, "tool_calls", None):
            return {"planner_scratch": [response]}
        return {
            "plan": _message_text(response),
            "iteration": state.get("iteration", 0) + 1,
            "planner_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    async def executor_agent(state: AgentState) -> dict:
        """計画に従ってユーザーへの最終回答ドラフトを生成するノード（scratch 上で ReAct）。"""
        scratch = state.get("executor_scratch") or []
        sys_text = _build_system_text(
            EXECUTOR_SYSTEM_PROMPT,
            state.get("recalled_profile", ""),
            state.get("recalled_memories", ""),
        )
        sys_text += f"\n\n<plan>\n{state.get('plan', '')}\n</plan>"
        prompt = [SystemMessage(content=sys_text)] + state["messages"] + list(scratch)
        response = await _pick_model(scratch).ainvoke(prompt)
        if getattr(response, "tool_calls", None):
            return {"executor_scratch": [response]}
        return {
            "draft": _message_text(response),
            "executor_scratch": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }

    async def evaluator_node(state: AgentState) -> dict:
        """計画とドラフトを評価し、合否とフィードバックを state に書くノード。"""
        sys_text = (
            EVALUATOR_SYSTEM_PROMPT
            + f"\n\n<plan>\n{state.get('plan', '')}\n</plan>"
            + f"\n\n<draft>\n{state.get('draft', '')}\n</draft>"
        )
        response = await llm_plain.ainvoke(
            [SystemMessage(content=sys_text)] + state["messages"]
        )
        evaluation = parse_evaluation(_message_text(response))
        feedback = evaluation["feedback"] if evaluation["verdict"] == "ng" else ""
        return {"evaluation": evaluation, "feedback": feedback}

    builder = StateGraph(AgentState)
    builder.add_node("recall", recall_node)
    builder.add_node("planner_agent", planner_agent)
    builder.add_node("planner_tools", ToolNode(tools, messages_key="planner_scratch"))
    builder.add_node("executor_agent", executor_agent)
    builder.add_node("executor_tools", ToolNode(tools, messages_key="executor_scratch"))
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "planner_agent")
    # Planner の ReAct ループ: ツール呼び出しがあれば実行して戻る、なければ Executor へ
    builder.add_conditional_edges(
        "planner_agent",
        planner_route,
        {"tools": "planner_tools", "next": "executor_agent"},
    )
    builder.add_edge("planner_tools", "planner_agent")
    # Executor の ReAct ループ: 同上。完了したら Evaluator へ
    builder.add_conditional_edges(
        "executor_agent",
        executor_route,
        {"tools": "executor_tools", "next": "evaluator"},
    )
    builder.add_edge("executor_tools", "executor_agent")
    # 評価結果で分岐: 合格/上限到達 → finalize、不合格 → 再計画
    builder.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {"finalize": "finalize", "planner_agent": "planner_agent"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(store=store, checkpointer=checkpointer)

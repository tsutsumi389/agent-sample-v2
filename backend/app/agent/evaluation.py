import json
import re


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

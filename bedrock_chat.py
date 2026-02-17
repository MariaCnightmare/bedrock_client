#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bedrock_chat.py v0.2.2
- Bedrock Converse API chat client (SSO-friendly)
- history save/load (JSONL)
- file attach
- foundation model ID -> inference profile auto-resolve (Nova Pro etc.)
- dev helper: repo snapshot / git diff / run commands & attach outputs (Codex-like)
- cost guard: attachment size limits + diff fallback
- ★FIX: /cmd prints output immediately + attaches output with exit code (no more blind debugging)
- ★NEW: --model alias (e.g. --model nova)
- ★NEW: --repo auto-detect from cwd (and accept repo name like "bedrock_client")
- ★NEW: guard for "cmd ..." (missing leading slash)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_REGION = "ap-northeast-1"
DEFAULT_PROFILE = "apiron-admin"
DEFAULT_MODEL_ID = "amazon.nova-pro-v1:0"

DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9

# ガード（2,000円/月運用前提：貼りすぎない）
MAX_FILE_CHARS = 200_000
MAX_ATTACH_CHARS = 60_000          # 1回の添付合計の上限
MAX_DIFF_CHARS = 45_000            # diff単体の上限（超えたらstatへ）
MAX_CMD_OUTPUT_CHARS = 25_000      # コマンド出力の上限
CMD_SHOW_CHARS = 6_000             # 画面に即表示する上限（長すぎると邪魔なので短め）

DEFAULT_SYSTEM_PROMPT = (
    "あなたは熟練のソフトウェアエンジニアです。"
    "回答はできるだけ短く、実装は最小差分（diff形式）を優先してください。"
    "不確実な点は推測せず、確認すべき事項を箇条書きで提示してください。"
    "ログやエラー出力が与えられた場合は、該当行を引用して根拠を明示してください。"
)

FOUNDATION_ID_RE = re.compile(r"^(amazon|anthropic|cohere|meta|mistral)\..+:\d+$")
INFERENCE_PROFILE_ARN_RE = re.compile(r"^arn:aws:bedrock:[a-z0-9-]+:\d{12}:inference-profile/.+$")

# Codexのノリで短縮指定
MODEL_ALIASES = {
    # Nova Pro（foundation id）→ 既存の推論プロファイル自動解決に流す
    "nova": "amazon.nova-pro-v1:0",
    "nova-pro": "amazon.nova-pro-v1:0",
    # 将来: nova-lite / nova-micro を足したくなったらここに追加
}


@dataclass
class ChatConfig:
    profile: str
    region: str
    model_id: str
    max_output_tokens: int
    temperature: float
    top_p: float
    system_prompt: str
    save_path: Optional[str]
    load_path: Optional[str]
    print_raw: bool
    repo: Optional[str]


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) > MAX_FILE_CHARS:
        raise ValueError(f"ファイルが大きすぎます（{len(text)} chars）。上限 {MAX_FILE_CHARS} chars。")
    return text


def jsonl_append(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return msgs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return msgs


def to_bedrock_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in history:
        role = m.get("role")
        text = m.get("text")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({"role": role, "content": [{"text": text}]})
    return out


def extract_assistant_text(resp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    text_parts: List[str] = []
    output_msg = resp.get("output", {}).get("message", {})
    for c in output_msg.get("content", []) or []:
        if "text" in c and isinstance(c["text"], str):
            text_parts.append(c["text"])
    text = "".join(text_parts).strip()

    usage = resp.get("usage", {}) or {}
    metrics = {
        "inputTokens": usage.get("inputTokens"),
        "outputTokens": usage.get("outputTokens"),
        "totalTokens": usage.get("totalTokens"),
        "latencyMs": resp.get("metrics", {}).get("latencyMs"),
        "stopReason": resp.get("stopReason"),
    }
    return text, metrics


def clamp_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, {len(s)} chars total)\n"


def run_cmd(cmd: List[str], cwd: Optional[str]) -> Tuple[int, str]:
    """
    Returns: (exit_code, combined_output)
    """
    try:
        cp = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        out = cp.stdout or ""
        return int(cp.returncode), out
    except Exception as ex:
        return 127, f"[command failed] {ex}"


def repo_snapshot(repo: str, include_diff: bool = True) -> str:
    lines: List[str] = []
    lines.append(f"[repo] {repo}")

    lines.append("\n$ git status -sb")
    rc, out = run_cmd(["git", "status", "-sb"], cwd=repo)
    lines.append(clamp_text(out, 4000))

    lines.append("\n$ git log -n 5 --oneline")
    rc, out = run_cmd(["git", "log", "-n", "5", "--oneline"], cwd=repo)
    lines.append(clamp_text(out, 4000))

    lines.append("\n$ git diff --stat")
    rc, diff_stat = run_cmd(["git", "diff", "--stat"], cwd=repo)
    lines.append(clamp_text(diff_stat, 8000))

    if include_diff:
        rc, diff = run_cmd(["git", "diff"], cwd=repo)
        if len(diff) > MAX_DIFF_CHARS:
            lines.append("\n$ git diff (too large -> omitted; use /gitdiff for targeted diff or review specific files)")
        else:
            lines.append("\n$ git diff")
            lines.append(diff)

    out_all = "\n".join(lines).strip()
    return clamp_text(out_all, MAX_ATTACH_CHARS)


def make_bedrock_runtime_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    cfg = Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=120)
    return session.client("bedrock-runtime", config=cfg)


def make_bedrock_control_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    cfg = Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=10, read_timeout=60)
    return session.client("bedrock", config=cfg)


def resolve_model_alias(model_id: str) -> str:
    key = (model_id or "").strip().lower()
    return MODEL_ALIASES.get(key, model_id)


def is_git_repo(path: str) -> bool:
    if not path:
        return False
    git_dir = os.path.join(path, ".git")
    if os.path.isdir(git_dir):
        return True
    # worktree など .git がファイルの場合もある
    return os.path.isfile(git_dir)


def find_repo_from_cwd(cwd: str) -> Optional[str]:
    """
    いまいる場所から上に向かって .git を探す。
    """
    cur = os.path.abspath(cwd)
    while True:
        if is_git_repo(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def resolve_repo_arg(repo_arg: Optional[str]) -> Optional[str]:
    """
    --repo 省略: cwd から git repo 自動検出（見つからなければ cwd を採用）
    --repo <name>: ~/workspace/<name> などから探索
    --repo <path>: そのまま（相対もOK）
    """
    if not repo_arg:
        found = find_repo_from_cwd(os.getcwd())
        return found or os.getcwd()

    # まずパスとして解釈
    cand = os.path.expanduser(repo_arg)
    if not os.path.isabs(cand):
        cand = os.path.abspath(cand)

    if os.path.exists(cand):
        # ディレクトリが git repo なら採用、そうでなくても cwd としては使えるので採用
        if os.path.isdir(cand):
            return cand
        # ファイル指定なら親をrepo候補にする
        if os.path.isfile(cand):
            return os.path.dirname(cand)

    # 名前指定として探索（よく使うワークスペース配下）
    name = repo_arg.strip().rstrip("/").split("/")[-1]
    roots = [
        os.path.expanduser("~/workspace"),
        os.path.expanduser("~/work"),
        os.path.expanduser("~/repos"),
    ]
    for r in roots:
        p = os.path.join(r, name)
        if os.path.isdir(p):
            return p

    # 最後に cwd（git検出があればそれ、なければcwd）
    return find_repo_from_cwd(os.getcwd()) or os.getcwd()


def resolve_model_id_if_needed(profile: str, region: str, model_id: str) -> str:
    # inference profile id っぽい
    if model_id.startswith(("apac.", "us.", "eu.")):
        return model_id
    # inference profile arn
    if INFERENCE_PROFILE_ARN_RE.match(model_id):
        return model_id
    # foundation model id じゃなさそうならそのまま
    if not FOUNDATION_ID_RE.match(model_id):
        return model_id

    ctl = make_bedrock_control_client(profile, region)
    summaries: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {}
        if token:
            kwargs["nextToken"] = token
        resp = ctl.list_inference_profiles(**kwargs)
        summaries.extend(resp.get("inferenceProfileSummaries", []) or [])
        token = resp.get("nextToken")
        if not token:
            break

    needle = f"/{model_id}"
    matches: List[Tuple[str, str]] = []
    for s in summaries:
        models = s.get("models")
        blob = json.dumps(models, ensure_ascii=False) if models is not None else ""
        if needle in blob:
            ip_id = s.get("inferenceProfileId") or ""
            ip_arn = s.get("inferenceProfileArn") or ""
            name = s.get("inferenceProfileName") or ""
            if ip_id:
                matches.append((name, ip_id))
            elif ip_arn:
                matches.append((name, ip_arn))

    if not matches:
        raise ValueError(f"inference profile not found for foundation model: {model_id}")

    chosen_name, chosen_id = matches[0]
    eprint(f"ℹ️  model '{model_id}' は on-demand 非対応のため、推論プロファイルへ解決しました: {chosen_name} -> {chosen_id}")
    return chosen_id


def converse(
    client,
    model_id: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    req: Dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": int(max_output_tokens),
            "temperature": float(temperature),
            "topP": float(top_p),
        },
    }
    if system_prompt.strip():
        req["system"] = [{"text": system_prompt}]
    return client.converse(**req)


def print_usage_hint() -> None:
    eprint(
        "\n操作:\n"
        "  /help                      ヘルプ\n"
        "  /exit                      終了\n"
        "  /reset                     履歴リセット\n"
        "  /save <path>               履歴保存先を変更（JSONL）\n"
        "  /load <path>               履歴を読み込む（JSONL）\n"
        "  /file <path>               ファイルを読み込み、次の発話に添付\n"
        "  /repo                      リポジトリ状態（status/log/diff）を次の発話に添付\n"
        "  /gitdiff                   git diff を次の発話に添付（大きい場合はstatへ縮退）\n"
        "  /cmd <shell>               コマンド実行結果を表示＆次の発話に添付（例: /cmd pytest -q）\n"
        "  /review <path>             指定ファイルを添付し、最小差分レビュー指示を自動付与\n"
        "  /sys <text>                システムプロンプトを変更\n"
        "  /model <modelId|alias>      モデルID切替（例: amazon.nova-pro-v1:0 / nova）\n"
        "  /max <n>                   max_output_tokens を変更\n"
        "  /temp <x>                  temperature を変更\n"
        "  /topp <x>                  top_p を変更\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bedrock Converse chat client (SSO-friendly)")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--save", default=None)
    parser.add_argument("--load", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--repo", default=None, help="repo path or repo name (default: auto-detect from cwd)")
    parser.add_argument("--print-raw", action="store_true")
    args = parser.parse_args()

    cfg = ChatConfig(
        profile=args.profile,
        region=args.region,
        model_id=resolve_model_alias(args.model),
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system or "",
        save_path=args.save,
        load_path=args.load,
        print_raw=args.print_raw,
        repo=resolve_repo_arg(args.repo),
    )

    history: List[Dict[str, Any]] = []
    pending_attach: List[str] = []
    pending_prefix_prompt: Optional[str] = None

    if cfg.load_path:
        loaded = load_jsonl(cfg.load_path)
        history.extend([m for m in loaded if m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str)])
        eprint(f"Loaded {len(history)} messages from {cfg.load_path}")

    try:
        cfg.model_id = resolve_model_id_if_needed(cfg.profile, cfg.region, cfg.model_id)
        runtime = make_bedrock_runtime_client(cfg.profile, cfg.region)
    except Exception as ex:
        eprint("❌ 初期化に失敗:", ex)
        eprint("  - aws sso login --profile <profile> を実行したか確認してください。")
        return 2

    def add_attach(text: str) -> None:
        nonlocal pending_attach
        if not text.strip():
            return
        joined = "\n\n".join(pending_attach + [text])
        if len(joined) > MAX_ATTACH_CHARS:
            remaining = MAX_ATTACH_CHARS - len("\n\n".join(pending_attach)) - 2
            pending_attach.append(clamp_text(text, max(0, remaining)))
        else:
            pending_attach.append(text)

    def build_user_payload(user_text: str) -> str:
        nonlocal pending_attach, pending_prefix_prompt
        payload = user_text.strip()
        if pending_prefix_prompt:
            payload = pending_prefix_prompt.strip() + "\n\n" + payload
            pending_prefix_prompt = None

        if pending_attach:
            payload = payload + "\n\n---\n[添付情報]\n" + "\n\n".join(pending_attach) + "\n---\n"
            pending_attach = []
        return payload

    def do_call(user_text: str) -> None:
        nonlocal history, cfg

        user_payload = build_user_payload(user_text)
        if not user_payload.strip():
            return

        history.append({"ts": now_iso(), "role": "user", "text": user_payload})
        if cfg.save_path:
            jsonl_append(cfg.save_path, {"ts": now_iso(), "role": "user", "text": user_payload})

        messages = to_bedrock_messages(history)
        t0 = time.time()
        try:
            resp = converse(
                client=runtime,
                model_id=cfg.model_id,
                system_prompt=cfg.system_prompt,
                messages=messages,
                max_output_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        except ClientError as ce:
            eprint("\n❌ AWS ClientError:", ce)
            return
        except BotoCoreError as be:
            eprint("\n❌ BotoCoreError:", be)
            return
        except Exception as ex:
            eprint("\n❌ 予期せぬエラー:", ex)
            return

        dt = int((time.time() - t0) * 1000)
        assistant_text, metrics = extract_assistant_text(resp)

        print("\n" + assistant_text + "\n")
        eprint(f"[model={cfg.model_id} maxOut={cfg.max_output_tokens} temp={cfg.temperature} topP={cfg.top_p} dt={dt}ms]")
        eprint(f"[usage] in={metrics.get('inputTokens')} out={metrics.get('outputTokens')} total={metrics.get('totalTokens')} stop={metrics.get('stopReason')}")

        history.append({"ts": now_iso(), "role": "assistant", "text": assistant_text, "metrics": metrics})
        if cfg.save_path:
            jsonl_append(cfg.save_path, {"ts": now_iso(), "role": "assistant", "text": assistant_text, "metrics": metrics})

    # one-shot mode
    if args.prompt is not None:
        if args.file:
            try:
                add_attach(f"[file:{args.file}]\n" + read_text_file(args.file))
            except Exception as ex:
                eprint("❌ ファイル読み込み失敗:", ex)
                return 2
        if cfg.repo:
            add_attach(repo_snapshot(cfg.repo, include_diff=True))
        do_call(args.prompt)
        return 0

    print("Bedrock chat client ready.")
    print(f"  profile={cfg.profile} region={cfg.region} model={cfg.model_id}")
    print("  tips: --model nova で Nova Pro を短縮指定できます（内部で推論プロファイルに解決されます）")
    print(f"  max_output_tokens={cfg.max_output_tokens} temperature={cfg.temperature} top_p={cfg.top_p}")
    if cfg.save_path:
        print(f"  save={cfg.save_path}")
    if cfg.load_path:
        print(f"  load={cfg.load_path}")
    if cfg.repo:
        print(f"  repo={cfg.repo}")
    print_usage_hint()

    while True:
        try:
            user_in = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        if not user_in:
            continue

        # よくある事故防止（/cmd のスラッシュ忘れ）
        if user_in.startswith("cmd "):
            print("❌ '/cmd' のスラッシュが抜けています。例: /cmd pytest -q")
            continue

        if user_in.startswith("/"):
            parts = user_in.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print("bye")
                return 0

            if cmd == "/help":
                print_usage_hint()
                continue

            if cmd == "/reset":
                history = []
                pending_attach = []
                print("✅ 履歴/添付をリセットしました。")
                continue

            if cmd == "/save":
                if not arg:
                    print("❌ /save <path>")
                    continue
                cfg.save_path = arg
                print(f"✅ save path = {cfg.save_path}")
                continue

            if cmd == "/load":
                if not arg:
                    print("❌ /load <path>")
                    continue
                loaded = load_jsonl(arg)
                history = [m for m in loaded if m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str)]
                cfg.load_path = arg
                print(f"✅ load {len(history)} messages from {arg}")
                continue

            if cmd == "/file":
                if not arg:
                    print("❌ /file <path>")
                    continue
                try:
                    add_attach(f"[file:{arg}]\n" + read_text_file(arg))
                    print(f"✅ 次の発話にファイル内容を添付します: {arg}")
                except Exception as ex:
                    print(f"❌ ファイル読み込み失敗: {ex}")
                continue

            if cmd == "/repo":
                if not cfg.repo:
                    print("❌ repo が解決できませんでした。起動場所を repo にするか、--repo <path|name> を指定してください。")
                    continue
                add_attach(repo_snapshot(cfg.repo, include_diff=True))
                print("✅ 次の発話にrepoスナップショットを添付します。")
                continue

            if cmd == "/gitdiff":
                if not cfg.repo:
                    print("❌ repo が解決できませんでした。--repo <path|name> を指定してください。")
                    continue
                rc, diff = run_cmd(["git", "diff"], cwd=cfg.repo)
                if len(diff) > MAX_DIFF_CHARS:
                    rc2, stat = run_cmd(["git", "diff", "--stat"], cwd=cfg.repo)
                    add_attach("[git diff was too large -> using stat]\n" + stat)
                else:
                    add_attach("[git diff]\n" + diff)
                print("✅ 次の発話にgit diffを添付します。")
                continue

            if cmd == "/cmd":
                if not arg:
                    print("❌ /cmd <command>")
                    continue
                cwd = cfg.repo or os.getcwd()
                rc, out = run_cmd(arg.split(), cwd=cwd)

                # まず “見える” ように表示（短縮）
                shown = clamp_text(out, CMD_SHOW_CHARS)
                print(f"\n[cmd:{arg} cwd={cwd} exit={rc}]\n{shown}\n")

                # 解析のために添付もする（必要なら長め）
                add_attach(f"[cmd:{arg} cwd={cwd} exit={rc}]\n" + clamp_text(out, MAX_CMD_OUTPUT_CHARS))
                print("✅ 次の発話にコマンド結果を添付します。")
                continue

            if cmd == "/review":
                if not arg:
                    print("❌ /review <path>")
                    continue
                try:
                    add_attach(f"[file:{arg}]\n" + read_text_file(arg))
                    pending_prefix_prompt = "次のコードをレビューし、改善案は最小変更のdiff形式で提示してください。"
                    print("✅ 次の発話はレビュー指示（diff優先）になります。続けて要望を書いてください。")
                except Exception as ex:
                    print(f"❌ ファイル読み込み失敗: {ex}")
                continue

            if cmd == "/sys":
                cfg.system_prompt = arg
                print("✅ system prompt updated.")
                continue

            if cmd == "/model":
                if not arg:
                    print("❌ /model <modelId|alias>")
                    continue
                try:
                    cfg.model_id = resolve_model_id_if_needed(cfg.profile, cfg.region, resolve_model_alias(arg))
                    print(f"✅ model = {cfg.model_id}")
                except Exception as ex:
                    print(f"❌ model 解決に失敗: {ex}")
                continue

            if cmd == "/max":
                try:
                    cfg.max_output_tokens = int(arg)
                    print(f"✅ max_output_tokens = {cfg.max_output_tokens}")
                except Exception:
                    print("❌ /max <int>")
                continue

            if cmd == "/temp":
                try:
                    cfg.temperature = float(arg)
                    print(f"✅ temperature = {cfg.temperature}")
                except Exception:
                    print("❌ /temp <float>")
                continue

            if cmd == "/topp":
                try:
                    cfg.top_p = float(arg)
                    print(f"✅ top_p = {cfg.top_p}")
                except Exception:
                    print("❌ /topp <float>")
                continue

            print("❌ unknown command. /help をどうぞ。")
            continue

        do_call(user_in)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


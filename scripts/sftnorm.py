"""Shared normalisation helpers for the SmallThinker coding-agent SFT corpus.

The SmallThinker control tags are assembled from string fragments at import
time.  Writing them as bare literals inside a source file is unsafe: some
tooling strips angle-bracket tags from file payloads, which silently corrupts
the dataset dialect (the tag simply vanishes and every downstream regex stops
matching).  Always build them by concatenation.

Canonical dialect produced by this module
-----------------------------------------
assistant content :  [think block] [prose] [tool call block] ...
think block       :  THINK_OPEN + "\\n" + reasoning + "\\n" + THINK_CLOSE
tool call block   :  TOOL_CALL_OPEN + "\\n" + json + "\\n" + TOOL_CALL_CLOSE
tool result       :  {"role": "tool", "content": ...}  (chat template renders
                     it as a <tool_response> block)

The tool-call JSON is produced with ``json.dumps(..., ensure_ascii=False)`` so
that it is byte-identical to what the model's chat template emits for a
structured ``tool_calls`` entry (Jinja ``tojson`` keeps insertion order and
uses the default ``", "`` / ``": "`` separators).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Control tags (built by concatenation on purpose - see module docstring).
# --------------------------------------------------------------------------
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
TOOL_CALL_OPEN = "<" + "tool_call>"
TOOL_CALL_CLOSE = "</" + "tool_call>"
TOOL_RESPONSE_OPEN = "<" + "tool_response>"
TOOL_RESPONSE_CLOSE = "</" + "tool_response>"
TOOL_RESULT_CLOSE = "</" + "tool_result>"
TOOL_NAME_OPEN = "<" + "tool_name>"
TOOL_CODE_OPEN = "<" + "tool_code>"
TOOL_CODE_CLOSE = "</" + "tool_code>"

DEFAULT_SYSTEM = (
    "You are a precise software engineering assistant. Understand the task, "
    "inspect relevant context, use available tools when they materially help, "
    "interpret tool results, make minimal correct changes, and verify important "
    "claims. Do not invent tool results. If no tool is needed, answer directly."
)

# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def tool_call_text(name: str, arguments: Any) -> str:
    """Render one tool call exactly the way the chat template renders it."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    payload = json.dumps({"name": str(name), "arguments": arguments}, ensure_ascii=False)
    return TOOL_CALL_OPEN + "\n" + payload + "\n" + TOOL_CALL_CLOSE


def think_block(reasoning: str) -> str:
    """Wrap a reasoning trace into the canonical inline think block."""
    r = (reasoning or "").strip()
    return THINK_OPEN + "\n" + r + "\n" + THINK_CLOSE


def assistant_text(
    reasoning: Optional[str] = None,
    content: Optional[str] = None,
    tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Build the canonical assistant turn content.

    Order is fixed: think block, then prose, then tool-call blocks.  The
    content never starts with whitespace - that keeps ``tokens(prompt)`` a
    strict prefix of ``tokens(prompt + target)`` at the generation boundary.
    """
    parts: List[str] = []
    if reasoning and reasoning.strip():
        parts.append(think_block(reasoning))
    body = (content or "").strip()
    if body:
        parts.append(body)
    for call in tool_calls or []:
        parts.append(tool_call_text(call["name"], call.get("arguments")))
    out = "\n".join(parts)
    return out.strip("\n")


def sanitize_assistant_content(text: str) -> str:
    """Strip leading/trailing whitespace so the generation boundary is stable."""
    return (text or "").strip()


# --------------------------------------------------------------------------
# Tool schema helpers
# --------------------------------------------------------------------------


def normalize_tool(tool: Any) -> Optional[Dict[str, Any]]:
    """Normalise any supported tool-schema dialect to OpenAI function form."""
    if isinstance(tool, str):
        try:
            tool = json.loads(tool)
        except Exception:
            return None
    if not isinstance(tool, dict):
        return None

    fn = tool.get("function")
    if not isinstance(fn, dict):
        fn = tool
    name = fn.get("name")
    if not name:
        return None
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = fn.get("input_schema")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    params = {k: v for k, v in params.items() if k != "$schema"}
    desc = fn.get("description") or ""
    return {
        "type": "function",
        "function": {
            "name": str(name),
            "description": str(desc),
            "parameters": params,
        },
    }


def normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    """Normalise a list of tool schemas, dropping unusable entries."""
    if not tools:
        return []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            return []
    if isinstance(tools, dict):
        tools = [tools]
    out: List[Dict[str, Any]] = []
    seen = set()
    for tool in tools:
        norm = normalize_tool(tool)
        if norm is None:
            continue
        key = norm["function"]["name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


# --------------------------------------------------------------------------
# Hermes flat-text parser
# --------------------------------------------------------------------------

_HERMES_BLOCK_RE = re.compile(
    "(?P<think>" + re.escape(THINK_OPEN) + ".*?" + re.escape(THINK_CLOSE) + ")"
    "|(?P<call>" + re.escape(TOOL_CALL_OPEN) + ".*?" + re.escape(TOOL_CALL_CLOSE) + ")"
    "|(?P<resp>" + re.escape(TOOL_RESPONSE_OPEN) + ".*?(?:"
    + re.escape(TOOL_RESPONSE_CLOSE) + "|" + re.escape(TOOL_RESULT_CLOSE) + "))"
    "|(?P<toolname>" + re.escape(TOOL_NAME_OPEN) + ".*?(?:"
    + re.escape(TOOL_RESPONSE_CLOSE) + "|" + re.escape(TOOL_RESULT_CLOSE) + "))"
    "|(?P<toolcode>" + re.escape(TOOL_CODE_OPEN) + ".*?" + re.escape(TOOL_CODE_CLOSE) + ")",
    re.S,
)


_PARAM_RE = re.compile(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.S)
_BRACKET_ARGS_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*\n?\s*arguments\s*\[(.*)\]\s*$", re.S)
_BRACKET_PAIR_RE = re.compile(r'([\w.-]+)\s*=\s*"((?:[^"\\]|\\.)*)"')
_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


def _valid_tool_name(name: Any) -> Optional[str]:
    """Return the name when it looks like a tool name, otherwise ``None``.

    Some source rows carry malformed payloads whose ``name`` field swallowed
    the rest of the message.  A tool name is always a short identifier.
    """
    if not isinstance(name, str):
        return None
    name = name.strip()
    if not _TOOL_NAME_RE.fullmatch(name):
        return None
    return name


def parse_call_payload(body: str) -> Optional[Dict[str, Any]]:
    """Parse the body of a tool-call block.

    Handles three dialects found in the wild:
      * plain JSON  ``{"name": "f", "arguments": {...}}``
      * XML-ish     ``<parameter name="cmd">...</parameter>``
      * shorthand   ``f\\narguments[key="value"]``
    """
    s = (body or "").strip()
    if not s:
        return None

    # --- dialect 2: <parameter name="...">...</parameter> -----------------
    if "<parameter" in s:
        params = {m.group(1): m.group(2) for m in _PARAM_RE.finditer(s)}
        if params:
            name = params.pop("name", None) or params.pop("tool", None)
            args = dict(params)
            for key in ("arguments", "args", "parameters"):
                if key in args and isinstance(args[key], str):
                    try:
                        args.update(json.loads(args[key]))
                    except Exception:
                        pass
                    args.pop(key, None)
            if _valid_tool_name(name):
                return {"name": str(name), "arguments": args}

    # --- dialect 3: name + arguments[key="value"] ------------------------
    m = _BRACKET_ARGS_RE.match(s)
    if m and _valid_tool_name(m.group(1)):
        args = {k: v.replace('\\"', '"') for k, v in _BRACKET_PAIR_RE.findall(m.group(2))}
        return {"name": m.group(1), "arguments": args}

    # --- dialect 1: JSON --------------------------------------------------
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function_name")
    if not name and isinstance(obj.get("function"), dict):
        name = obj["function"].get("name")
    name = _valid_tool_name(name)
    if not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("args")
    if args is None:
        args = {}
    return {"name": str(name), "arguments": args}


_THINK_BLOCK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.S)
_CALL_BLOCK_RE = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.S)


def canonicalize_assistant(content: str) -> Optional[str]:
    """Repair an assistant turn and put it back into the canonical dialect.

    * tool-call blocks are extracted first, so a call that the source nested
      inside a think block is hoisted out into its canonical position
    * think blocks are re-wrapped so the tags are always balanced
    * tool-call blocks that do not parse are dropped (never kept as prose)
    * stray control tags left over from malformed source text are removed
    * the final order is always: think block, prose, tool-call blocks

    Returns ``None`` when nothing usable is left.
    """
    if content is None:
        return None
    calls: List[Dict[str, Any]] = []

    def _take_call(m: "re.Match[str]") -> str:
        parsed = parse_call_payload(m.group(1))
        if parsed is not None:
            calls.append(parsed)
        return " "

    # calls first: the source sometimes emits <tool_call> inside <think>
    rest = _CALL_BLOCK_RE.sub(_take_call, content)

    thinks: List[str] = []

    def _take_think(m: "re.Match[str]") -> str:
        body = m.group(1)
        # a nested think tag inside the body would unbalance the counts
        body = body.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
        body = body.strip()
        if body:
            thinks.append(body)
        return " "

    rest = _THINK_BLOCK_RE.sub(_take_think, rest)
    for tag in (THINK_OPEN, THINK_CLOSE, TOOL_CALL_OPEN, TOOL_CALL_CLOSE):
        rest = rest.replace(tag, " ")
    prose = re.sub(r"\n{3,}", "\n\n", rest).strip()
    if not thinks and not calls and not prose:
        return None
    out = assistant_text(reasoning="\n\n".join(thinks) if thinks else None,
                         content=prose or None,
                         tool_calls=calls)
    return out or None


def sanitize_conversation(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Canonicalise every assistant turn and drop orphaned tool results."""
    out: List[Dict[str, str]] = []
    last_had_call = False
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            content = canonicalize_assistant(message.get("content") or "")
            if content is None:
                last_had_call = False
                continue
            out.append({"role": "assistant", "content": content})
            last_had_call = TOOL_CALL_OPEN in content
        elif role == "tool":
            if not last_had_call:
                continue  # orphaned tool result
            out.append({"role": "tool", "content": message.get("content") or ""})
        else:
            out.append({"role": role, "content": message.get("content") or ""})
            if role != "system":
                last_had_call = False
    # a trailing tool result with no following assistant turn is harmless but
    # pointless - keep it, the template renders it fine
    return out


def parse_hermes_events(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split a Hermes flat trajectory into (leading text, events).

    Event kinds: ``think``, ``call``, ``resp``, ``text``.
    """
    events: List[Tuple[str, str]] = []
    pos = 0
    leading = ""
    first = True
    for m in _HERMES_BLOCK_RE.finditer(text):
        chunk = text[pos : m.start()]
        if chunk.strip():
            if first:
                leading = chunk
                first = False
            else:
                events.append(("text", chunk))
        kind = m.lastgroup or "text"
        body = m.group(kind)
        if kind == "think":
            body = body[len(THINK_OPEN) : -len(THINK_CLOSE)]
        elif kind == "call":
            body = body[len(TOOL_CALL_OPEN) : -len(TOOL_CALL_CLOSE)]
        elif kind == "resp":
            body = body[len(TOOL_RESPONSE_OPEN) :]
            for close in (TOOL_RESPONSE_CLOSE, TOOL_RESULT_CLOSE):
                if body.endswith(close):
                    body = body[: -len(close)]
                    break
        elif kind == "toolname":
            body = body[len(TOOL_NAME_OPEN) :]
            for close in (TOOL_RESPONSE_CLOSE, TOOL_RESULT_CLOSE):
                if body.endswith(close):
                    body = body[: -len(close)]
                    break
        elif kind == "toolcode":
            body = body[len(TOOL_CODE_OPEN) : -len(TOOL_CODE_CLOSE)]
        events.append((kind, body))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        events.append(("text", tail))
    return leading, events


def hermes_user_content(leading: str, task: str) -> str:
    """Recover the user turn from the leading text of a Hermes trajectory."""
    lead = (leading or "").strip()
    if lead.lower().startswith("user:"):
        lead = lead[5:].strip()
    else:
        lead = ""
    # strip a trailing markdown rule used as a separator
    lead = re.sub(r"\n-{3,}\n*$", "", lead).strip()
    if lead:
        return lead
    return (task or "").strip()


def hermes_events_to_messages(events: Sequence[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Turn parsed Hermes events into a canonical message list."""
    messages: List[Dict[str, str]] = []
    think_parts: List[str] = []
    prose_parts: List[str] = []
    calls: List[Dict[str, Any]] = []

    def _add_call(parsed: Optional[Dict[str, Any]], body: str) -> None:
        if parsed is None:
            if body.strip():
                prose_parts.append(body.strip())
            return
        parsed = dict(parsed)
        parsed["name"] = HERMES_TOOL_ALIASES.get(parsed["name"], parsed["name"]).strip()
        calls.append(parsed)

    def flush() -> None:
        nonlocal think_parts, prose_parts, calls
        if not (think_parts or prose_parts or calls):
            return
        content = assistant_text(
            reasoning="\n\n".join(think_parts) if think_parts else None,
            content="\n".join(prose_parts) if prose_parts else None,
            tool_calls=calls,
        )
        if content.strip():
            messages.append({"role": "assistant", "content": sanitize_assistant_content(content)})
        think_parts, prose_parts, calls = [], [], []

    for kind, body in events:
        if kind == "think":
            b = body.strip()
            if b:
                think_parts.append(b)
        elif kind == "call":
            _add_call(parse_call_payload(body), body)
        elif kind == "resp":
            flush()
            messages.append({"role": "tool", "content": body.strip()})
        elif kind == "toolname":
            _add_call(parse_call_payload(body), body)
        elif kind == "toolcode":
            _add_call(parse_call_payload(body), body)
        else:
            b = body.strip()
            if b:
                prose_parts.append(b)
    flush()
    return messages


# --------------------------------------------------------------------------
# Synthesised tool schemas (Hermes ships no schemas of its own)
# --------------------------------------------------------------------------

HERMES_TOOL_DESCRIPTIONS = {
    "terminal": "Run a shell command in the working directory and return its output.",
    "bash": "Run a shell command in the working directory and return its output.",
    "execute_code": "Execute a snippet of code and return its result.",
    "read_file": "Read the contents of a file from the working directory.",
    "write_file": "Write content to a file in the working directory.",
    "edit_file": "Apply an edit to a file in the working directory.",
    "search_files": "Search the working directory for files matching a pattern.",
    "list_files": "List the files and directories of the working directory.",
    "web_search": "Search the web and return the relevant results.",
    "memory": "Store or recall a persistent note between sessions.",
    "session_search": "Search previous sessions for discussions matching a query.",
    "skill_manage": "Create, list or update a reusable skill definition.",
    "skill_view": "Show the definition of a saved skill.",
    "skills_list": "List the skills available in this workspace.",
    "delegate_task": "Delegate a self-contained subtask to a sub-agent.",
    "cronjob": "Schedule, list or remove a recurring job.",
    "todo": "Record the current task plan as a list of todo items.",
    "clarify": "Ask the user a clarifying question.",
    "process": "Inspect or control a background process.",
    "patch": "Apply a unified diff to a file in the working directory.",
    "vision_analyze": "Analyse an image and describe what it shows.",
}

# The corpus is synthetic, so a handful of tool names are misspellings or
# synonyms of the real ones.  Fold them onto the canonical name so the
# synthesised schema stays small and consistent.
HERMES_TOOL_ALIASES = {
    "terminl": "terminal", "termial": "terminal", "termin": "terminal",
    "term": "terminal", "bash": "terminal", "shell": "terminal",
    "run_terminal": "terminal", "execute_command": "terminal",
    "install": "terminal", "expect": "terminal",
    " read_file": "read_file", "skim_file": "read_file",
    "search_file": "search_files",
    "exec": "execute_code", "exec_code": "execute_code", "execode": "execute_code",
    "execute_python": "execute_code", "execute_node": "execute_code",
    "file_write": "write_file",
    "files_list": "list_files", "list_directory": "list_files",
    "directory_list": "list_files",
    "to_do": "todo", "todo_write": "todo",
    "think": "todo", "task_status": "todo",
    "complete": "todo", "summary": "todo", "final_summary": "todo",
    "task_summary": "todo",
    "help": "list_files", "tool_name": "list_files",
    "ask_user": "clarify",
    # long-tail outliers observed in the synthetic traces (1-4 calls each)
    "terminate": "terminal", "terminals": "terminal", "execution": "terminal",
    "expect": "terminal", "daily_code_review": "terminal",
    "file": "write_file", "files": "read_file", "view_file": "read_file",
    "edit": "write_file", "whiteboard_create": "write_file",
    "executive_plan": "write_file", "express-api": "write_file",
    "file_search": "search_files", "searching codebase": "search_files",
    "task": "delegate_task", "review_task": "delegate_task",
    "tool": "web_search",
}

# Short descriptions for the argument names that actually matter.
HERMES_ARG_DESCRIPTIONS = {
    "terminal": {
        "command": "Shell command to execute.",
        "timeout": "Timeout for the command in milliseconds.",
        "workdir": "Working directory for the command.",
        "work_dir": "Working directory for the command.",
        "working_dir": "Working directory for the command.",
        "cwd": "Working directory for the command.",
        "description": "Short description of what the command does.",
        "background": "Run the command in the background.",
        "shell": "Shell used to run the command.",
        "blocking": "Block until the command finishes.",
    },
    "read_file": {
        "path": "Path of the file to read.",
        "file_path": "Path of the file to read.",
        "file": "Path of the file to read.",
        "offset": "First line to read.",
        "limit": "Maximum number of lines to read.",
        "recursive": "Search recursively.",
    },
    "search_files": {
        "pattern": "Glob or regular expression to match.",
        "path": "Directory to search in.",
        "recursive": "Search recursively.",
        "case_sensitive": "Match case sensitively.",
        "caseSensitive": "Match case sensitively.",
        "limit": "Maximum number of results to return.",
        "output_mode": "How results should be formatted.",
        "target_directory": "Directory to search in.",
    },
    "execute_code": {
        "code": "Source code to execute.",
        "language": "Programming language of the snippet.",
        "timeout": "Timeout in milliseconds.",
        "command": "Command to run instead of a snippet.",
        "cwd": "Working directory for the execution.",
    },
    "web_search": {
        "query": "Search query.",
        "num_results": "Number of results to return.",
        "max_results": "Maximum number of results to return.",
        "source": "Where to search.",
    },
    "memory": {
        "action": "Operation to perform: store, search, update or delete.",
        "query": "Search query.",
        "key": "Identifier of the memory entry.",
        "value": "Value to store.",
        "content": "Content of the memory entry.",
        "category": "Category of the memory entry.",
    },
    "session_search": {
        "query": "Search query.",
        "limit": "Maximum number of sessions to return.",
        "session_id": "Identifier of a specific session.",
    },
    "todo": {
        "content": "Description of the task.",
        "id": "Identifier of the task.",
        "status": "Status of the task.",
        "todos": "The full list of tasks.",
    },
    "write_file": {
        "path": "Path of the file to write.",
        "file_path": "Path of the file to write.",
        "content": "New file content.",
    },
    "skill_manage": {
        "name": "Name of the skill.",
        "action": "Operation to perform.",
        "content": "Definition of the skill.",
    },
    "delegate_task": {
        "task": "Description of the delegated task.",
        "context": "Context the sub-agent needs.",
    },
    "clarify": {
        "question": "Question to ask the user.",
    },
    "cronjob": {
        "action": "Operation to perform.",
        "schedule": "Cron schedule expression.",
        "command": "Command to run.",
    },
    "patch": {
        "path": "Path of the file to patch.",
        "diff": "Unified diff to apply.",
    },
    "process": {
        "id": "Identifier of the process.",
        "action": "Operation to perform.",
    },
}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str):
        return "string"
    return "string"


class ToolSpecCollector:
    """Collect tool-name / argument statistics so schemas can be synthesised.

    The corpus is synthetic and uses a very wide spread of argument spellings
    (102 distinct keys for ``terminal`` alone).  Only the most frequent keys
    are kept in the schema - the rare spellings stay in the training target
    but must not bloat the system prompt.
    """

    def __init__(self, max_args: int = 8) -> None:
        self.max_args = max_args
        self.arg_types: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.arg_presence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.calls: Counter = Counter()

    def observe(self, name: str, arguments: Any) -> None:
        name = HERMES_TOOL_ALIASES.get(name, name).strip()
        if not name:
            return
        self.calls[name] += 1
        if not isinstance(arguments, dict):
            return
        for key, value in arguments.items():
            self.arg_types[name][key][_json_type(value)] += 1
            self.arg_presence[name][key] += 1

    def build_specs(self) -> Dict[str, Dict[str, Any]]:
        specs: Dict[str, Dict[str, Any]] = {}
        for name, n_calls in self.calls.items():
            ranked = sorted(self.arg_presence.get(name, {}).items(),
                            key=lambda kv: (-kv[1], kv[0]))
            kept = ranked[: self.max_args]
            props: Dict[str, Any] = {}
            required: List[str] = []
            arg_desc = HERMES_ARG_DESCRIPTIONS.get(name, {})
            for key, seen in kept:
                type_counter = self.arg_types[name][key]
                chosen = type_counter.most_common(1)[0][0]
                entry: Dict[str, Any] = {"type": chosen}
                if key in arg_desc:
                    entry["description"] = arg_desc[key]
                props[key] = entry
                if seen >= 0.9 * n_calls:
                    required.append(key)
            spec = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": HERMES_TOOL_DESCRIPTIONS.get(
                        name, "Invoke the '%s' tool." % name),
                    "parameters": {"type": "object", "properties": props},
                },
            }
            if required:
                spec["function"]["parameters"]["required"] = sorted(required)
            specs[name] = spec
        return specs


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def dedup_key(prompt: str, target: str) -> str:
    """Stable dedup key over the prompt and the head of the target."""
    h = hashlib.sha1()
    h.update(prompt.encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(target[:256].encode("utf-8", "ignore"))
    return h.hexdigest()


def ensure_system(messages: List[Dict[str, str]], default: str = DEFAULT_SYSTEM) -> None:
    """Make sure the conversation starts with a non-empty system message."""
    if messages and messages[0].get("role") == "system":
        if not (messages[0].get("content") or "").strip():
            messages[0]["content"] = default
        return
    messages.insert(0, {"role": "system", "content": default})


# --------------------------------------------------------------------------
# Tool-schema compression
# --------------------------------------------------------------------------


def compress_tool(tool: Dict[str, Any], desc_chars: int = 140,
                  prop_desc_chars: int = 70) -> Dict[str, Any]:
    """Shrink a tool schema so the system prompt stays affordable.

    Keeps the function name, a shortened description, and the argument names /
    types / required flags.  Drops verbose JSON-schema noise.
    """
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    props_in = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    props: Dict[str, Any] = {}
    for key, spec in props_in.items():
        if not isinstance(spec, dict):
            props[key] = {"type": "string"}
            continue
        entry: Dict[str, Any] = {"type": spec.get("type", "string")}
        desc = spec.get("description")
        if desc:
            entry["description"] = str(desc)[:prop_desc_chars]
        props[key] = entry
    out_params: Dict[str, Any] = {"type": "object", "properties": props}
    required = params.get("required")
    if isinstance(required, list) and required:
        out_params["required"] = [r for r in required if r in props]
    return {
        "type": "function",
        "function": {
            "name": str(fn.get("name", "")),
            "description": str(fn.get("description") or "")[:desc_chars],
            "parameters": out_params,
        },
    }


def compress_tools(tools: Sequence[Dict[str, Any]], **kw: Any) -> List[Dict[str, Any]]:
    return [compress_tool(t, **kw) for t in tools]


# --------------------------------------------------------------------------
# Fast chat-template renderer
# --------------------------------------------------------------------------
#
# The SmallThinker chat template is fully known (see tokenizer_config.json), so
# a conversation can be rendered block-by-block.  Tokenising block-by-block is
# exact because every block boundary falls on a special token
# (<|im_start|> / <|im_end|>), and the tool-call / tool-response markers are
# themselves special or otherwise split identically in both paths.
#
# This turns the O(k^2) "render the whole prefix per turn" cost into O(n) and
# is what makes a 90k-turn corpus tractable to filter exactly.

IM_START = "<" + "|im_start|>"
IM_END = "<" + "|im_end|>"
GEN_PROMPT = IM_START + "assistant\n"
DEFAULT_SYSTEM_TEXT = "You are SmallThinker. You are a helpful assistant."

_TOOLS_HEADER = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_TOOLS_FOOTER = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    + TOOL_CALL_OPEN + TOOL_CALL_CLOSE + " XML tags:\n"
    + TOOL_CALL_OPEN + "\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    + TOOL_CALL_CLOSE + IM_END + "\n"
)


def _tojson(obj: Any) -> str:
    """Mirror of the Jinja ``tojson`` filter used by transformers chat templates."""
    return json.dumps(obj, ensure_ascii=False)


def render_system_block(messages: Sequence[Dict[str, Any]], tools: Optional[Sequence[Dict[str, Any]]]) -> str:
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
    else:
        content = DEFAULT_SYSTEM_TEXT
    if tools:
        parts = [IM_START + "system\n" + content + _TOOLS_HEADER]
        for tool in tools:
            parts.append("\n" + _tojson(tool))
        parts.append(_TOOLS_FOOTER)
        return "".join(parts)
    return IM_START + "system\n" + content + IM_END + "\n"


def render_message_block(message: Dict[str, Any]) -> str:
    role = message.get("role")
    content = message.get("content") or ""
    if role == "user":
        return IM_START + "user\n" + content + IM_END + "\n"
    if role == "system":
        return IM_START + "system\n" + content + IM_END + "\n"
    if role == "assistant":
        calls = message.get("tool_calls")
        if calls:
            out = IM_START + "assistant"
            if content:
                out += "\n" + content
            for call in calls:
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                out += ("\n" + TOOL_CALL_OPEN + "\n{\"name\": \"" + str(fn.get("name", ""))
                        + "\", \"arguments\": " + _tojson(fn.get("arguments", {})) + "}\n"
                        + TOOL_CALL_CLOSE)
            return out + IM_END + "\n"
        return IM_START + "assistant\n" + content + IM_END + "\n"
    raise ValueError("unsupported role: %r" % role)


def render_blocks(messages: Sequence[Dict[str, Any]], tools: Optional[Sequence[Dict[str, Any]]]) -> Tuple[str, List[str]]:
    """Return (system block, per-message blocks) exactly as the template emits them."""
    sys_block = render_system_block(messages, tools)
    blocks: List[str] = []
    idx = 1 if (messages and messages[0].get("role") == "system") else 0
    i = idx
    n = len(messages)
    while i < n:
        message = messages[i]
        if message.get("role") == "tool":
            run: List[Dict[str, Any]] = []
            while i < n and messages[i].get("role") == "tool":
                run.append(messages[i])
                i += 1
            out = IM_START + "user"
            for tool_message in run:
                out += ("\n" + TOOL_RESPONSE_OPEN + "\n" + (tool_message.get("content") or "")
                        + "\n" + TOOL_RESPONSE_CLOSE)
            blocks.append(out + IM_END + "\n")
        else:
            blocks.append(render_message_block(message))
            i += 1
    return sys_block, blocks


def render_conversation(messages: Sequence[Dict[str, Any]], tools: Optional[Sequence[Dict[str, Any]]]) -> str:
    sys_block, blocks = render_blocks(messages, tools)
    return sys_block + "".join(blocks)

"""A generator for shell command lines that knows what it built.

`hooks._segments` is a hand-written shell parser, and the four commits before
this file existed were four bugs in it -- each found by a person typing one more
string. The example-based tests in `tests/test_hooks.py` are the record of those
four strings. What they cannot do is find the fifth.

So this builds command lines from a grammar instead, and carries the answer
alongside the text: every node knows which command heads the shell would
*execute* in it, which makes `Node.heads` the oracle. No shell is invoked -- the
point is to compare the parser against the language, not against another parser
-- so the grammar is restricted to constructs whose semantics are not in dispute:

  * the operators `hooks._OPERATORS` already lists, plus the newline;
  * `( ... )` and `{ ...; }`, which run their contents;
  * `$( ... )` and backticks, which run their contents and are live inside
    double quotes;
  * single quotes, which suppress all of it;
  * redirections, which move a stream and run nothing -- so the head of a
    command carrying one is the head it had without it.

Everything the module docstring names as a known bypass -- `bash -c`,
`ssh host "cmd"`, aliases, `eval`, environment indirection -- is *not* generated.
Those are architectural, the docstring says so, and a property suite that
generated them would be asserting a claim the code has never made.

**Heredocs are generated, and they are generated `commands()` cannot reach
them.** A body lives on the lines *after* the command, so an opener composed as
the left half of `a && b` would put `&& b` on the terminator line and the
construct would stop being the one the oracle described. `heredoc_commands` is
therefore a top-level strategy of its own rather than a node in the recursion --
which costs nesting and buys an oracle that is exactly right.

They were excluded before, on the reasoning that a body needs lines the grammar
had no way to emit. Two bugs lived in that exclusion, and the second was
introduced *by the fix for the first*: stripping heredoc bodies before matching
(right for `cat`) also stripped them for `bash`, where the body is the program
arriving on stdin -- and because the strip ran before segmentation it removed
`ssh` from `bash <<EOF / ssh box / EOF` before the head rule ever ran. A
generator emitting opener-plus-body-plus-terminator as one unit would have
caught that the day it landed, which is the whole argument for this file.

**Interpreter payloads stay out**, and that one is a decision. `bash -c "..."`
is already excluded above as a known bypass; generating it would need the oracle
to say what the payload runs, which is the parse the module declines to do. The
heredoc node sidesteps that only because it never has to: for `bash <<EOF` the
body is *shell*, so the oracle is the body's own heads.

The oracle is allowed to under-claim and never to over-claim. `$(echo ls)` at
command position really does run `ls` in a shell, and `heads` says only `echo`;
that direction costs a missed counterexample, which is a weaker suite. The other
direction would cost a false failure, which is a suite people learn to ignore.

The vocabulary keeps the two directions independent: `SAFE_HEADS` and
`SAFE_ARGS` contain nothing that trips the three regex rules in `evaluate_bash`
(`rm -rf`, a download piped into a shell, a credential read), so a denial from a
generated command is always attributable to the head that earned it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypothesis import strategies as st

import hooks

#: Command heads the deny list refuses. Taken from the module rather than
#: copied, so a new entry is covered by every property here the day it lands.
DENIED_HEADS = sorted(hooks._DENIED_COMMANDS)

#: Heads that must never be denied. Deliberately includes `python`, because
#: `python -m tools.gpu submit` is the *suggested* route and denying the
#: suggestion would be the worst possible failure.
SAFE_HEADS = ["echo", "true", "ls", "cat", "python", "git", "grep", "wc"]

#: Argument words. No `-rf`, no URL, no `keyring`: see the module docstring.
SAFE_ARGS = ["x", "-n", "file.txt", "--json", "notes/", "1", "a.py"]


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """A fragment of shell, and what it runs.

    `has_squote` and `has_backtick` exist because two of the wrappers below are
    not re-entrant in the shell itself. `'a'b'c'` is three fragments rather than
    a nested quote, and a backtick cannot contain a backtick without escaping --
    so nesting either inside itself would produce a string whose real meaning is
    not the one this node claims, and the oracle would be lying rather than the
    parser wrong. Both are filtered at the point of composition.
    """

    text: str
    #: Heads this fragment executes. Empty for a fragment that is only data.
    heads: tuple[str, ...]
    has_squote: bool = False
    has_backtick: bool = False

    def runs_denied(self) -> bool:
        return any(head in hooks._DENIED_COMMANDS for head in self.heads)


def _simple(
    head: str,
    spelling: str,
    args: list[str],
    assignment: str | None,
    redirection: tuple[str, str] | None,
) -> Node:
    """One command: an optional `VAR=value` prefix, a head, some arguments.

    `spelling` is how the head is written -- bare, path-qualified, or with the
    Windows extension `_head` strips. All four resolve to the same command, and
    a deny list that only knows the bare form is one `/usr/bin/` away from
    silent.

    `redirection` attaches at the leaf and nowhere else, because that is where
    the shell puts it: a redirection is part of a *simple* command, so
    `>out if true; then ls; fi` is not a thing bash runs and generating one
    would have the oracle asserting about a syntax error. Composed into the
    operators, groups and substitutions by the recursion above, one leaf
    redirection reaches every construct in the grammar anyway.
    """
    written = {"bare": head, "path": f"/usr/bin/{head}", "dot": f"./{head}",
               "exe": f"{head}.exe"}[spelling]
    parts: list[str] = []
    if assignment:
        parts.append(assignment)
    if redirection and redirection[0] == "lead":
        parts.append(redirection[1])
    parts.append(written)
    parts.extend(args)
    if redirection and redirection[0] == "trail":
        parts.append(redirection[1])
    return Node(" ".join(parts).strip(), (head,))


_HEAD_SPELLINGS = st.sampled_from(["bare", "path", "dot", "exe"])
_ASSIGNMENTS = st.sampled_from([None, "FOO=1", "LC_ALL=C"])

#: Redirections, in both spacings, both positions, and with and without a file
#: descriptor. **The head is unchanged by every one of them**, which is the whole
#: assertion: a redirection moves a stream, it does not run a program, so
#: `2>/dev/null ssh box` runs exactly what `ssh box` runs.
#:
#: This node is why the file exists in the form it does. The grammar generated
#: every operator, `( )`, `{ ; }`, `$( )`, backticks and quotes -- the entire
#: shell vocabulary *except* redirection -- and `SAFE_ARGS` contains no token
#: with a `>` or a `<` in it. So the two fail-open bugs that lived in exactly
#: that gap were invisible to a green suite by construction, and green CI was
#: never evidence against them. A property suite that cannot express a bug does
#: not fail to find it; it fails to be asked.
#:
#: Semantics are not in dispute for any of these, which is the bar the module
#: docstring sets. `<<` is the one left out: a heredoc needs a body on the
#: following lines, and a generator that emits the opener without one produces a
#: string no shell would accept rather than a command with a known answer.
#:
#: The targets are ordinary filenames on purpose -- nothing here trips the three
#: whole-string rules, so a denial from a generated command is still always
#: attributable to the head that earned it.
_REDIRECTIONS = st.sampled_from([
    None,
    ("lead", ">out.txt"), ("lead", "> out.txt"),
    ("lead", ">>log.txt"), ("lead", ">> log.txt"),
    ("lead", "<in.txt"), ("lead", "< in.txt"),
    ("lead", "2>/dev/null"), ("lead", "2> /dev/null"),
    ("lead", "2>&1"), ("lead", "1>&2"), ("lead", ">&2"),
    ("lead", "&>out.txt"), ("lead", "&> out.txt"), ("lead", "&>>log.txt"),
    ("lead", ">|out.txt"), ("lead", ">| out.txt"),
    ("lead", "<<< word"),
    ("trail", "> out.txt"), ("trail", ">>log.txt"), ("trail", "< in.txt"),
    ("trail", "2>/dev/null"), ("trail", "2>&1"), ("trail", "&>out.txt"),
    ("trail", ">|out.txt"), ("trail", "<<< word"),
])


def simple_commands(heads: list[str]) -> st.SearchStrategy[Node]:
    return st.builds(
        _simple,
        st.sampled_from(heads),
        _HEAD_SPELLINGS,
        st.lists(st.sampled_from(SAFE_ARGS), max_size=3),
        _ASSIGNMENTS,
        _REDIRECTIONS,
    )


#: Operators that separate two commands, both of which run. `&` backgrounds the
#: left one and a newline ends it; in neither case does the right one stop
#: running, which is the only thing the deny list has to agree with.
_BINARY = ["&&", "||", ";", "|", "&", "\n"]


def _merge(text: str, *parts: Node, extra: tuple[str, ...] = ()) -> Node:
    return Node(
        text,
        tuple(h for p in parts for h in p.heads) + extra,
        any(p.has_squote for p in parts),
        any(p.has_backtick for p in parts),
    )


def _binary(op: str, left: Node, right: Node) -> Node:
    sep = op if op == "\n" else f" {op} "
    return _merge(f"{left.text}{sep}{right.text}", left, right)


def _subshell(inner: Node) -> Node:
    """`( cmd )` -- a subshell. A different process, the same language."""
    return _merge(f"( {inner.text} )", inner)


def _brace_group(inner: Node) -> Node:
    """`{ cmd; }` -- a group. The trailing `;` is what the shell requires."""
    return _merge(f"{{ {inner.text}; }}", inner)


def _conditional(inner: Node) -> Node:
    """`if true; then cmd; fi`. The body runs, and `then` is not a command."""
    return _merge(f"if true; then {inner.text}; fi", inner)


def _loop(inner: Node) -> Node:
    """`for i in 1 2; do cmd; done`. Same shape, and `do` is not a command."""
    return _merge(f"for i in 1 2; do {inner.text}; done", inner)


#: Reserved words that take a command and are valid in any command position, so
#: the grammar can nest them freely. `!` and `time` are *not* here: bash requires
#: both at the start of a pipeline, so `a | ! b` is a syntax error rather than a
#: command, and generating one would have the property demand a denial for a
#: string no shell would run. They are covered by
#: `test_a_reserved_word_does_not_hide_the_command_after_it` instead.
_COMPOUND = (_subshell, _brace_group, _conditional, _loop)


def _substitution(inner: Node, form: str) -> Node:
    """Command substitution. Live unquoted and inside double quotes alike.

    The quoted forms are the ones worth generating most: they are where a
    reader's intuition says "this is a string" and the shell says "this is a
    command", and three of the four fixed bugs lived in exactly that gap.
    """
    text = {
        "bare": f"$({inner.text})",
        "double": f'"$({inner.text})"',
        "backtick": f'"`{inner.text}`"',
        "argument": f'echo "$({inner.text})"',
    }[form]
    node = _merge(text, inner, extra=("echo",) if form == "argument" else ())
    if form == "backtick":
        node = Node(node.text, node.heads, node.has_squote, True)
    return node


def _single_quoted(inner: Node) -> Node:
    """`echo '$(cmd)'` -- data, not a command. The shell runs nothing in here.

    Generated so the no-false-denial property has something with the shape of an
    execution and none of the substance. Wrapped in `echo` because a bare quoted
    string is not a command line anybody writes.
    """
    return Node(f"echo '$({inner.text})'", ("echo",), True, inner.has_backtick)


def commands(heads: list[str], *, max_leaves: int = 4) -> st.SearchStrategy[Node]:
    """Command lines built from `heads`, with the executed set carried along.

    Wrapped in `deferred` for its repr and nothing else: `recursive` renders as
    its whole expansion, and a failure report that opens with 54 kB of strategy
    before it gets to the counterexample is a report nobody reads to the end.
    """
    return st.deferred(
        lambda: st.recursive(
            simple_commands(heads),
            lambda children: st.one_of(
                st.builds(_binary, st.sampled_from(_BINARY), children, children),
                st.one_of(*[st.builds(wrap, children) for wrap in _COMPOUND]),
                st.builds(
                    _substitution,
                    children,
                    st.sampled_from(["bare", "double", "argument"]),
                ),
                st.builds(
                    _substitution,
                    children.filter(lambda n: not n.has_backtick),
                    st.just("backtick"),
                ),
                st.builds(_single_quoted, children.filter(lambda n: not n.has_squote)),
            ),
            max_leaves=max_leaves,
        )
    )


# ---------------------------------------------------------------------------
# heredocs, which are their own top-level shape
# ---------------------------------------------------------------------------
#: Commands that take a heredoc and treat the body as **data**: they write it
#: somewhere. The body of one of these runs nothing, whatever it says.
DATA_CARRIERS = ["cat", "tee"]

#: Commands that take a heredoc and treat the body as **the program**, arriving
#: on stdin instead of in a `-c` argument. `bash -s` is the same thing spelled
#: with the flag that says so.
CODE_CARRIERS = ["bash", "sh", "zsh", "python", "python3", "bash -s"]

#: Deliberately not `EOF`. The oracle depends on the terminator appearing on a
#: line of its own and nowhere else, and a delimiter that no generated token can
#: equal is what makes that true by construction rather than by luck.
_DELIMITER = "GRADEOF"


def _heredoc(inner: Node, carrier: str, form: str, redirect: str) -> Node:
    """One heredoc: an opener, a body, and a terminator on its own line.

    **The carrier decides what the body is**, and that is the entire property.
    `cat <<EOF / ssh box / EOF` writes a file and runs `cat`. `bash <<EOF /
    ssh box / EOF` runs `bash` *and* `ssh`. The two differ by one word and the
    deny list has to tell them apart -- it did not, and the fix that made `cat`
    right made `bash` a bypass of every entry in `_DENIED_COMMANDS`.

    `form` covers `<<` against `<<-` and a bare delimiter against a quoted one.
    Quoting the delimiter suppresses every expansion in the body, which makes it
    the strongest "this is data" signal the shell has -- and, crucially, does
    *not* make the body data when the carrier is an interpreter. That pair is
    the trap this node exists to generate.
    """
    opener = {
        "plain": f"<<{_DELIMITER}",
        "quoted": f"<<'{_DELIMITER}'",
        "dash": f"<<-{_DELIMITER}",
        "spaced": f"<< {_DELIMITER}",
    }[form]
    text = f"{carrier} {opener}{redirect}\n{inner.text}\n{_DELIMITER}"
    runs_body = hooks._is_interpreter(hooks._head(carrier))
    return Node(text, (hooks._head(carrier),) + (inner.heads if runs_body else ()))


def heredoc_commands(
    heads: list[str], *, carriers: list[str] | None = None
) -> st.SearchStrategy[Node]:
    """Heredocs whose body is one simple command.

    Top-level only -- see the module docstring. A body of one command rather
    than an arbitrary node keeps the terminator unreachable from the body's own
    text, which is what lets the oracle be exact instead of conservative.
    """
    return st.builds(
        _heredoc,
        simple_commands(heads),
        st.sampled_from(carriers if carriers is not None else DATA_CARRIERS + CODE_CARRIERS),
        st.sampled_from(["plain", "quoted", "dash", "spaced"]),
        st.sampled_from(["", " > out.txt", " >> log.txt"]),
    )


def describe(node: Node) -> dict[str, Any]:
    """What to print when a property fails, so the report is the whole story."""
    segments = hooks._segments(node.text)
    return {
        "command": node.text,
        "executes": sorted(set(node.heads)),
        "segments": segments,
        "parsed heads": [hooks._head(s) for s in segments],
    }

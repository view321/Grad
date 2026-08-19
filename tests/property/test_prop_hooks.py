"""Properties of the Bash deny list.

Two of them, and they pull in opposite directions on purpose.

`test_a_denied_head_anywhere_is_denied` is the safety direction: if the shell
would run `ssh`, the hook has to say so, however the string was folded. This is
the one that finds bypasses.

`test_safe_commands_are_never_denied` is the usability direction: a hook that
denies `grep -n "ssh" hooks.py` teaches people to turn it off, and the person
most likely to type that command is the one auditing this file. A deny list is
allowed to over-split -- that only ever costs a false *segment* -- but it is not
allowed to over-deny.

Both take their vocabulary from `hooks` itself, so adding an entry to
`_DENIED_COMMANDS` extends the suite rather than leaving a hole in it.
"""

from __future__ import annotations

import shlex

import pytest
import shellgrammar as sg
from hypothesis import assume, given, note
from hypothesis import strategies as st

import hooks


# ---------------------------------------------------------------------------
# the safety direction
# ---------------------------------------------------------------------------
@given(sg.commands(sg.DENIED_HEADS + sg.SAFE_HEADS))
def test_a_denied_head_anywhere_is_denied(node: sg.Node) -> None:
    """Whatever the shell runs, the hook sees.

    The generator folds a command into operators, subshells, brace groups,
    substitutions and quotes, and remembers which heads survive as commands.
    Every one of those foldings is something a person types; none of them is
    the `bash -c` class the module docstring rules out of scope.
    """
    assume(node.runs_denied())
    note(sg.describe(node))
    assert hooks.evaluate_bash(node.text) is not None


@given(sg.commands(sg.SAFE_HEADS))
def test_safe_commands_are_never_denied(node: sg.Node) -> None:
    """Nothing built from harmless heads is refused.

    The vocabulary in `shellgrammar` is chosen to keep this honest: no `rm -rf`,
    no download piped into a shell, no credential read, so the only thing that
    could deny here is the head rule misreading a segment.
    """
    note(sg.describe(node))
    assert hooks.evaluate_bash(node.text) is None


@given(st.sampled_from(sg.DENIED_HEADS))
def test_a_reserved_word_does_not_hide_the_command_after_it(denied: str) -> None:
    """`if`, `do`, `then`, `!` and `time` introduce a command; they are not one.

    Left unhandled, each of these was a one-word bypass: `for h in a b; do ssh
    $h; done` split cleanly on the `;` and then reported the head of
    ` do ssh $h` as `do`.

    `!` and `time` are here rather than in the grammar because bash requires
    both at the head of a pipeline, so they cannot be nested arbitrarily without
    generating strings no shell would accept.
    """
    for command in (
        f"! {denied} box",
        f"time {denied} box",
        f"if {denied} box; then true; fi",
        f"if true; then {denied} box; fi",
        f"if false; then true; else {denied} box; fi",
        f"while {denied} box; do true; done",
        f"until true; do {denied} box; done",
        f"for h in a b; do {denied} $h; done",
    ):
        note(command)
        assert hooks.evaluate_bash(command) is not None, command


@given(st.sampled_from(sg.DENIED_HEADS), st.sampled_from(sg.SAFE_HEADS))
def test_a_denied_word_as_an_argument_is_not_a_command(denied: str, safe: str) -> None:
    """`grep -n "ssh" hooks.py` is a read, not a remote execution.

    The false denial this is about was a real one: the blind split left a tail
    of `b" file` for `grep -n "a\\|b" file`, and a pattern containing a denied
    word was refused as though it had been typed as a command.
    """
    for command in (
        f'{safe} -n "{denied}" hooks.py',
        f"{safe} '{denied} box' notes.md",
        f'{safe} "a|{denied}" file.txt',
        f"{safe} --pattern={denied}",
    ):
        note(command)
        assert hooks.evaluate_bash(command) is None


# ---------------------------------------------------------------------------
# heredocs, where the carrier decides whether the body is a program
# ---------------------------------------------------------------------------
@given(sg.heredoc_commands(sg.DENIED_HEADS + sg.SAFE_HEADS))
def test_a_heredoc_body_runs_exactly_when_its_carrier_is_an_interpreter(
    node: sg.Node,
) -> None:
    """`bash <<EOF` runs its body; `cat <<EOF` writes it.

    One word apart, opposite answers, and the deny list has to tell them apart.
    It did not: the fix that stopped `cat <<'EOF'` being denied for quoting the
    deny list's own vocabulary removed the body *before segmentation*, so
    `bash <<EOF / ssh box / EOF` lost its `ssh` before the head rule ran. Not a
    hole in one regex -- a bypass of every entry in `_DENIED_COMMANDS`.

    Generated rather than listed because the previous two rounds of this were
    also found by hand, one construct at a time, and the point of this file is
    to stop paying that way."""
    assume(node.runs_denied())
    note(sg.describe(node))
    assert hooks.evaluate_bash(node.text) is not None


@given(sg.heredoc_commands(sg.DENIED_HEADS, carriers=sg.DATA_CARRIERS))
def test_a_data_heredoc_body_is_never_a_command(node: sg.Node) -> None:
    """The direction the whole heredoc fix exists for.

    A body handed to `cat` or `tee` is a file being written, however loudly it
    quotes the deny list -- and writing a document *about* this file is how the
    false denial was found in the first place. `runs_denied` is False for every
    node here by construction, which is the oracle agreeing that these are data.
    """
    assert not node.runs_denied()
    note(sg.describe(node))
    assert hooks.evaluate_bash(node.text) is None


@given(sg.heredoc_commands(sg.SAFE_HEADS))
def test_a_safe_heredoc_is_never_denied(node: sg.Node) -> None:
    """Neither carrier may deny a body that runs nothing denied."""
    note(sg.describe(node))
    assert hooks.evaluate_bash(node.text) is None


# ---------------------------------------------------------------------------
# the splitter itself
# ---------------------------------------------------------------------------
@given(sg.commands(sg.DENIED_HEADS + sg.SAFE_HEADS))
def test_segments_invent_nothing(node: sg.Node) -> None:
    """Every character of every segment came from the command.

    A splitter that emits text the caller never typed can deny a command that
    was never written, and the head rule downstream would have no way to tell.
    Checked as a multiset over characters rather than as substrings, because
    `$(` is consumed rather than kept and the segment boundaries genuinely do
    not line up with the input.
    """
    segments = hooks._segments(node.text)
    note(sg.describe(node))
    source = list(node.text)
    for segment in segments:
        for char in segment:
            assert char in source, f"segment {segment!r} invented {char!r}"
            source.remove(char)


#: Text made mostly of the characters `_segments` gives meaning to. Plain
#: `st.text()` is the wrong alphabet for a parser: it draws from the whole of
#: Unicode, so a backslash or a `$(` turns up rarely enough that the branches
#: that matter go unvisited. Mutation testing is what showed this -- eight
#: mutants inside the backslash-escape branch survived, including two that make
#: `_segments` raise IndexError on a command ending in a backslash, which no
#: generated example had ever produced.
shellish = st.text(
    alphabet=st.sampled_from(list("\\'\"$(){}[]|&;`<>\n\r\t abcdefgHIJ.-/=*")),
    max_size=60,
)


@given(st.one_of(shellish, st.text(max_size=60)))
def test_the_splitter_terminates_on_anything(text: str) -> None:
    """Arbitrary text in, a list of strings out, no exception.

    `evaluate_bash` runs on whatever the model emitted, which is not
    necessarily a shell command at all -- and a hook that raises is a hook that
    fails open, because the SDK has nothing to do with the exception but let the
    call through. A command ending in a lone backslash is the shortest way to
    get there, and it is one keystroke from something a person types.
    """
    segments = hooks._segments(text)
    assert all(isinstance(s, str) for s in segments)
    assert all(s.strip() for s in segments)
    verdict = hooks.evaluate_bash(text)
    assert verdict is None or isinstance(verdict, hooks.Denial)


@given(st.sampled_from(sg.DENIED_HEADS), st.sampled_from(["", " ", "\t"]))
def test_a_command_ending_in_a_backslash_is_still_a_command(
    denied: str, trailing: str
) -> None:
    """The last character is the one with no character after it.

    `_segments` consumes a backslash *and the character it escapes*, which needs
    a bounds check, and the bounds check had no test: mutating `i + 1 < n` to
    `i - 1 < n` or `i + 1 <= n` survived the whole suite. Both raise IndexError
    here, and `evaluate_bash` raising is `pre_tool_use` raising, which the SDK
    resolves by letting the call through -- so the failure mode of a missing
    bounds check in a deny list is that it stops denying.
    """
    for command in (f"{denied} box{trailing}\\", f"ls{trailing}\\", "\\"):
        note(command)
        hooks._segments(command)  # must not raise
        hooks.evaluate_bash(command)
    assert hooks.evaluate_bash(f"{denied} box \\") is not None


@given(st.sampled_from(sg.DENIED_HEADS))
def test_a_quoted_windows_path_resolves_to_its_program(denied: str) -> None:
    """`"C:\\tools\\ssh.exe" box` is `ssh`, and Windows is the first-class target.

    `_head` strips a backslash-separated directory and a `.exe` suffix for
    exactly this, and nothing tested either: `rsplit("\\\\", 1)` mutated to
    `split("\\\\", 1)` survived the suite, which means no test had ever passed it
    a path with two backslashes in it.

    Quoted, because that is the spelling that runs. Bare `C:\\tools\\ssh.exe`
    is not a Windows path to the shell the hook protects -- bash eats the
    backslashes and tries to execute `C:toolsssh.exe`, which is nothing -- so
    `_head` declining to see a program there agrees with what would happen.
    """
    for command in (
        f'"C:\\tools\\{denied}.exe" box',
        f"'C:\\Program Files\\bin\\{denied}.exe' box",
        f'"D:\\a\\b\\{denied}" box',
    ):
        note(command)
        assert hooks.evaluate_bash(command) is not None, command


@given(st.sampled_from(sg.SAFE_HEADS), st.sampled_from(["|", ";", "&"]))
def test_an_escaped_operator_is_not_an_operator(safe: str, operator: str) -> None:
    """`echo a\\;b` is one command, because the backslash took the `;` with it.

    The false-denial direction of the escape branch. Over-splitting here would
    make the tail a segment of its own, and a tail that happens to start with a
    denied word would be refused as though it had been typed as a command.

    One character each, because a backslash escapes exactly one: `a\\&&b` is
    `a&` followed by a real `&`, and splitting it is right. The first version of
    this test listed `&&` and was wrong about the shell rather than about the
    code.
    """
    command = f"{safe} a\\{operator}b"
    note(command)
    assert hooks._segments(command) == [command]
    assert hooks.evaluate_bash(command) is None


@given(st.sampled_from(sg.DENIED_HEADS), st.sampled_from(["|", ";", "&&"]))
def test_an_escape_does_not_hide_the_next_command(denied: str, operator: str) -> None:
    """...and the direction that would cost more if it were wrong.

    A backslash escapes one character. `echo \\x {op} ssh box` still has a real
    operator in it, and the escape branch must not run past it.
    """
    command = f"echo \\x {operator} {denied} box"
    note(command)
    assert hooks.evaluate_bash(command) is not None


@given(st.one_of(shellish, st.text(max_size=60)))
def test_the_head_of_a_segment_is_one_of_its_tokens(text: str) -> None:
    """`_head` names a token that is there, lowercased and stripped of its path.

    The bug this guards against is the one fixed two commits ago in reverse: a
    closing delimiter left on a token made `ssh)"` out of `ssh`. Anything that
    produces a head no tokeniser would agree with is a head the deny list is
    matching by accident.
    """
    for segment in hooks._segments(text):
        head = hooks._head(segment)
        if not head:
            continue
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        stripped = {
            t.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower().removesuffix(".exe")
            for t in tokens
        }
        assert head in stripped, f"{head!r} is not a token of {segment!r}"


# ---------------------------------------------------------------------------
# cost-bearing detection
# ---------------------------------------------------------------------------
@given(
    st.sampled_from(hooks._COST_BEARING),
    st.sampled_from(["", "cd notes && ", "true; ", "FOO=1 "]),
    st.sampled_from(["", " --json", " --spec s.toml --expect e1"]),
)
def test_a_cost_bearing_command_is_recognised_however_it_is_written(
    pair: tuple[str, str], prefix: str, suffix: str
) -> None:
    """The budget hook's half of §15 finds the module and verb it is looking for.

    This is the token loop's *only* enforcement point -- no submitter sees a
    turn -- so a spelling it fails to recognise is a ceiling that silently is
    not one.
    """
    module, verb = pair
    command = f"{prefix}python -m {module} {verb}{suffix}"
    note(command)
    assert hooks.cost_bearing_command(command) == (module, verb)


@given(st.sampled_from(hooks._COST_BEARING))
def test_a_cost_bearing_verb_inside_single_quotes_is_not_a_submission(
    pair: tuple[str, str],
) -> None:
    """Writing *about* a submit is not submitting.

    `echo 'python -m tools.jobs submit'` spends nothing, and a budget hook that
    denies it is denying the agent's ability to explain what it was going to do.
    """
    module, verb = pair
    assert hooks.cost_bearing_command(f"echo 'python -m {module} {verb}'") is None


# ---------------------------------------------------------------------------
# the regex rules
# ---------------------------------------------------------------------------
@given(
    st.lists(st.sampled_from(["-r", "-f", "-R", "--recursive", "--force", "-v"]),
             min_size=2, max_size=4, unique=True),
    st.sampled_from(["notes", "data/", "/tmp/x", "."]),
)
def test_recursive_force_delete_is_denied_in_any_flag_order(
    flags: list[str], target: str
) -> None:
    """`rm -r -f`, `rm -f -r`, `rm --force --recursive`: one rule, every order.

    The combined-only pattern this replaced let the separated form straight
    through, which is the spelling a model writes when it is being careful.
    """
    recursive = {"-r", "-R", "--recursive"} & set(flags)
    force = {"-f", "--force"} & set(flags)
    assume(recursive and force)
    command = " ".join(["rm", *flags, target])
    note(command)
    assert hooks.evaluate_bash(command) is not None


@pytest.mark.parametrize("empty", ["", "   ", "\n", "\t\n "])
def test_nothing_is_not_denied(empty: str) -> None:
    assert hooks.evaluate_bash(empty) is None

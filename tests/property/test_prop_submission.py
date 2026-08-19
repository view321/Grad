"""Properties of the submission hash and the overrides that feed it.

The hash is the identity of an experiment. `core/experiments.py` stores the
resolved document in an archive and re-derives the hash from it later, possibly
on another machine and long after the spec file was edited; `core/gates.py`
looks a preflight record up by it before any money is spent. So two things have
to be true, and neither is checkable from a single example:

  * the same document always hashes the same, whatever order its keys were
    built in and whatever route it took through JSON;
  * different documents hash differently, or the preflight gate can be satisfied
    by a dry run of something else.

`parse_override` and `_set_dotted` are here because they are how a document
acquires the values that get hashed -- `--set lr=3e-4` has to survive as the
float `0.0003` rather than the string `"3e-4"`, since the two are different
experiments to the hash and the same experiment to a reader.
"""

from __future__ import annotations

import json

from hypothesis import assume, given, note
from hypothesis import strategies as st

from core.errors import ConfigError
from core.submission import _set_dotted, hash_resolved, parse_override

#: TOML-shaped values: what a resolved spec document actually contains after
#: `tomllib` has read it and the overrides have been folded in.
values = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=30),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=8), children, max_size=4),
    ),
    max_leaves=8,
)
documents = st.dictionaries(st.text(min_size=1, max_size=10), values, max_size=6)

#: Dotted paths with no empty component. `a..b` is not a path anybody writes and
#: `parse_override` makes no promise about it.
key_paths = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=6),
    min_size=1,
    max_size=4,
).map(".".join)


# ---------------------------------------------------------------------------
# the hash
# ---------------------------------------------------------------------------
@given(documents)
def test_the_hash_is_a_function_of_the_document_and_nothing_else(doc: dict) -> None:
    """Same content, same hash, however the dict was assembled.

    Python preserves insertion order and `sort_keys=True` is what makes that
    irrelevant. Without it, a document built by folding overrides in a different
    order would be a different experiment, and `preflight` would refuse a dry
    run it had already done.
    """
    shuffled = dict(reversed(list(doc.items())))
    assert hash_resolved(doc) == hash_resolved(shuffled)
    assert hash_resolved(doc) == hash_resolved(doc)


@given(documents)
def test_the_hash_survives_the_archive(doc: dict) -> None:
    """A document written to disk and read back is the same experiment.

    This is the round trip `core/experiments.py` actually performs, and the one
    that would make the verifier report a mismatch for every archived run if the
    canonical form and the stored form disagreed.
    """
    archived = json.loads(json.dumps(doc, default=str))
    note({"original": doc, "archived": archived})
    assert hash_resolved(archived) == hash_resolved(doc)


@given(documents, documents)
def test_different_documents_get_different_hashes(a: dict, b: dict) -> None:
    """The direction the preflight gate depends on.

    A collision means a dry run of one pipeline satisfies the gate for another,
    which is the one failure this hash exists to prevent. Truncated to
    `HASH_LEN`, so this is a statement about the truncation being long enough as
    much as about SHA-256.
    """
    assume(a != b)
    assert hash_resolved(a) != hash_resolved(b)


@given(documents, st.integers(min_value=4, max_value=64))
def test_a_shorter_hash_is_a_prefix_of_the_longer_one(doc: dict, length: int) -> None:
    """Truncation is truncation, not a different digest.

    A record written with one length and looked up with another must still
    match on the shared prefix, or the archive and the ledger disagree about
    which run is which.
    """
    full = hash_resolved(doc, length=None)
    assert full.startswith(hash_resolved(doc, length=length))
    assert len(hash_resolved(doc, length=length)) == length


# ---------------------------------------------------------------------------
# overrides
# ---------------------------------------------------------------------------
@given(key_paths, values)
def test_a_json_value_survives_the_round_trip_through_an_override(
    key: str, value: object
) -> None:
    """`--set lr=3e-4` is the float, not the string.

    Types survive into the hash by going through JSON, so this is the property
    that makes `--set epochs=10` and `--set epochs="10"` two different
    experiments -- which they are.
    """
    text = f"{key}={json.dumps(value)}"
    parsed_key, parsed_value = parse_override(text)
    note({"text": text, "key": parsed_key, "value": parsed_value})
    assert parsed_key == key
    assert parsed_value == value


@given(key_paths, st.text(max_size=20))
def test_a_value_that_is_not_json_stays_a_string(key: str, raw: str) -> None:
    """An unquoted word is a string rather than an error.

    `--set model=resnet50` is the common case and is not valid JSON; refusing it
    would make every override need quoting that the shell would then eat.
    """
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        assume(False)
    assert parse_override(f"{key}={raw}") == (key, raw)


@given(key_paths, st.text(max_size=10))
def test_only_the_first_equals_separates_key_from_value(key: str, tail: str) -> None:
    """`--set cmd=a=b` sets `cmd` to `a=b`.

    Splitting on every `=` would silently truncate any value containing one,
    and a command line or a query string is exactly the kind of value that
    does.
    """
    parsed_key, parsed_value = parse_override(f"{key}=a={tail}")
    assert parsed_key == key
    assert parsed_value == f"a={tail}"


@given(st.text(max_size=20).filter(lambda s: "=" not in s))
def test_an_override_with_no_value_is_refused_with_the_fix(text: str) -> None:
    """A refusal names the thing the caller skipped, literally enough to paste."""
    try:
        parse_override(text)
    except ConfigError as exc:
        assert "--set" in str(getattr(exc, "fix", "") or "")
    else:
        raise AssertionError(f"{text!r} was accepted as an override")


@given(key_paths, values)
def test_a_dotted_path_is_readable_back_from_where_it_was_written(
    key: str, value: object
) -> None:
    """`_set_dotted` and the obvious walk agree.

    The nesting is what the hash sees, so a path that writes to the wrong depth
    produces a document that differs from the one the user asked for by a level
    nobody looks at.
    """
    target: dict = {}
    _set_dotted(target, key, value)
    node = target
    for part in key.split(".")[:-1]:
        node = node[part]
    assert node[key.split(".")[-1]] == value


@given(key_paths, values, values)
def test_the_last_override_of_a_path_wins(key: str, first: object, second: object) -> None:
    """Overrides are applied in order and the later one is the answer."""
    target: dict = {}
    _set_dotted(target, key, first)
    _set_dotted(target, key, second)
    node = target
    for part in key.split(".")[:-1]:
        node = node[part]
    assert node[key.split(".")[-1]] == second


@given(st.lists(key_paths, min_size=2, max_size=4, unique=True), values)
def test_writing_one_path_leaves_the_others_alone(paths: list[str], value: object) -> None:
    """Setting `train.lr` must not drop `train.epochs`.

    A `_set_dotted` that replaced an existing dict rather than descending into
    it would pass every single-key test and silently delete a sibling on the
    second `--set`.
    """
    # Sorted so a path is never written after something that would make it a
    # non-dict: `--set a=1 --set a.b=2` is a genuine conflict and the last one
    # legitimately wins.
    paths = sorted(paths)
    assume(not any(b.startswith(a + ".") for a in paths for b in paths if a != b))
    target: dict = {}
    for path in paths:
        _set_dotted(target, path, value)
    note(target)
    for path in paths:
        node = target
        for part in path.split(".")[:-1]:
            node = node[part]
        assert path.split(".")[-1] in node

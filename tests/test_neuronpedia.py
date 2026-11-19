from autointerp.neuronpedia import decode_tokens, source_groups, window_text


def test_plain_spaces_and_newlines():
    # real tokens from gpt2-small/6-res-jb feature 100
    toks = [" cost", "Ċ", "Ċ", "Research", " is", " showing", " the", " effect", " of"]
    assert window_text(toks) == " cost\n\nResearch is showing the effect of"


def test_multibyte_character_in_one_token():
    assert decode_tokens([" Tomblin", "âĢĻ", "s"]) == [" Tomblin", "’", "s"]
    assert decode_tokens([" âĢĶ"]) == [" —"]


def test_multibyte_character_split_across_tokens():
    # "’" = e2 80 99 split as "âĢ" + "Ļ", as in the feature 100 windows
    out = decode_tokens([" California", "âĢ", "Ļ", "s"])
    assert out == [" California", "’", "", "s"]
    assert len(out) == 4


def test_special_token_kept_and_dangling_bytes_replaced():
    assert decode_tokens(["<|endoftext|>", " a"]) == ["<|endoftext|>", " a"]
    assert decode_tokens(["âĢ"]) == ["�"]


def test_source_groups_links_shifted_windows_transitively():
    a = list("abcdefghij")
    b = list("cdefghijkl")  # a shifted by 2
    c = list("ijklmnopqr")  # shares a run with b only
    d = list("zyxwvutsrq")  # unrelated
    assert source_groups([a, b, c, d], k=3) == [0, 0, 0, 1]

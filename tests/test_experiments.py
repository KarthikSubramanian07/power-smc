from experiments import answers_match, extract_boxed, normalize_answer


def test_extract_boxed_basic():
    assert extract_boxed("the answer is \\boxed{42}.") == "42"


def test_extract_boxed_nested_braces():
    assert extract_boxed("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_takes_last():
    assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"


def test_extract_boxed_missing():
    assert extract_boxed("no box here") is None


def test_normalize_answer_strips_formatting():
    assert normalize_answer("\\left( 3 \\right)") == "(3)"
    assert normalize_answer("\\dfrac{1}{2}") == "\\frac{1}{2}"


def test_answers_match_exact_and_numeric():
    assert answers_match("42", "42")
    assert answers_match("42.0", "42")
    assert answers_match("\\frac{1}{2}", "\\dfrac{1}{2}")
    assert not answers_match("41", "42")
    assert not answers_match(None, "42")

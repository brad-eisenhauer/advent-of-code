import pytest

import advent_of_code.util.grammar as G


@pytest.fixture
def context_free_grammar() -> G.ContextFreeGrammar:
    return G.ContextFreeGrammar(
        rules=[
            G.Rule("e", ("H",)),
            G.Rule("e", ("O",)),
            G.Rule("H", ("H", "O")),
            G.Rule("H", ("O", "H")),
            G.Rule("O", ("H", "H")),
        ],
        start_symbol="e",
    )


class TestContextFreeGrammar:
    def test_grammar_is_cnf(self, context_free_grammar):
        assert not context_free_grammar.is_cnf()

    def test_grammar_to_cnf(self, context_free_grammar: G.ContextFreeGrammar):
        expected = G.ContextFreeGrammar(
            rules=[
                G.Rule("A", ("H",), 0),
                G.Rule("B", ("O",), 0),
                G.Rule("e", ("H",)),
                G.Rule("e", ("O",)),
                G.Rule("e", ("A", "B"), 2),
                G.Rule("e", ("B", "A"), 2),
                G.Rule("e", ("A", "A"), 2),
                G.Rule("A", ("A", "B")),
                G.Rule("A", ("B", "A")),
                G.Rule("B", ("A", "A")),
            ],
            start_symbol="e",
        )
        result = context_free_grammar.to_cnf({"H", "O"}, iter("ABCDEFGIJKLMNPQRSTUVWXYZ"))
        assert result.is_cnf()
        assert result == expected


class TestCYK:
    grammar = G.ContextFreeGrammar(
        rules=[
            G.Rule("S", ("NP", "VP")),
            G.Rule("VP", ("VP", "PP")),
            G.Rule("VP", ("V", "NP")),
            G.Rule("VP", ("eats",)),
            G.Rule("PP", ("P", "NP")),
            G.Rule("NP", ("Det", "N")),
            G.Rule("NP", ("she",)),
            G.Rule("V", ("eats",)),
            G.Rule("P", ("with",)),
            G.Rule("N", ("fish",)),
            G.Rule("N", ("fork",)),
            G.Rule("Det", ("a",)),
        ],
        start_symbol="S",
    )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (["she", "eats", "a", "fish", "with", "a", "fork"], True),
            (["she", "eats"], True),
            (["she", "eats", "a", "fish"], True),
            (["she", "eats", "with", "a", "fork"], True),
            (["she", "with", "eats", "a", "fork", "fish"], False),
            (["a", "eats", "fish", "fork", "she", "with"], False),
            (["eats", "a", "fish"], False),
        ],
    )
    def test_cyk(self, text, expected):
        result = G.cyk(text, self.grammar)
        assert bool(result) is expected

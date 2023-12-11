import pytest

from advent_of_code.util import cfg


@pytest.fixture()
def context_free_grammar() -> cfg.ContextFreeGrammar:
    return cfg.ContextFreeGrammar(
        rules = [
            cfg.Rule("e", ["H"]),
            cfg.Rule("e", ["O"]),
            cfg.Rule("H", ["H", "O"]),
            cfg.Rule("H", ["O", "H"]),
            cfg.Rule("O", ["H", "H"]),
        ],
        start_symbol="e",
    )


class TestContextFreeGrammar:
    def test_grammar_is_cnf(self, context_free_grammar):
        assert not context_free_grammar.is_cnf()

    def test_grammar_to_cnf(self, context_free_grammar: cfg.ContextFreeGrammar):

        expected = cfg.ContextFreeGrammar(
            rules=[
                cfg.Rule("e", ["H", "O"]),
                cfg.Rule("e", ["O", "H"]),
                cfg.Rule("e", ["H", "H"]),
                cfg.Rule("H", ["H", "O"]),
                cfg.Rule("H", ["O", "H"]),
                cfg.Rule("O", ["H", "H"]),
            ],
            start_symbol="e",
        )
        result = context_free_grammar.to_cnf({"H", "O"}, iter("ABCDEFGIJKLMNPQRSTUVWXYZ"))
        assert result.is_cnf()


class TestCYK:
    def test_cyk(self):
        grammar = cfg.ContextFreeGrammar(
            rules=[
                cfg.Rule("S", ["VP", "NP"]),
                cfg.Rule("VP", ["VP", "PP"]),
                cfg.Rule("VP", ["V", "NP"]),
                cfg.Rule("VP", ["eats"]),
                cfg.Rule("PP", ["P", "NP"]),
                cfg.Rule("NP", ["Det", "N"]),
                cfg.Rule("NP", ["she"]),
                cfg.Rule("V", ["eats"]),
                cfg.Rule("P", ["with"]),
                cfg.Rule("N", ["fish"]),
                cfg.Rule("N", ["fork"]),
                cfg.Rule("Det", ["a"]),
            ],
            start_symbol="S",
        )
        text = ["she", "eats", "fish", "with", "a", "fork"]
        result = cfg.cyk(text, grammar)
        assert result is True

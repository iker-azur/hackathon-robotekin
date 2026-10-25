"""Spoken orders -> piece routing, on readings shaped like the intent model's replies."""

import importlib.util
import sys
import unittest
from pathlib import Path


def load():
    path = Path(__file__).resolve().parent.parent / "demo_orders.py"
    spec = importlib.util.spec_from_file_location("demo_orders_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


orders = load()


def rule(feature="any", quality="any", count="all", box="big", p=0.99, source="said"):
    return {"feature": feature, "quality": quality, "count": count, "box": box, "p_feature": p, "p_quality": p,
            "p_count": p, "p_box": p, "box_source": source}


def reading(*rules, status="ok", intent="move", p_intent=0.99):
    return {"status": status, "intent": intent, "confidence": p_intent, "rules": list(rules), "note": "",
            "probabilities": {"intent": {intent: p_intent}}}


class VocabularyTests(unittest.TestCase):
    def test_green_is_the_big_box_and_small_is_the_white_box(self):
        said = orders.cell_vocabulary
        self.assertEqual(said("Move all the pieces to the green box"), "Move all the pieces to the big box")
        self.assertEqual(said("Move the bad pieces to the small box"), "Move the bad pieces to the white box")
        self.assertEqual(said("Mueve las piezas malas a la caja verde"), "Mueve las piezas malas a la caja grande")
        self.assertEqual(said("Pon las piezas en la caja pequeña"), "Pon las piezas en la caja blanca")
        self.assertEqual(said("Eraman pieza txarrak kutxa berdera"), "Eraman pieza txarrak kutxa handira")
        self.assertEqual(said("Eraman pieza onak kutxa txikira"), "Eraman pieza onak kutxa zurira")

    def test_other_words_and_case_are_left_alone(self):
        self.assertEqual(orders.cell_vocabulary("Mueve las piezas con perno a la caja blanca"),
                         "Mueve las piezas con perno a la caja blanca")
        self.assertEqual(orders.cell_vocabulary("Move the pieces to the Green box"), "Move the pieces to the big box")


class OrderBookTests(unittest.TestCase):
    def test_good_pieces_to_white_sets_only_the_pin_class(self):
        book = orders.OrderBook()
        outcome = book.apply(reading(rule(quality="good", box="white")))
        self.assertEqual(outcome.kind, "set")
        self.assertEqual(book.destination_for(orders.PIN), "white")
        self.assertIsNone(book.destination_for(orders.NO_PIN))

    def test_two_rules_in_one_sentence(self):
        book = orders.OrderBook()
        book.apply(reading(rule(quality="good", box="white"), rule(quality="bad", box="big"), intent="both"))
        self.assertEqual((book.destination_for(orders.PIN), book.destination_for(orders.NO_PIN)), ("white", "green"))

    def test_all_pieces_to_the_big_box_covers_both_classes(self):
        book = orders.OrderBook()
        book.apply(reading(rule(box="big")))
        self.assertEqual({book.destination_for(orders.PIN), book.destination_for(orders.NO_PIN)}, {"green"})

    def test_a_later_order_replaces_only_the_classes_it_names(self):
        book = orders.OrderBook()
        book.apply(reading(rule(box="big")))
        book.apply(reading(rule(quality="good", box="white")))
        self.assertEqual((book.destination_for(orders.PIN), book.destination_for(orders.NO_PIN)), ("white", "green"))

    def test_bad_has_pin_swaps_good_and_bad(self):
        book = orders.OrderBook(good_has_pin=False)
        book.apply(reading(rule(quality="good", box="white")))
        self.assertEqual(book.destination_for(orders.NO_PIN), "white")
        self.assertIsNone(book.destination_for(orders.PIN))

    def test_feature_names_the_class_directly(self):
        book = orders.OrderBook()
        book.apply(reading(rule(feature="no bolt", box="white")))
        self.assertEqual(book.destination_for(orders.NO_PIN), "white")

    def test_good_and_without_pin_contradict(self):
        book = orders.OrderBook()
        outcome = book.apply(reading(rule(feature="no bolt", quality="good", box="white")))
        self.assertEqual(outcome.kind, "contradiction")
        self.assertFalse(book.active)

    def test_quantity_counts_down_and_the_order_ends(self):
        book = orders.OrderBook()
        book.apply(reading(rule(quality="good", count="two", box="big")))
        self.assertEqual(book.routes[orders.PIN].remaining, 2)
        book.completed(orders.PIN)
        self.assertTrue(book.active)
        book.completed(orders.PIN)
        self.assertFalse(book.active)

    def test_a_quantity_belongs_to_the_rule_not_to_each_class(self):
        book = orders.OrderBook()
        book.apply(reading(rule(count="two", box="big")))  # "move two pieces to the big box": two in all
        book.completed(orders.PIN)
        self.assertEqual({r.remaining for r in book.routes.values()}, {1})
        book.completed(orders.NO_PIN)
        self.assertFalse(book.active)

    def test_picking_one_piece_ends_for_both_classes(self):
        book = orders.OrderBook()
        book.apply(reading(rule(count="one", box="none"), intent="pick"))
        book.completed(orders.NO_PIN)
        self.assertFalse(book.active)

    def test_separate_rules_count_separately(self):
        book = orders.OrderBook()
        book.apply(reading(rule(quality="good", count="two", box="white"), rule(quality="bad", box="big"),
                           intent="both"))
        book.completed(orders.PIN)
        book.completed(orders.NO_PIN)
        self.assertEqual(book.routes[orders.PIN].remaining, 1)
        self.assertIsNone(book.routes[orders.NO_PIN].remaining)
        book.completed(orders.PIN)
        self.assertEqual(set(book.routes), {orders.NO_PIN})

    def test_unlimited_order_never_ends(self):
        book = orders.OrderBook()
        book.apply(reading(rule(box="big")))
        for _ in range(5):
            book.completed(orders.PIN)
        self.assertTrue(book.active)

    def test_pick_without_a_box_means_hold(self):
        book = orders.OrderBook()
        book.apply(reading(rule(count="one", box="none"), intent="pick", status="ok"))
        self.assertEqual(book.destination_for(orders.PIN), orders.HOLD)

    def test_stop_clears_everything_in_every_language(self):
        for language, text in (("es", "Parado"), ("en", "Stopped"), ("eu", "Geldituta")):
            book = orders.OrderBook()
            book.preset("all_green")
            outcome = book.apply(reading(status="stop", intent="stop"), language)
            self.assertEqual(outcome.kind, "stop")
            self.assertTrue(outcome.message.startswith(text))
            self.assertFalse(book.active)

    def test_unnamed_box_other_box_and_no_action_change_nothing(self):
        for status, kind in (("ask_box", "ask_box"), ("not_a_cell_box", "not_a_cell_box"), ("no_action", "no_action"),
                             ("answer", "no_action")):
            book = orders.OrderBook()
            book.preset("all_white")
            outcome = book.apply(reading(rule(box="none"), status=status))
            self.assertEqual(outcome.kind, kind)
            self.assertEqual(book.destination_for(orders.PIN), "white")

    def test_low_probability_of_a_deciding_part_is_not_accepted(self):
        book = orders.OrderBook()
        low = rule(quality="good", box="white")
        low["p_box"] = 0.6
        outcome = book.apply(reading(low, status="unsure"))
        self.assertEqual(outcome.kind, "unsure")
        self.assertFalse(book.active)

    def test_an_unstated_count_does_not_veto_a_sure_order(self):
        book = orders.OrderBook()
        vague = rule(feature="no bolt", box="big", count="none")
        vague["p_count"] = 0.3  # the model's own gate would call this unsure; the count decides nothing here
        self.assertEqual(book.apply(reading(vague, status="unsure")).kind, "set")

    def test_unsure_intent_is_not_accepted(self):
        book = orders.OrderBook()
        self.assertEqual(book.apply(reading(rule(box="big"), p_intent=0.5, status="unsure")).kind, "unsure")

    def test_other_box_inside_a_rule_is_refused(self):
        book = orders.OrderBook()
        self.assertEqual(book.apply(reading(rule(box="other"))).kind, "not_a_cell_box")

    def test_presets_and_description(self):
        book = orders.OrderBook()
        self.assertEqual(book.describe(), "Sin orden activa")
        book.preset("sort")
        self.assertEqual((book.destination_for(orders.PIN), book.destination_for(orders.NO_PIN)), ("white", "green"))
        self.assertIn("CON perno", book.describe())
        self.assertIn("VERDE", book.describe())


if __name__ == "__main__":
    unittest.main()

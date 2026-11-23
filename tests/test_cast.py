import pytest

from memcraft.cast import (
    CastError,
    build_cast,
    chat_lines,
    director_lines,
    parse_names,
    tags_in,
    used,
)


def test_names_per_photo_as_lists_or_comma_strings():
    assert parse_names([["denys", " maksym "], "olga, ", []], 3) == [
        ["denys", "maksym"],
        ["olga"],
        [],
    ]
    assert parse_names(None, 2) == [[], []]
    # A friend who types the brackets too gets the bare name.
    assert parse_names(["<denys>"], 1) == [["denys"]]
    # Letters of any alphabet, digits, - and _.
    assert parse_names(["Денис, ann-marie, bob_2"], 1) == [["Денис", "ann-marie", "bob_2"]]


@pytest.mark.parametrize(
    ("raw", "count", "message"),
    [
        ([[]], 2, "Send the names for each photo once."),
        ("denys", 1, "Send the names for each photo once."),
        ([[1]], 1, "Photo 1: the names could not be read."),
        (["a,b,c,d,e,f"], 1, "Photo 1: name at most 5 people."),
        (["x" * 21], 1, "cannot be a name"),
        (["two words"], 1, "cannot be a name"),
        (["-dash"], 1, "cannot be a name"),
        (["a!"], 1, "cannot be a name"),
        (["person2"], 1, '"person2" is kept for the app'),
        (["Image1"], 1, '"Image1" is kept for the app'),
        (["denys", "Denys"], 2, 'The name "Denys" is used twice.'),
    ],
)
def test_bad_names(raw, count, message):
    with pytest.raises(CastError, match=message):
        parse_names(raw, count)


def test_twenty_characters_and_five_names_are_fine():
    names = ["a" * 20, "b", "c", "d", "e"]
    assert parse_names([names], 1) == [names]


def test_tags_in_an_idea():
    idea = '<denys> says "Hi!", <maksym> answers, <Denys> laughs. 2 < 3 > 1, <a b>'
    assert tags_in(idea) == ["denys", "maksym"]
    assert tags_in(None) == []


def test_the_cast_keeps_every_named_person():
    cast = build_cast([["denys", "maksym"], ["olga"]], '<denys> says "Hi!"')
    assert cast == [
        {"tag": "denys", "photo": 1, "position": 1},
        {"tag": "maksym", "photo": 1, "position": 2},
        {"tag": "olga", "photo": 2, "position": 1},
    ]
    assert build_cast([[], []], "no tags") is None
    assert build_cast([[]], None) is None


def test_tags_match_names_in_any_case():
    assert build_cast([["Denys"]], "<denys> waves")[0]["tag"] == "Denys"


def test_person_tags_follow_the_names_when_every_photo_has_them():
    cast = build_cast([["denys", "maksym"], ["olga"]], "<person3> hugs <person1>")
    assert cast[3:] == [
        {"tag": "person3", "photo": 2, "position": 1},
        {"tag": "person1", "photo": 1, "position": 1},
    ]
    with pytest.raises(CastError, match="<person4>"):
        build_cast([["denys", "maksym"], ["olga"]], "<person4> waves")


def test_person_tags_without_names_count_in_reading_order():
    assert build_cast([[], []], "<person2> trips over <person1>") == [
        {"tag": "person2", "photo": None, "position": 2},
        {"tag": "person1", "photo": None, "position": 1},
    ]
    with pytest.raises(CastError, match="<person16>"):
        build_cast([[]], "<person16> waves")
    with pytest.raises(CastError, match="<person0>"):
        build_cast([[]], "<person0> waves")


def test_unknown_tags_say_what_is_known():
    with pytest.raises(CastError) as error:
        build_cast([["denys"]], "<bob> and <denys> and <eve>")
    assert str(error.value) == (
        "The idea names <bob>, <eve>, but no one in the photos has that name. "
        "The names you gave are <denys>; you can also write <person1>, <person2>, …"
    )
    with pytest.raises(CastError, match="Type the names under the photos"):
        build_cast([[]], "<bob> waves")


def test_only_the_people_the_idea_names_go_to_the_models():
    cast = build_cast([["denys", "maksym"]], "<maksym> waves")
    assert used(cast, "<maksym> waves") == [{"tag": "maksym", "photo": 1, "position": 2}]
    assert used(cast, "nobody") == []
    assert used(None, "<maksym>") == []


def test_lines_for_gemma():
    cast = build_cast([["denys", "maksym"], ["olga", "ivan", "petro"], ["solo"]], None)
    idea = "<denys> <maksym> <olga> <ivan> <petro> <solo>"
    assert chat_lines(cast, idea) == [
        "<denys> = the person on the left in photo 1",
        "<maksym> = the person on the right in photo 1",
        "<olga> = the person on the left in photo 2",
        "<ivan> = the person in the middle in photo 2",
        "<petro> = the person on the right in photo 2",
        "<solo> = the person in photo 3",
    ]
    # Only maksym is named in the idea, but denys still makes him the one on the right.
    assert chat_lines(cast, "<maksym> waves") == ["<maksym> = the person on the right in photo 1"]
    four = build_cast([["a", "b", "c", "d"]], None)
    assert chat_lines(four, "<d>") == ["<d> = the 4th person from the left in photo 1"]
    generic = build_cast([[]], "<person2>")
    assert chat_lines(generic, "<person2>") == [
        "<person2> = the 2nd person, counting the people of photo 1 from left to right, "
        "then photo 2, then photo 3"
    ]
    assert chat_lines(cast, "no tags") == []


def test_lines_for_the_director():
    cast = build_cast([["denys", "maksym"], ["olga"]], None)
    assert director_lines(cast, "<olga> <maksym> <denys>") == [
        "<denys> = <image1>, the 1st of 2 people from the left",
        "<maksym> = <image1>, the 2nd of 2 people from the left",
        "<olga> = <image2>, the person in it (the leftmost if there are several)",
    ]
    assert director_lines(cast, "") == []

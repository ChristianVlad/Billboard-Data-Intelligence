import pandas as pd

from billboard_intel import genres


def test_split_preserves_slash_and_commas():
    assert genres.split_music_tags("Funk/Soul;Pop") == ["Funk/Soul", "Pop"]
    assert genres.split_music_tags("Folk, World, & Country") == ["Folk, World, & Country"]
    assert genres.split_music_tags(float("nan")) == []


def test_canonicalize_variants():
    assert genres.canonicalize_tag("Funk / Soul") == "Funk/Soul"
    assert genres.canonicalize_tag("  Pop ") == "Pop"


def _tags():
    df = pd.DataFrame({"Song": list("abcd"), "Artist": list("wxyz"),
                       "Date": pd.to_datetime(["1960-01-01", "1962-01-01", "2001-01-01", "2002-01-01"]),
                       "Epoch": pd.Categorical(["1958-1969", "1958-1969", "2000s", "2000s"]),
                       "G": ["Pop;Rock", "Pop", "Pop;Rock;Funk/Soul", None]})
    return genres.build_tag_table(df, "G")


def test_cooccurrence_weights_and_symmetry():
    nodes, edges = genres.cooccurrence(_tags())
    w = {(r.Source, r.Target): r.Weight for r in edges.itertuples()}
    assert w[("Pop", "Rock")] == 2 and w[("Funk/Soul", "Pop")] == 1
    assert (edges["Source"] < edges["Target"]).all()          # sin pares duplicados A-B / B-A
    assert nodes.set_index("Tag").loc["Pop", "Song_Count"] == 3


def test_association_metrics_known_values():
    a = genres.association_metrics(_tags()).set_index(["Source", "Target"])
    # Pop=3, Rock=2, juntos=2, total=3 → Jaccard = 2/(3+2-2) ; Lift = 2/(3*2/3)
    assert a.loc[("Pop", "Rock"), "Jaccard"] == 2 / 3
    assert a.loc[("Pop", "Rock"), "Lift"] == 1.0


def test_prevalence_denominator_is_classified_songs():
    p = genres.tag_prevalence(_tags())
    row = p[(p["Epoch"] == "2000s") & (p["Tag_Canonical"] == "Pop")].iloc[0]
    assert row["Classified_Songs"] == 1 and row["Percentage"] == 100.0   # la canción sin etiqueta no cuenta


def test_graph_and_metrics(results):
    nm = results.tables["node_metrics__discogs_genre"]
    assert {"Genre", "Song_Count", "Connected_Genres", "Cooccurrence_Strength"} <= set(nm.columns)
    assert len(results.tables["layout__discogs_genre"]) == len(nm)

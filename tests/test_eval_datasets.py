"""The generated datasets: determinism, and that the planted effects are really there.

WHY THIS FILE IS NOT OPTIONAL
-----------------------------
Every expected answer in the benchmark is derived from this data. If a generator
changes and the effect it was built to contain quietly disappears, the questions still
run, still produce numbers, and still report a score -- against data that no longer
means what the questions assume.

Building this suite already caught four such errors: three sensor event dates written
by hand and all off by one day, and an `audience_segment` column that was a perfect
alias of `channel` because both indexed a five-element list by `index % 5`. Neither
was visible in the generator; both were obvious the moment the data was queried.

These tests assert the EFFECTS, not the exact numbers. Asserting "Q3 profit is lower
than Q2" survives a tolerable amount of generator tuning; asserting "Q3 profit is
20259.27" would fail on any change and teach nothing about whether the story is intact.
"""

from __future__ import annotations

import duckdb
import pytest

from eval.datasets import BY_NAME, ecommerce, marketing, sensors


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Generate all three datasets once and expose a DuckDB connection over them."""
    directory = tmp_path_factory.mktemp("eval-data")
    con = duckdb.connect(":memory:")
    for name, spec in BY_NAME.items():
        path = spec.generate(directory)
        con.execute(
            f"CREATE VIEW \"{name}\" AS SELECT * FROM read_csv('{path.as_posix()}', sample_size=-1)"
        )
    return con


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def rows(con, sql):
    return con.execute(sql).fetchall()


# ----------------------------------------------------------------------- determinism


def test_generation_is_byte_identical_across_runs(tmp_path):
    """A committed expected answer is only meaningful if the data is reproducible."""
    first = ecommerce.SPEC.generate(tmp_path / "a")
    second = ecommerce.SPEC.generate(tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()


def test_a_different_seed_produces_different_data(tmp_path):
    """Confirms the seed is actually used, rather than the generator being constant."""
    directory = tmp_path / "c"
    directory.mkdir()
    default = directory / "default.csv"
    other = directory / "other.csv"
    ecommerce.build(default, ecommerce.SPEC.seed)
    ecommerce.build(other, ecommerce.SPEC.seed + 1)
    assert default.read_bytes() != other.read_bytes()


# ------------------------------------------------------------------------- ecommerce


def test_q3_revenue_rises_while_profit_falls(generated):
    """The flagship effect. Without it, the whole diagnosis category is meaningless."""
    result = dict(
        (q, (rev, profit))
        for q, rev, profit in rows(
            generated,
            "SELECT quarter(order_date), sum(revenue), sum(revenue - cost) "
            "FROM ecommerce GROUP BY 1",
        )
    )
    q2_revenue, q2_profit = result[2]
    q3_revenue, q3_profit = result[3]
    assert q3_revenue > q2_revenue
    assert q3_profit < q2_profit


def test_q3_has_both_planted_causes(generated):
    """Discounting deepens AND the low-margin category grows. Both, not either."""
    discounts = dict(
        rows(generated, "SELECT quarter(order_date), avg(discount_pct) FROM ecommerce GROUP BY 1")
    )
    assert discounts[3] > discounts[2] * 2

    shares = dict(
        rows(
            generated,
            "SELECT quarter(order_date), "
            "count(*) FILTER (WHERE category = 'Electronics') * 1.0 / count(*) "
            "FROM ecommerce GROUP BY 1",
        )
    )
    assert shares[3] > shares[2] * 1.8


def test_west_leads_on_revenue_and_trails_on_margin(generated):
    """The trap: the obvious answer and the correct answer differ."""
    by_revenue = rows(
        generated,
        "SELECT region FROM ecommerce GROUP BY 1 ORDER BY sum(revenue) DESC",
    )
    by_margin = rows(
        generated,
        "SELECT region FROM ecommerce GROUP BY 1 ORDER BY sum(revenue - cost) / sum(revenue) DESC",
    )
    assert by_revenue[0][0] == "West"
    assert by_margin[-1][0] == "West"


def test_revenue_leader_is_not_the_margin_leader(generated):
    """ecom-004 and ecom-005 must have different answers, or ecom-005 tests nothing."""
    top_revenue = scalar(
        generated,
        "SELECT category FROM ecommerce GROUP BY 1 ORDER BY sum(revenue) DESC LIMIT 1",
    )
    top_margin = scalar(
        generated,
        "SELECT category FROM ecommerce GROUP BY 1 "
        "ORDER BY sum(revenue - cost) / sum(revenue) DESC LIMIT 1",
    )
    assert top_revenue != top_margin


def test_apparel_is_returned_most(generated):
    worst = scalar(
        generated,
        "SELECT category FROM ecommerce GROUP BY 1 "
        "ORDER BY avg(CASE WHEN returned THEN 1.0 ELSE 0.0 END) DESC LIMIT 1",
    )
    assert worst == "Apparel"


def test_shipping_cost_has_the_planted_nulls(generated):
    fraction = scalar(
        generated,
        "SELECT (count(*) - count(shipping_cost)) * 1.0 / count(*) FROM ecommerce",
    )
    assert 0.01 < fraction < 0.035


# ------------------------------------------------------------------------- marketing


def test_variant_b_converts_better(generated):
    result = dict(
        rows(
            generated,
            "SELECT variant, sum(conversions) * 1.0 / sum(clicks) FROM marketing GROUP BY 1",
        )
    )
    assert result["B"] > result["A"] * 1.10


def test_conv_is_a_distinct_legacy_column(generated):
    """If `conv` ever equals `conversions`, the ambiguity trap has evaporated."""
    ratio = scalar(generated, "SELECT sum(conv) * 1.0 / sum(conversions) FROM marketing")
    assert 0.80 < ratio < 0.90
    disagreeing = scalar(
        generated, "SELECT count(*) FILTER (WHERE conv <> conversions) FROM marketing"
    )
    assert disagreeing > 1000


def test_cvr_is_conversion_rate_as_a_percentage(generated):
    ratio = scalar(generated, "SELECT avg(cvr) / avg(conversion_rate) FROM marketing")
    assert abs(ratio - 100.0) < 0.5


def test_cost_per_click_duplicates_cpc(generated):
    assert (
        scalar(generated, "SELECT count(*) FILTER (WHERE cpc <> cost_per_click) FROM marketing")
        == 0
    )


def test_audience_segment_is_not_an_alias_of_channel(generated):
    """Both lists have five entries; indexing them the same way made them identical.

    A cross-tab is the direct test: if each audience appears in exactly one channel,
    the column carries no independent information and every audience question is a
    channel question in disguise.
    """
    pairs = scalar(
        generated,
        "SELECT count(*) FROM (SELECT DISTINCT channel, audience_segment FROM marketing)",
    )
    channels = scalar(generated, "SELECT count(DISTINCT channel) FROM marketing")
    audiences = scalar(generated, "SELECT count(DISTINCT audience_segment) FROM marketing")
    assert pairs > max(channels, audiences)


def test_mobile_takes_the_most_clicks_and_converts_worst(generated):
    ordered = rows(
        generated,
        "SELECT device, sum(clicks) AS c, sum(conversions) * 1.0 / sum(clicks) AS r "
        "FROM marketing GROUP BY 1 ORDER BY c DESC",
    )
    assert ordered[0][0] == "Mobile"
    assert ordered[0][2] == min(row[2] for row in ordered)


def test_channel_returns_are_plausible(generated):
    """Display must lose money and nothing may return an absurd multiple.

    An earlier version had Email at 32x, which is arithmetically consistent and not a
    thing that happens; a benchmark answer nobody would believe is not worth asking for.
    """
    result = dict(
        rows(generated, "SELECT channel, sum(revenue) / sum(spend) FROM marketing GROUP BY 1")
    )
    assert result["Display"] < 1.0
    assert max(result.values()) < 12.0


def test_dead_columns_are_present(generated):
    assert scalar(generated, "SELECT count(DISTINCT account_currency) FROM marketing") == 1
    assert scalar(generated, "SELECT count(DISTINCT data_source) FROM marketing") == 1
    notes_null = scalar(generated, "SELECT 1 - count(notes) * 1.0 / count(*) FROM marketing")
    assert notes_null > 0.9


# --------------------------------------------------------------------------- sensors


def test_temperature_peaks_at_the_planted_hour(generated):
    hottest = scalar(
        generated,
        "SELECT hour(reading_time) FROM sensors GROUP BY 1 "
        "ORDER BY avg(temperature_c) DESC LIMIT 1",
    )
    assert hottest == sensors.DAILY_PEAK_HOUR


def test_the_hottest_sensor_differs_between_all_time_and_recently(generated):
    """The dilution trap. If these ever match, sens-010 stops testing anything."""
    all_period = scalar(
        generated,
        "SELECT sensor_id FROM sensors GROUP BY 1 "
        "ORDER BY avg(temperature_c) DESC NULLS LAST LIMIT 1",
    )
    recent = scalar(
        generated,
        "SELECT sensor_id FROM sensors WHERE reading_time >= TIMESTAMP '2024-06-16' "
        "GROUP BY 1 ORDER BY avg(temperature_c) DESC NULLS LAST LIMIT 1",
    )
    assert all_period != recent
    assert recent == sensors.DRIFT_SENSOR


def test_the_vibration_spike_is_on_the_derived_date(generated):
    """Guards the off-by-one that hand-written event dates produced."""
    sensor, day = rows(
        generated,
        "SELECT sensor_id, reading_time::DATE FROM sensors "
        "GROUP BY 1, 2 ORDER BY max(vibration_mm_s) DESC LIMIT 1",
    )[0]
    assert sensor == sensors.SPIKE_SENSOR
    assert day == sensors.SPIKE_DATE


def test_the_dead_sensor_keeps_its_rows_but_loses_its_readings(generated):
    """Row counts stay equal; only a null-aware count reveals the failure."""
    counts = dict(rows(generated, "SELECT sensor_id, count(*) FROM sensors GROUP BY 1"))
    assert len(set(counts.values())) == 1

    missing = dict(
        rows(
            generated,
            "SELECT sensor_id, 1 - count(temperature_c) * 1.0 / count(*) FROM sensors GROUP BY 1",
        )
    )
    assert missing[sensors.DEAD_SENSOR] == pytest.approx(sensors.DEAD_FRACTION, abs=0.001)
    assert all(v == 0 for k, v in missing.items() if k != sensors.DEAD_SENSOR)


def test_the_dead_sensor_is_not_also_the_hottest(generated):
    """The two effects must stay independent, or sens-010 is confounded by the outage."""
    assert sensors.DEAD_SENSOR != sensors.DRIFT_SENSOR
    hottest = scalar(
        generated,
        "SELECT sensor_id FROM sensors GROUP BY 1 "
        "ORDER BY avg(temperature_c) DESC NULLS LAST LIMIT 1",
    )
    assert hottest != sensors.DEAD_SENSOR


def test_last_reading_precedes_the_outage_date(generated):
    last = scalar(
        generated,
        f"SELECT max(reading_time)::DATE FROM sensors "
        f"WHERE sensor_id = '{sensors.DEAD_SENSOR}' AND temperature_c IS NOT NULL",
    )
    assert last < sensors.DEAD_FROM_DATE


def test_weekend_vibration_is_lower(generated):
    result = dict(
        rows(
            generated,
            "SELECT CASE WHEN dayofweek(reading_time) IN (0, 6) THEN 'weekend' "
            "ELSE 'weekday' END, avg(vibration_mm_s) FROM sensors GROUP BY 1",
        )
    )
    assert result["weekend"] < result["weekday"] * 0.7


# ----------------------------------------------------------------- spec bookkeeping


def test_every_spec_declares_its_effects():
    for spec in BY_NAME.values():
        assert spec.planted_effects, f"{spec.name} declares no planted effects"
        assert spec.seed


def test_marketing_is_actually_wide():
    assert len(marketing.COLUMNS) >= 40

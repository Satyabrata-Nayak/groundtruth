"""Industrial sensor readings: the TIME SERIES dataset. Hourly, 8 sensors, 90 days.

WHY A THIRD DATASET, AND WHY THIS SHAPE
---------------------------------------
E-commerce tests diagnosis and marketing tests schema comprehension. Neither tests
*time*, and time is where analytical SQL gets genuinely hard: bucketing to an hour or
a day, comparing a period against the one before it, separating a daily cycle from a
trend, and locating a window rather than a row.

It is also the dataset where the naive answer is most reliably wrong. "Which sensor
runs hottest?" over the whole period answers S-05, which is simply a hot machine
running normally. Over the final two weeks it answers S-07, which is a fault in
progress. An all-period average dilutes the fault into nothing, and the question
set asks it both ways so that the dilution is measurable rather than assumed.

WHAT IS PLANTED
---------------
    daily cycle       temperature peaks near 14:00 and troughs near 04:00, every day,
                      every sensor -- so an hourly average has real structure
    a drift fault     S-07 climbs steadily from day 60; by day 90 it is ~8C above
                      its neighbours, but its all-period average barely moves
    a spike window    S-03 vibration runs ~5x normal for six hours on day 45, and
                      nowhere else. It is invisible in a daily average and obvious
                      in an hourly one.
    a dead sensor     S-02 stops reporting on day 75: battery reaches zero and its
                      measurements become NULL while its rows keep arriving
    a weekly rhythm   vibration drops at weekends, because the machines are idle

The dead sensor is the data-quality trap. Its rows exist, so `count(*)` per sensor
looks even; only a null-aware count shows that a sixth of its readings are missing,
and any average that ignores this silently describes 75 days as if it were 90.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

from eval.datasets.base import DatasetSpec

DAYS = 90
HOURS = 24
START = datetime(2024, 4, 1, 0, 0, 0)

# (sensor_id, location, machine_type, baseline temperature, baseline vibration)
SENSORS = (
    ("S-01", "Line A", "Press", 41.0, 2.1),
    ("S-02", "Line A", "Conveyor", 33.5, 1.4),
    ("S-03", "Line B", "Press", 42.5, 2.3),
    ("S-04", "Line B", "Conveyor", 34.0, 1.5),
    ("S-05", "Line C", "Compressor", 57.5, 3.8),
    ("S-06", "Line C", "Press", 40.5, 2.0),
    ("S-07", "Line D", "Compressor", 54.0, 3.6),
    ("S-08", "Line D", "Conveyor", 33.0, 1.3),
)

# The daily cycle: amplitude in degrees, and the hour it peaks.
DAILY_AMPLITUDE_C = 6.5
DAILY_PEAK_HOUR = 14

# THE DRIFT FAULT.
DRIFT_SENSOR = "S-07"
DRIFT_START_DAY = 60
DRIFT_C_PER_DAY = 0.27  # ~8C by day 90

# THE SPIKE WINDOW.
SPIKE_SENSOR = "S-03"
SPIKE_DAY = 45
SPIKE_HOURS = range(8, 14)
SPIKE_MULTIPLIER = 5.0

# THE DEAD SENSOR.
DEAD_SENSOR = "S-02"
DEAD_FROM_DAY = 75

WEEKEND_VIBRATION_MULTIPLIER = 0.45

# The calendar dates those day offsets land on.
#
# These are DERIVED, never typed. The first version of this file spelled them out by
# hand and got all three wrong by one day -- April has 30 days, so day 45 is 16 May,
# not 15 May. Every one of those dates is quoted in a golden question's reference SQL,
# so a hand-written date would have made the question silently unanswerable: the query
# runs, returns the wrong window, and the "expected" value is confidently incorrect.
DRIFT_START_DATE = (START + timedelta(days=DRIFT_START_DAY)).date()
SPIKE_DATE = (START + timedelta(days=SPIKE_DAY)).date()
DEAD_FROM_DATE = (START + timedelta(days=DEAD_FROM_DAY)).date()
END_DATE = (START + timedelta(days=DAYS - 1)).date()

# Share of the dead sensor's readings that are missing, for the same reason.
DEAD_FRACTION = (DAYS - DEAD_FROM_DAY) / DAYS

COLUMNS = (
    "reading_id",
    "reading_time",
    "sensor_id",
    "location",
    "machine_type",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "vibration_mm_s",
    "battery_pct",
    "status",
)


def build(destination: Path, seed: int) -> int:
    rng = random.Random(seed)
    rows = 0

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)

        for day in range(DAYS):
            for hour in range(HOURS):
                moment = START + timedelta(days=day, hours=hour)
                is_weekend = moment.weekday() >= 5

                # One cosine over 24 hours, shifted so its maximum lands on the peak
                # hour. Shared by every sensor, so the cycle is a property of the day
                # rather than noise that happens to look periodic.
                cycle = math.cos((hour - DAILY_PEAK_HOUR) / HOURS * 2 * math.pi)

                for sensor_id, location, machine, base_temp, base_vibration in SENSORS:
                    rows += 1
                    dead = sensor_id == DEAD_SENSOR and day >= DEAD_FROM_DAY

                    if dead:
                        # Rows keep arriving; the measurements do not. This is what a
                        # failed sensor actually looks like in a warehouse, and it is
                        # why count(*) per sensor is a misleading completeness check.
                        writer.writerow(
                            [
                                rows,
                                moment.isoformat(sep=" "),
                                sensor_id,
                                location,
                                machine,
                                "",
                                "",
                                "",
                                "",
                                0,
                                "offline",
                            ]
                        )
                        continue

                    temperature = base_temp + DAILY_AMPLITUDE_C * cycle + rng.gauss(0, 0.8)
                    if sensor_id == DRIFT_SENSOR and day >= DRIFT_START_DAY:
                        temperature += (day - DRIFT_START_DAY) * DRIFT_C_PER_DAY

                    vibration = base_vibration * rng.uniform(0.85, 1.15)
                    if is_weekend:
                        vibration *= WEEKEND_VIBRATION_MULTIPLIER
                    if sensor_id == SPIKE_SENSOR and day == SPIKE_DAY and hour in SPIKE_HOURS:
                        vibration *= SPIKE_MULTIPLIER

                    humidity = 45 + 12 * math.sin(day / 14.0) + rng.gauss(0, 3.0)
                    pressure = 1013.0 - day * 0.05 + rng.gauss(0, 2.2)

                    # Battery falls over the whole period and is recharged on the 1st
                    # of each month, so it is a sawtooth rather than a straight line.
                    battery = max(5.0, 100.0 - (day % 30) * 2.6 - rng.uniform(0, 1.5))

                    status = "ok"
                    if vibration > base_vibration * 3:
                        status = "alarm"
                    elif battery < 15:
                        status = "low_battery"

                    writer.writerow(
                        [
                            rows,
                            moment.isoformat(sep=" "),
                            sensor_id,
                            location,
                            machine,
                            round(temperature, 2),
                            round(max(0.0, min(100.0, humidity)), 2),
                            round(pressure, 2),
                            round(vibration, 3),
                            round(battery, 1),
                            status,
                        ]
                    )

    return rows


SPEC = DatasetSpec(
    name="sensors",
    description=(
        "17,280 hourly readings from 8 industrial sensors over 90 days: temperature, "
        "humidity, pressure, vibration, battery and status."
    ),
    seed=20240722,
    planted_effects=(
        "Temperature follows a daily cycle peaking at 14:00 and troughing around 02:00-04:00.",
        f"S-07 drifts upward from {DRIFT_START_DATE}; over the last ten days it "
        f"averages about 6.6C above its own earlier baseline, while its all-period "
        f"average stays within ~0.3C of the other compressor.",
        f"S-03 vibration is about 5x normal for six hours on {SPIKE_DATE} "
        f"(08:00-13:00), and normal everywhere else. Its maximum is ~12.7 mm/s "
        f"against a ~2.0 mm/s average.",
        f"S-02 stops reporting on {DEAD_FROM_DATE}: its rows continue but every "
        f"measurement is NULL and its battery reads 0.",
        "Vibration at weekends is roughly 45% of the weekday level, across all sensors.",
        "Compressors (S-05, S-07) run hotter and vibrate more than presses or "
        "conveyors. S-05 is the hottest sensor over the whole period; S-07 is the "
        "hottest over the final two weeks, because of its drift.",
        f"Every sensor has exactly {DAYS * HOURS} rows, so only a null-aware count "
        f"reveals that S-02 is missing {DEAD_FRACTION:.1%} of its measurements.",
    ),
    build=build,
)

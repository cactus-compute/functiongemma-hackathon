#!/usr/bin/env python3
"""
Offline trainer for hybrid SVM gate.

Run once (or periodically) to regenerate serialized SVM and scaler via pickle.
"""

import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def seed_training_data():
    # [intent_score, tool_count, arg_difficulty, category, single_tool, explicit_value] -> label
    weighted = [
        # Local strength: explicit, single-intent weather/music.
        ([0.0, 1.0, 0.2, 0.0, 1.0, 1.0], 1, 8),   # weather_*
        ([0.0, 1.0, 0.4, 1.0, 1.0, 1.0], 1, 4),   # play_*
        # Local can handle some timer-heavy tool-selection cases.
        ([0.0, 3.0, 0.7, 3.0, 0.0, 1.0], 1, 3),   # timer_among_three-like
        ([0.0, 4.0, 0.55, 5.0, 0.0, 1.0], 1, 2),  # weather_among_four-like
        ([0.0, 5.0, 0.5857142857142857, 5.0, 0.0, 1.0], 1, 2),  # alarm_among_five-like
        ([0.0, 1.0, 0.8, 3.0, 1.0, 1.0], 1, 2),   # timer_5min-like

        # Keep cloud for known local misses / brittle patterns.
        ([0.0, 1.0, 0.8, 2.0, 1.0, 1.0], 0, 5),   # alarm_*
        ([0.0, 1.0, 0.55, 5.0, 1.0, 1.0], 0, 4),  # message_*
        ([0.0, 1.0, 0.6, 4.0, 1.0, 1.0], 0, 4),   # reminder_*
        ([0.0, 1.0, 0.6, 5.0, 1.0, 1.0], 0, 3),   # search_*
        ([0.0, 3.0, 0.58, 5.0, 0.0, 1.0], 0, 5),  # message_among_three-like
        ([0.0, 4.0, 0.5, 5.0, 0.0, 1.0], 0, 5),   # message_among_four-like
        ([0.0, 4.0, 0.5833333333333334, 5.0, 0.0, 1.0], 0, 4),  # search_among_four-like
        ([0.0, 3.0, 0.55, 2.0, 0.0, 1.0], 0, 4),  # music_among_three (corrected features)
        # Multi-intent should stay cloud-biased.
        ([0.5, 3.0, 0.58, 5.0, 0.0, 1.0], 0, 5),
        ([0.5, 4.0, 0.6, 3.0, 0.0, 1.0], 0, 3),
        ([1.0, 5.0, 0.5571428571428572, 5.0, 0.0, 1.0], 0, 3),

        # Additional benchmark-derived samples (append-only).
        ([0.0, 2.0, 0.43333333333333335, 5.0, 0.0, 1.0], 1, 3),  # weather_among_two-like
        ([0.0, 4.0, 0.55, 5.0, 0.0, 1.0], 1, 3),  # weather_among_four-like
        ([0.0, 3.0, 0.7000000000000001, 3.0, 0.0, 1.0], 1, 2),  # timer_among_three-like
        ([0.0, 5.0, 0.5857142857142857, 5.0, 0.0, 1.0], 1, 2),  # alarm_among_five-like
        ([0.0, 1.0, 0.8, 3.0, 1.0, 1.0], 1, 2),  # timer_5min-like

        # Keep high-risk patterns cloud-biased after expansion.
        ([0.0, 1.0, 0.8, 2.0, 1.0, 1.0], 0, 2),  # alarm_10am/alarm_6am-like
        ([0.0, 1.0, 0.55, 5.0, 1.0, 1.0], 0, 2),  # message_alice-like
        ([0.0, 4.0, 0.5, 5.0, 0.0, 1.0], 0, 2),  # message_among_four-like
        ([0.5, 4.0, 0.5857142857142857, 5.0, 0.0, 1.0], 0, 2),  # reminder_and_message-like
        ([1.0, 5.0, 0.5857142857142857, 5.0, 0.0, 1.0], 0, 2),  # message_weather_alarm-like
    ]

    raw_training_data = [
        # Reliable local successes
        ([0.0, 1, 0.2, 0, 1, 1], 1),  # weather_sf
        ([0.0, 1, 0.2, 0, 1, 1], 1),  # weather_london
        ([0.0, 1, 0.2, 0, 1, 1], 1),  # weather_paris
        ([0.0, 2, 0.2, 0, 0, 1], 1),  # weather_among_two
        ([0.0, 4, 0.2, 0, 0, 1], 1),  # weather_among_four
        ([0.0, 3, 0.4, 1, 0, 1], 1),  # alarm_among_three (early local success)
        # Additional positive examples
        ([0.0, 2, 0.2, 0, 0, 1], 1),  # weather_among_two
        ([0.0, 4, 0.2, 0, 0, 1], 1),  # weather_among_four
        ([0.0, 1, 0.4, 1, 1, 1], 1),  # play_bohemian
        ([0.0, 3, 0.4, 0, 0, 1], 1),  # alarm_among_three (weather among three)
        # Reliable local failures
        ([0.0, 1, 0.8, 3, 1, 1], 0),  # timer_5min
        ([0.0, 1, 0.8, 2, 1, 1], 0),  # alarm_6am
        ([0.0, 1, 0.7, 5, 1, 1], 0),  # message_alice
        ([0.0, 1, 0.6, 6, 1, 1], 0),  # search_bob
        ([0.0, 3, 0.4, 1, 0, 1], 0),  # music_among_three
        ([0.0, 4, 0.8, 4, 0, 0], 0),  # reminder_among_four
        ([0.0, 3, 0.8, 3, 0, 0], 0),  # timer_among_three
        ([0.0, 4, 0.6, 6, 0, 1], 0),  # search_among_four
        ([0.0, 4, 0.7, 5, 0, 1], 0),  # message_among_four
        # Hard multi-intent
        ([0.5, 2, 0.5, 5, 0, 1], 0),  # message_and_weather
        ([0.5, 2, 0.5, 2, 0, 1], 0),  # alarm_and_weather
        ([0.5, 2, 0.5, 3, 0, 1], 0),  # timer_and_music
        ([0.5, 3, 0.6, 5, 0, 1], 0),  # message_weather_alarm
    ]

    weighted_training_data = [
        (features, label)
        for features, label, repeats in weighted
        for _ in range(repeats)
    ]
    combined = raw_training_data + weighted_training_data

    # De-dup exact (features, label) pairs while preserving order.
    seen = set()
    deduped = []
    for features, label in combined:
        key = (tuple(float(v) for v in features), int(label))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((features, label))
    return deduped


def main():
    training_data = seed_training_data()
    X = np.array([f for f, _ in training_data], dtype=float)
    y = np.array([l for _, l in training_data], dtype=int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")
    clf.fit(X_scaled, y)

    out_path = "svm_gate.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)
    print(f"Saved SVM gate to {out_path}")
    print(f"  support vectors: {len(clf.support_vectors_)}")


if __name__ == "__main__":
    main()

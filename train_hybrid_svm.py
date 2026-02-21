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
    return [
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

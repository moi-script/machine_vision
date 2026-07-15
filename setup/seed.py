"""Opt-in seed script: loads demo players/sessions/attendance into MongoDB.

Mirrors the frontend's SEED_PLAYERS / SEED_SESSIONS in
`src/store/appStore.tsx` exactly (same ids, same field values).
"""
from app import db

ZONE_KEYS = [
    "front-left", "front-center", "front-right",
    "back-left", "back-center", "back-right",
]


def _zones(t):
    """Build a zones map from [shots, scores] tuples in ZONES order."""
    return {k: {"shots": t[i][0], "scores": t[i][1]} for i, k in enumerate(ZONE_KEYS)}


AVATAR_COLORS = [
    "#ff3366",  # red-pink       [0]
    "#ff5e00",  # orange         [1]
    "#00e676",  # neon green     [2]
    "#00f0ff",  # cyan           [3]
    "#b026ff",  # purple         [4]
    "#00b8ff",  # blue           [5]
    "#ff9100",  # yellow-orange  [6]
    "#ff007b",  # hot pink       [7]
]

PLAYERS = [
    {
        "_id": "p1", "jerseyNumber": 23, "name": "Marco Dela Cruz", "age": 22,
        "role": "player", "avatarColor": AVATAR_COLORS[3], "faceEnrolled": True,
        "joinedAt": "2023-01-15", "isActive": True,
        "trainingDays": ["2025-05-01", "2025-05-03", "2025-05-06", "2025-05-08", "2025-05-10", "2025-05-12"],
        "stats": {"totalShots": 480, "totalScores": 451,
                   "zones": _zones([[84, 82], [80, 79], [78, 74], [80, 75], [82, 80], [76, 61]])},
    },
    {
        "_id": "p2", "jerseyNumber": 2, "name": "Aisha Santos", "age": 19,
        "role": "player", "avatarColor": AVATAR_COLORS[7], "faceEnrolled": True,
        "joinedAt": "2023-03-08", "isActive": True,
        "trainingDays": ["2025-05-01", "2025-05-06", "2025-05-08", "2025-05-10", "2025-05-12"],
        "stats": {"totalShots": 360, "totalScores": 315,
                   "zones": _zones([[62, 58], [60, 56], [58, 52], [60, 44], [60, 55], [60, 50]])},
    },
    {
        "_id": "p3", "jerseyNumber": 3, "name": "Liam Reyes", "age": 16,
        "role": "player", "avatarColor": AVATAR_COLORS[2], "faceEnrolled": True,
        "joinedAt": "2024-02-20", "isActive": True,
        "trainingDays": ["2025-05-03", "2025-05-06", "2025-05-10"],
        "stats": {"totalShots": 240, "totalScores": 168,
                   "zones": _zones([[42, 32], [40, 30], [38, 22], [40, 28], [40, 30], [40, 26]])},
    },
    {
        "_id": "p4", "jerseyNumber": 4, "name": "Priya Navarro", "age": 24,
        "role": "player", "avatarColor": AVATAR_COLORS[4], "faceEnrolled": True,
        "joinedAt": "2022-09-01", "isActive": True,
        "trainingDays": ["2025-05-01", "2025-05-03", "2025-05-06", "2025-05-08", "2025-05-10", "2025-05-12"],
        "stats": {"totalShots": 540, "totalScores": 522,
                   "zones": _zones([[90, 88], [92, 90], [88, 85], [90, 86], [92, 89], [88, 84]])},
    },
    {
        "_id": "p5", "jerseyNumber": 5, "name": "Carlos Tan", "age": 18,
        "role": "player", "avatarColor": AVATAR_COLORS[6], "faceEnrolled": True,
        "joinedAt": "2023-08-14", "isActive": True,
        "trainingDays": ["2025-05-03", "2025-05-08", "2025-05-10"],
        "stats": {"totalShots": 300, "totalScores": 246,
                   "zones": _zones([[52, 44], [50, 42], [48, 40], [50, 40], [50, 34], [50, 46]])},
    },
    {
        "_id": "p6", "jerseyNumber": 6, "name": "Elena Ocampo", "age": 21,
        "role": "player", "avatarColor": AVATAR_COLORS[5], "faceEnrolled": True,
        "joinedAt": "2023-05-05", "isActive": False,
        "trainingDays": ["2025-04-20", "2025-04-24"],
        "stats": {"totalShots": 320, "totalScores": 269,
                   "zones": _zones([[54, 40], [52, 46], [54, 48], [52, 45], [54, 47], [54, 43]])},
    },
    {
        "_id": "p7", "jerseyNumber": 7, "name": "Jun Wei", "age": 26,
        "role": "staff", "avatarColor": AVATAR_COLORS[0], "faceEnrolled": True,
        "joinedAt": "2021-06-01", "isActive": True,
        "trainingDays": [],
        "stats": {"totalShots": 700, "totalScores": 686,
                   "zones": _zones([[118, 116], [116, 115], [116, 113], [116, 114], [118, 116], [116, 112]])},
    },
    {
        "_id": "p8", "jerseyNumber": 8, "name": "Sofia Bautista", "age": 17,
        "role": "player", "avatarColor": AVATAR_COLORS[1], "faceEnrolled": True,
        "joinedAt": "2024-09-01", "isActive": True,
        "trainingDays": ["2025-05-06", "2025-05-08", "2025-05-12"],
        "stats": {"totalShots": 180, "totalScores": 99,
                   "zones": _zones([[32, 22], [30, 20], [28, 17], [30, 14], [30, 15], [30, 11]])},
    },
]

SESSIONS = [
    {
        "_id": "s1", "title": "Elite Morning Drill", "coachId": "p7",
        "difficulty": "hard", "status": "live",
        "scheduledAt": "2025-05-13T07:00:00", "startedAt": "2025-05-13T07:03:00", "endedAt": None,
        "assignedPlayerIds": ["p1", "p4"],
        "liveData": [
            {"playerId": "p1", "joinedAt": "2025-05-13T07:03:00", "shots": 60, "scores": 52,
             "zones": _zones([[10, 9], [10, 9], [10, 8], [10, 9], [10, 9], [10, 8]])},
            {"playerId": "p4", "joinedAt": "2025-05-13T07:03:30", "shots": 60, "scores": 57,
             "zones": _zones([[10, 10], [10, 9], [10, 9], [10, 10], [10, 10], [10, 9]])},
        ],
        "notes": "Focus on backhand smash return.",
    },
    {
        "_id": "s2", "title": "Junior Footwork", "coachId": "p7",
        "difficulty": "easy", "status": "scheduled",
        "scheduledAt": "2025-05-13T10:00:00", "startedAt": None, "endedAt": None,
        "assignedPlayerIds": ["p3", "p8"],
        "liveData": [],
        "notes": "Shadow footwork + cone drill.",
    },
    {
        "_id": "s3", "title": "Senior Net Kill", "coachId": "p7",
        "difficulty": "medium", "status": "completed",
        "scheduledAt": "2025-05-12T08:00:00", "startedAt": "2025-05-12T08:05:00", "endedAt": "2025-05-12T09:35:00",
        "assignedPlayerIds": ["p2", "p6"],
        "liveData": [
            {"playerId": "p2", "joinedAt": "2025-05-12T08:05:00", "shots": 72, "scores": 63,
             "zones": _zones([[12, 11], [12, 11], [12, 10], [12, 9], [12, 11], [12, 11]])},
            {"playerId": "p6", "joinedAt": "2025-05-12T08:05:00", "shots": 66, "scores": 55,
             "zones": _zones([[11, 8], [11, 9], [11, 10], [11, 9], [11, 10], [11, 9]])},
        ],
        "notes": "Net kill accuracy review.",
    },
    {
        "_id": "s4", "title": "Intermediate Clear & Smash", "coachId": None,
        "difficulty": "medium", "status": "completed",
        "scheduledAt": "2025-05-10T09:00:00", "startedAt": "2025-05-10T09:02:00", "endedAt": "2025-05-10T10:20:00",
        "assignedPlayerIds": ["p5"],
        "liveData": [
            {"playerId": "p5", "joinedAt": "2025-05-10T09:02:00", "shots": 60, "scores": 49,
             "zones": _zones([[10, 9], [10, 8], [10, 8], [10, 8], [10, 7], [10, 9]])},
        ],
        "notes": "",
    },
    {
        "_id": "s5", "title": "Elite Reaction Block", "coachId": "p7",
        "difficulty": "medium", "status": "completed",
        "scheduledAt": "2025-05-08T07:00:00", "startedAt": "2025-05-08T07:02:00", "endedAt": "2025-05-08T08:10:00",
        "assignedPlayerIds": ["p1", "p4"],
        "liveData": [
            {"playerId": "p1", "joinedAt": "2025-05-08T07:02:00", "shots": 66, "scores": 55,
             "zones": _zones([[11, 9], [11, 10], [11, 9], [11, 9], [11, 10], [11, 8]])},
            {"playerId": "p4", "joinedAt": "2025-05-08T07:02:00", "shots": 66, "scores": 61,
             "zones": _zones([[11, 10], [11, 11], [11, 10], [11, 10], [11, 10], [11, 10]])},
        ],
        "notes": "",
    },
]


def _attendance():
    rows = []
    for s in SESSIONS:
        for pid in s["assignedPlayerIds"]:
            rows.append({
                "playerId": pid,
                "sessionId": s["_id"],
                "date": s["scheduledAt"].split("T")[0],
                "present": s["status"] != "cancelled",
                "lateMinutes": 0,
            })
    return rows


def clear_db():
    db.players().delete_many({})
    db.sessions().delete_many({})
    db.attendance().delete_many({})
    print("cleared players, sessions, attendance")


def seed():
    db.players().delete_many({})
    db.sessions().delete_many({})
    db.attendance().delete_many({})
    db.players().insert_many(PLAYERS)
    db.sessions().insert_many(SESSIONS)
    db.attendance().insert_many(_attendance())
    print(f"seeded {len(PLAYERS)} players, {len(SESSIONS)} sessions")


if __name__ == "__main__":
    import sys
    if "--clear" in sys.argv:
        clear_db()
    else:
        seed()

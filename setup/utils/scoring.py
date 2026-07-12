# ============================================================
# scoring.py — Individual player scoring per zone
# ============================================================

from config.settings import PLAYER_ZONES, ZONE_WEAK_THRESHOLD


class PlayerScores:
    """
    Tracks accumulated score per player ID per zone.
    Each player gets a score entry the first time they are detected.
    """

    def __init__(self):
        self.scores = {}
        # { player_id: { zone_name: { score, shots }, total_score, total_shots } }

    def init_player(self, player_id):
        if player_id not in self.scores:
            self.scores[player_id] = {
                zone: {"score": 0, "shots": 0}
                for zone in PLAYER_ZONES.keys()
            }
            self.scores[player_id]["total_score"] = 0
            self.scores[player_id]["total_shots"] = 0
            print(f"[SCORING] Player {player_id} initialized")

    def record_shot(self, player_id, zone):
        """Called when shuttle crosses net into a zone targeting this player"""
        self.init_player(player_id)
        if zone in self.scores[player_id]:
            self.scores[player_id][zone]["shots"]  += 1
            self.scores[player_id]["total_shots"]  += 1

    def record_score(self, player_id, zone):
        """Called when player successfully returns the shuttle"""
        self.init_player(player_id)
        if zone in self.scores[player_id]:
            self.scores[player_id][zone]["score"]  += 1
            self.scores[player_id]["total_score"]  += 1

    def get_zone_accuracy(self, player_id, zone):
        """Returns accuracy % for a specific player and zone"""
        if player_id not in self.scores:
            return 0.0
        data  = self.scores[player_id][zone]
        shots = data["shots"]
        if shots == 0:
            return 0.0
        return (data["score"] / shots) * 100

    def get_total_accuracy(self, player_id):
        """Returns overall accuracy % for a player"""
        if player_id not in self.scores:
            return 0.0
        total = self.scores[player_id]["total_shots"]
        if total == 0:
            return 0.0
        return (self.scores[player_id]["total_score"] / total) * 100

    def get_weak_zones(self, player_id):
        """Returns list of zones where player accuracy < threshold"""
        weak = []
        if player_id not in self.scores:
            return weak
        for zone in PLAYER_ZONES.keys():
            acc = self.get_zone_accuracy(player_id, zone)
            shots = self.scores[player_id][zone]["shots"]
            if shots > 0 and acc < ZONE_WEAK_THRESHOLD:
                weak.append(zone)
        return weak

    def print_assessment(self):
        """Print full drill assessment for all players"""
        print("\n" + "=" * 50)
        print("         PLAYER ASSESSMENT REPORT")
        print("=" * 50)

        for pid, data in self.scores.items():
            total_shots = data["total_shots"]
            total_score = data["total_score"]
            accuracy    = self.get_total_accuracy(pid)
            weak_zones  = self.get_weak_zones(pid)

            print(f"\n  Player {pid}")
            print(f"  Overall: {total_score}/{total_shots} shots ({accuracy:.1f}%)")
            print(f"  {'Zone':<14} {'Hit':>4} {'Shot':>5} {'Acc':>6}")
            print(f"  {'-' * 34}")

            for zone in PLAYER_ZONES.keys():
                s   = data[zone]["score"]
                sh  = data[zone]["shots"]
                acc = self.get_zone_accuracy(pid, zone)
                flag = " ← weak!" if zone in weak_zones else ""
                print(f"  {zone:<14} {s:>4} {sh:>5} {acc:>5.0f}%{flag}")

            if weak_zones:
                print(f"\n  Focus areas: {', '.join(weak_zones)}")
            else:
                print(f"\n  Strong all-round performance!")

        print("\n" + "=" * 50)

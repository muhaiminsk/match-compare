import os, requests
h = {"x-apisports-key": os.environ["APIFOOTBALL_KEY"]}
base = "https://v3.football.api-sports.io"

lg = requests.get(f"{base}/leagues", headers=h, params={"id": 140}).json()   # 140 = La Liga
print("errors:", lg.get("errors"))
for s in lg["response"][0]["seasons"]:
    c = s["coverage"]["fixtures"]
    print(s["year"], "player stats:", c["statistics_players"], "| fixture stats:", c["statistics_fixtures"], "| current:", s["current"])

r = requests.get(f"{base}/fixtures", headers=h, params={"league": 140, "season": 2026, "team": 529}).json()  # 529 = Barcelona
print("fixtures errors:", r.get("errors"), "| fixtures returned:", len(r.get("response", [])))

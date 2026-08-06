"""
Kaggriculture agent.

Strategy summary
----------------
* Livestock + daily CARE is the strongest engine in the game. The banked care
  bonus is paid out in full on the next scheduled production, so a sheep cared
  for every day yields 4 wool per 3 days (~$267/tile-day) and a cow 3 milk per
  2 days (~$240/tile-day). That beats every crop by a wide margin.
* Melon is the best crop (~$150/tile-day) but only the town centre demands it,
  so it is capped at ~16 tiles.
* Wheat is grown on-farm to feed animals; market wheat gets expensive because
  the town drains ~630 units a season.
* Fertilizer is spent on ongoing crops (doubles each scheduled production:
  +4 units on a strawberry for ~2 fertilizer) and the surplus is sold.
* Farm hands cost fib(n) and reset daily -- 12 hands is only $376/day for 288
  extra actions. Hire aggressively.
* Selling is paced, not dumped. Premium goods (strawberry, milk, wool, melon)
  hit the $1 floor after only ~30-110 units of glut, but the town drains stock
  every turn and lifts the price back up. We sell down to a price floor each
  turn and let demand regenerate.

The agent is defensive: every decision path is wrapped so that a malformed or
unexpected observation degrades to PASS rather than erroring out the episode.
"""

import math

# ----------------------------------------------------------------------------
# Static game data
# ----------------------------------------------------------------------------

TURNS_PER_DAY = 24
SEASON_DAYS = 30
LAST_DAY = SEASON_DAYS - 1

SHAPES = {
    "linear": lambda x: x,
    "sq": lambda x: x * x,
    "sqrt": lambda x: math.sqrt(x),
    "log": lambda x: math.log(1.0 + x),
    "log10": lambda x: math.log10(1.0 + x),
}

# base, I0, T, (below_func, below_target), (above_func, above_target)
MARKET_PARAMS = {
    "WHEAT":      (25.0, 10000, 400, ("sqrt", 0.80), ("log", 0.20)),
    "CARROT":     (35.0, 10000, 450, ("log", 0.20), ("sqrt", 0.70)),
    "TOMATO":     (60.0, 10000, 200, ("linear", 0.40), ("sqrt", 0.60)),
    "STRAWBERRY": (120.0, 10000, 100, ("sqrt", 0.70), ("linear", 1.60)),
    "MELON":      (250.0, 10000, 300, ("log", 0.20), ("sq", 3.60)),
    "EGG":        (50.0, 10000, 332, ("linear", 0.40), ("log", 0.20)),
    "MILK":       (160.0, 10000, 122, ("sqrt", 0.60), ("linear", 1.60)),
    "WOOL":       (200.0, 10000, 105, ("log", 0.20), ("sq", 3.20)),
    "FERTILIZER": (100.0, 10000, 200, ("linear", 0.40), ("linear", 0.40)),
}

# crop -> parameters.  bonus window starts at ceil(max_yield_day / 2).
CROPS = {
    "WHEAT": {
        "seed": 10, "first": 2, "maxday": 4, "ongoing": False,
        "yield_water": 4, "yield_fert": 6, "product": "WHEAT",
    },
    "CARROT": {
        "seed": 20, "first": 2, "maxday": 3, "ongoing": False,
        "yield_water": 3, "yield_fert": 4, "product": "CARROT",
    },
    "MELON": {
        "seed": 80, "first": 10, "maxday": 10, "ongoing": False,
        "yield_water": 6, "yield_fert": 6, "product": "MELON",
    },
    "TOMATO": {
        "seed": 50, "first": 8, "maxday": 11, "ongoing": True,
        "prod": (8, 9, 10, 11), "product": "TOMATO",
    },
    "STRAWBERRY": {
        "seed": 100, "first": 10, "maxday": 16, "ongoing": True,
        "prod": (10, 12, 14, 16), "product": "STRAWBERRY",
    },
}
for _c, _i in CROPS.items():
    _i["bonus_start"] = int(math.ceil(_i["maxday"] / 2.0))

ANIMALS = {
    "GOOSE": {"cost": 300, "struct": "COOP", "product": "EGG",
              "first": 4, "interval": 1, "max_held": 4},
    "COW":   {"cost": 400, "struct": "PASTURE", "product": "MILK",
              "first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"cost": 500, "struct": "PASTURE", "product": "WOOL",
              "first": 6, "interval": 3, "max_held": 6},
}
STRUCT_ANIMALS = {"COOP": ["GOOSE"], "PASTURE": ["SHEEP", "COW"]}

LAND_COSTS = [1000, 2000, 4000]
FIB = [1, 1]
while len(FIB) < 20:
    FIB.append(FIB[-1] + FIB[-2])


def fib_sum(k):
    """Total cost of hiring k hands in one day."""
    return sum(FIB[:max(0, k)])


# ----------------------------------------------------------------------------
# Market price model (reproduces the documented curve exactly)
# ----------------------------------------------------------------------------

def price_at(product, inv):
    p = MARKET_PARAMS.get(product)
    if p is None:
        return 1
    base, i0, t, below, above = p
    d = inv - i0
    if d == 0:
        return int(round(base))
    if d < 0:
        fname, target, sign = below[0], below[1], 1.0
    else:
        fname, target, sign = above[0], above[1], -1.0
    f = SHAPES.get(fname, SHAPES["linear"])
    denom = f(t)
    if denom <= 0:
        return int(round(base))
    amp = target * base / denom
    return max(1, int(round(base + sign * amp * f(abs(d)))))


def sellable_count(product, inv, have, min_price, cap=None):
    """How many units can be sold before the marginal price drops below
    min_price.  Mirrors the engine: price is quoted at the pre-sell inventory,
    and units sold at the $1 floor do not add to inventory."""
    if have <= 0:
        return 0
    limit = have if cap is None else min(have, cap)
    n, cur = 0, inv
    while n < limit:
        pr = price_at(product, cur)
        if pr < min_price:
            break
        n += 1
        if pr > 1:
            cur += 1
    return n


# ----------------------------------------------------------------------------
# Observation access helpers (tolerant of dict / Struct style observations)
# ----------------------------------------------------------------------------

def _get(obj, key, default=None):
    try:
        v = obj[key]
    except Exception:
        try:
            v = getattr(obj, key)
        except Exception:
            return default
    return default if v is None else v


def _num(v, default=0):
    try:
        return float(v)
    except Exception:
        return default


def _as_dict(v):
    if isinstance(v, dict):
        return v
    try:
        return dict(v)
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Direction calibration
#
# The board is indexed tiles[y][x].  We assume NORTH decreases y and EAST
# increases x, but we verify it at runtime from the main farmer's movement and
# self-correct if the engine disagrees.
# ----------------------------------------------------------------------------

# Keyed by player id: two instances of this agent may share one process (the
# Kaggle validation episode is a self-play match), and a shared compass would
# have each instance calibrating against the other's farmer.
_STATES = {}


def compass(player):
    st = _STATES.get(player)
    if st is None:
        st = {"ns": -1, "ew": 1, "probe": None}
        _STATES[player] = st
    return st


def _calibrate(st, step, fx, fy):
    probe = st.get("probe")
    st["probe"] = None
    if not probe or probe["step"] != step - 1:
        return
    d, px, py = probe["dir"], probe["x"], probe["y"]
    if d in ("NORTH", "SOUTH") and fy != py:
        dy = 1 if fy > py else -1
        st["ns"] = dy if d == "NORTH" else -dy
    elif d in ("EAST", "WEST") and fx != px:
        dx = 1 if fx > px else -1
        st["ew"] = dx if d == "EAST" else -dx


def move_toward(x, y, tx, ty, st):
    """One movement op that reduces Manhattan distance, or None if arrived."""
    if x != tx:
        return "EAST" if (tx - x) * st["ew"] > 0 else "WEST"
    if y != ty:
        return "NORTH" if (ty - y) * st["ns"] > 0 else "SOUTH"
    return None


def dist(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


# ----------------------------------------------------------------------------
# Farm survey
# ----------------------------------------------------------------------------

class Farm(object):
    """Digest of the observation into everything the planner needs."""

    def __init__(self, obs):
        self.player = int(_num(_get(obs, "player", 0)))
        self.step = int(_num(_get(obs, "step", 0)))
        self.day = int(_num(_get(obs, "day", 0)))
        self.hour = int(_num(_get(obs, "hour", 0)))

        farms = _get(obs, "farms", []) or []
        self.me = farms[self.player] if len(farms) > self.player else {}
        self.money = _num(_get(self.me, "money", 0))
        self.tiles = _get(self.me, "tiles", []) or []
        self.size = len(self.tiles) if self.tiles else 10
        self.half = self.size // 2
        self.shed_tiles = [
            (self.half - 1, self.half - 1), (self.half, self.half - 1),
            (self.half - 1, self.half), (self.half, self.half),
        ]
        self.unlocked_quads = list(_get(self.me, "unlocked_quadrants", ["NW"]) or ["NW"])
        self.hires_today = int(_num(_get(self.me, "hires_today", 0)))

        priv = _get(obs, "private", {}) or {}
        self.shed = _as_dict(_get(priv, "shed", {}))
        self.seeds = _as_dict(_get(priv, "seeds", {}))
        raw_inv = _get(priv, "inventories", []) or []
        self.inventories = [_as_dict(i) for i in raw_inv]

        market = _get(obs, "market", {}) or {}
        self.mkt_inv = _as_dict(_get(market, "inventory", {}))
        self.mkt_price = _as_dict(_get(market, "prices", {}))

        # unit positions: index 0 is the main farmer, then hands in order
        farmer = _get(self.me, "farmer", [self.half - 1, self.half - 1])
        try:
            self.units = [(int(farmer[0]), int(farmer[1]))]
        except Exception:
            self.units = [(self.half - 1, self.half - 1)]
        self.n_hands = 0
        for h in (_get(self.me, "hands", []) or []):
            try:
                self.units.append((int(h[0]), int(h[1])))
                self.n_hands += 1
            except Exception:
                pass
        while len(self.inventories) < len(self.units):
            self.inventories.append({})

        self._scan()

    # -- tile scanning -----------------------------------------------------
    def _scan(self):
        self.empty = []          # (x, y) unlocked and empty
        self.plants = []         # (x, y, tile)
        self.structs = []        # (x, y, tile)  coop / pasture
        self.weeds = []          # (x, y)
        self.n_unlocked = 0
        self.crop_counts = {}
        self.animal_counts = {}
        self.n_pasture = 0
        self.n_coop = 0
        self.free_struct = {"COOP": [], "PASTURE": []}

        for y, row in enumerate(self.tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED" or tile is None and not self._in_unlocked(x, y):
                    if tile == "LOCKED":
                        continue
                if isinstance(tile, str):
                    continue
                self.n_unlocked += 1
                if tile is None:
                    self.empty.append((x, y))
                    continue
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                if kind == "PLANT":
                    self.plants.append((x, y, tile))
                    c = tile.get("crop")
                    self.crop_counts[c] = self.crop_counts.get(c, 0) + 1
                elif kind == "WEED":
                    self.weeds.append((x, y))
                elif kind in ("COOP", "PASTURE"):
                    self.structs.append((x, y, tile))
                    if kind == "COOP":
                        self.n_coop += 1
                    else:
                        self.n_pasture += 1
                    a = tile.get("animal")
                    if a:
                        self.animal_counts[a] = self.animal_counts.get(a, 0) + 1
                    else:
                        self.free_struct[kind].append((x, y))

        self.empty.sort(key=lambda p: (self._shed_dist(p[0], p[1]), p[1], p[0]))
        self.n_animals = sum(self.animal_counts.values())
        # animals bought but not yet placed
        self.spare_animals = {}
        for a in ANIMALS:
            n = int(_num(self.shed.get(a, 0)))
            for inv in self.inventories:
                n += int(_num(inv.get(a, 0)))
            if n:
                self.spare_animals[a] = n

    def _in_unlocked(self, x, y):
        return True

    def _shed_dist(self, x, y):
        return min(dist(x, y, sx, sy) for sx, sy in self.shed_tiles)

    # -- convenience -------------------------------------------------------
    def price(self, product):
        v = self.mkt_price.get(product)
        if v is None:
            inv = self.mkt_inv.get(product)
            if inv is None:
                base = MARKET_PARAMS.get(product, (1,))[0]
                return float(base)
            return float(price_at(product, int(_num(inv, 10000))))
        return max(1.0, _num(v, 1))

    def inv_of(self, product):
        v = self.mkt_inv.get(product)
        return int(_num(v, 10000)) if v is not None else 10000

    def shed_total(self):
        return sum(int(_num(v, 0)) for v in self.shed.values())

    def nearest_shed(self, x, y):
        return min(self.shed_tiles, key=lambda s: (dist(x, y, s[0], s[1]), s[1], s[0]))


# ----------------------------------------------------------------------------
# Composition targets
# ----------------------------------------------------------------------------

# --- tunables -------------------------------------------------------------
MAX_SHEEP = 8
MAX_COW = 8
MAX_GOOSE = 6
TILES_PER_ANIMAL = 11
MAX_QUADS = 3
MAX_HANDS = 12
SAFETY_WATER = False
LABOR_CAP = False


def target_animals(farm):
    """How many of each animal we ultimately want, scaled to unlocked land."""
    u = farm.n_unlocked
    sheep = min(MAX_SHEEP, max(0, u // TILES_PER_ANIMAL))
    cow = min(MAX_COW, max(0, u // TILES_PER_ANIMAL))
    goose = min(MAX_GOOSE, max(0, u // (TILES_PER_ANIMAL + 7)))
    out = {"SHEEP": sheep, "COW": cow, "GOOSE": goose}
    for a, deadline in BUY_DEADLINE.items():
        if farm.day > deadline:
            out[a] = farm.animal_counts.get(a, 0) + farm.spare_animals.get(a, 0)
    return out


# last day a purchase still reaches a useful number of productions
BUY_DEADLINE = {"COW": 19, "SHEEP": 20, "GOOSE": 23}


def buy_order(day):
    """Cows need the longest lead time, so they get first claim on pasture."""
    return ("COW", "SHEEP", "GOOSE") if day <= 14 else ("SHEEP", "COW", "GOOSE")


CAP_MELON = 24
GLUT_OK = 0.55   # price/base below this means the good is gluting
CAP_STRAWBERRY = 10
CAP_TOMATO = 6
CAP_CARROT = 8
CAP_WHEAT = 22


def price_health(farm, product):
    """Live price as a fraction of base price, clamped to [0, 1].

    Melon's curve above I0 is ('sq', 3.6) over t=300: a squared falloff that
    takes it from $250 to $25 on ~250 units of surplus and to $1 by ~500.
    A fixed tile cap cannot know how much of that headroom the opponent has
    already burned, so the cap has to read the market instead of assuming it.
    """
    prm = MARKET_PARAMS.get(product)
    if not prm:
        return 1.0
    base = prm[0]
    if base <= 0:
        return 1.0
    return max(0.0, min(1.0, farm.price(product) / base))


def crop_caps(farm):
    """Tile caps per crop, demand-limited then scaled to unlocked land.

    Caps for the price-fragile crops are throttled by live market health, so a
    crop that is crashing stops consuming tiles and labour that a still-scarce
    crop can use. Goods trading at or above base are untouched.
    """
    scale = farm.n_unlocked / 100.0
    animals = farm.n_animals + sum(farm.spare_animals.values())

    def throttle(cap, product, floor=0.15):
        h = price_health(farm, product)
        if h >= GLUT_OK:
            return cap
        f = max(floor, h / GLUT_OK)
        return int(round(cap * f))

    return {
        "MELON": max(2, throttle(int(round(CAP_MELON * scale)), "MELON")),
        "STRAWBERRY": max(1, throttle(int(round(CAP_STRAWBERRY * scale)),
                                      "STRAWBERRY")),
        "TOMATO": max(1, throttle(int(round(CAP_TOMATO * scale)), "TOMATO")),
        "WHEAT": min(CAP_WHEAT, max(5, animals + 4)),
        "CARROT": max(4, int(round(CAP_CARROT * scale))),
    }


def crop_score(farm, crop):
    """Expected coins per tile-day if we plant this crop right now.
    Uses live market prices so the agent reacts to a crashed product."""
    info = CROPS[crop]
    day = farm.day
    price = farm.price(info["product"])
    have_fert = int(_num(farm.shed.get("FERTILIZER", 0))) > 6
    if not info["ongoing"]:
        cycle = info["maxday"]
        if day + cycle > LAST_DAY:
            return -1.0
        units = info["yield_water"]
        return (units * price - info["seed"]) / float(cycle)
    prods = [a for a in info["prod"] if day + a <= LAST_DAY]
    if not prods:
        return -1.0
    per = 2.0 if have_fert else 1.0
    occupancy = min(LAST_DAY - day, info["maxday"] + 1)
    return (len(prods) * per * price - info["seed"]) / float(max(1, occupancy))


def plan_empty_tiles(farm):
    """Decide what each currently-empty tile should become.
    Returns (list of (x, y, 'BUILD_COOP'|'BUILD_PASTURE'|crop), want_seeds)."""
    plan = []
    want_seeds = {}
    if farm.day >= LAST_DAY:
        return plan, want_seeds

    counts = dict(farm.crop_counts)
    caps = crop_caps(farm)
    tgt = target_animals(farm)

    # Labour budget: roughly 2.8 actions per plant per day (watering plus the
    # walking to reach it) and 5 per animal. Over-planting past this is how a
    # whole cohort of crops dies of thirst on the same night.
    crew = max(len(farm.units), min(MAX_HANDS, hands_target(farm)) + 1)
    capacity = (crew * TURNS_PER_DAY - farm.n_animals * 5) / 2.8
    allowed = int(max(0, capacity * 0.9) - len(farm.plants))
    allowed = min(allowed, max(4, int(capacity // 4)))
    if not LABOR_CAP:
        allowed = 999

    # structures: build slightly ahead of animals actually owned so tiles are
    # never left idle waiting on cash
    owned_p = farm.animal_counts.get("SHEEP", 0) + farm.animal_counts.get("COW", 0)
    owned_c = farm.animal_counts.get("GOOSE", 0)
    spare_p = farm.spare_animals.get("SHEEP", 0) + farm.spare_animals.get("COW", 0)
    spare_c = farm.spare_animals.get("GOOSE", 0)
    want_pasture = min(tgt["SHEEP"] + tgt["COW"], owned_p + spare_p + 2) - farm.n_pasture
    want_coop = min(tgt["GOOSE"], owned_c + spare_c + 1) - farm.n_coop

    for (x, y) in farm.empty:
        on_shed = (x, y) in farm.shed_tiles
        if want_pasture > 0 and not on_shed:
            plan.append((x, y, "BUILD_PASTURE"))
            want_pasture -= 1
            continue
        if want_coop > 0 and not on_shed:
            plan.append((x, y, "BUILD_COOP"))
            want_coop -= 1
            continue
        if allowed <= 0:
            continue
        best, best_s = None, 0.0
        for crop in CROPS:
            if counts.get(crop, 0) >= caps.get(crop, 0):
                continue
            s = crop_score(farm, crop)
            if s > best_s:
                best, best_s = crop, s
        if best is None:
            continue
        counts[best] = counts.get(best, 0) + 1
        allowed -= 1
        plan.append((x, y, best))
        want_seeds[best] = want_seeds.get(best, 0) + 1
    return plan, want_seeds


# ----------------------------------------------------------------------------
# Task generation
# ----------------------------------------------------------------------------

# tier constants (higher = done first)
T_SURVIVE = 100   # plant would weed / animal would escape
T_RESCUE = 92     # produce about to decay or overflow its cap
T_FEED = 84
T_WATER = 78      # watering that actually adds yield
T_CARE = 72
T_PLACE = 62
T_WATER_SAFE = 66  # top up a plant early to rebuild its safety margin
T_FERT = 60       # doubles every scheduled yield of an ongoing crop
T_HARVEST = 58
T_BUILD = 56
T_PLANT = 52
T_DUNG = 34
T_DIG = 28


def next_prod_age(info, age, strictly_after=False):
    """Age at which this animal next produces."""
    f, iv = info["first"], max(1, info["interval"])
    if age < f:
        return f
    off = (age - f) % iv
    if off == 0 and not strictly_after:
        return age
    return age + (iv - off if off else iv)


def plant_tasks(farm, tasks):
    day, step = farm.day, farm.step
    for (x, y, tile) in farm.plants:
        crop = tile.get("crop")
        info = CROPS.get(crop)
        if info is None:
            continue
        price = farm.price(info["product"])
        age = day - int(_num(tile.get("planted_day", day)))
        units = int(_num(tile.get("yield_units", 0)))
        watered = bool(tile.get("watered_today", False))
        unwatered = int(_num(tile.get("consecutive_unwatered", 0)))
        fert_until = int(_num(tile.get("fertilized_until_day", -1)))
        life = int(_num(tile.get("max_lifespan_step", -1)))
        decaying = life >= 0 and step >= life - 1

        # ---- harvest ----
        if units > 0:
            if not info["ongoing"]:
                if decaying:
                    tasks.append(mk(T_RESCUE, units * price, x, y, ["HARVEST"]))
                elif age >= info["maxday"]:
                    tasks.append(mk(T_HARVEST, units * price, x, y, ["HARVEST"]))
                elif day >= LAST_DAY:
                    tasks.append(mk(T_HARVEST, units * price, x, y, ["HARVEST"]))
            else:
                tier = T_RESCUE if (decaying or units >= 4) else T_HARVEST
                tasks.append(mk(tier, units * price, x, y, ["HARVEST"]))

        if day >= LAST_DAY:
            continue

        # ---- watering ----
        if not watered:
            if unwatered >= 1:
                # plant dies tonight if we skip this
                remaining = units if age >= info["maxday"] else \
                    (info.get("yield_water", 4) if not info["ongoing"] else 2)
                tasks.append(mk(T_SURVIVE, max(1.0, remaining * price),
                                x, y, ["WATER"]))
            elif not info["ongoing"] and info["bonus_start"] - 1 <= age <= info["maxday"]:
                gain = 2.0 if fert_until >= day else 1.0
                tasks.append(mk(T_WATER, gain * price, x, y, ["WATER"]))
            elif info["ongoing"] and age in info["prod"] and fert_until >= day:
                tasks.append(mk(T_WATER, price, x, y, ["WATER"]))
            elif SAFETY_WATER:
                # not required today, but doing it now means a missed turn
                # tomorrow is survivable instead of fatal
                stake = info.get("yield_water", 3) if not info["ongoing"] else 2
                tasks.append(mk(T_WATER_SAFE, 0.15 * stake * price, x, y, ["WATER"]))

        # ---- fertilizing ----
        have_fert = int(_num(farm.shed.get("FERTILIZER", 0)))
        if have_fert > 0 and fert_until < day:
            if info["ongoing"]:
                # +1 unit per scheduled production covered (fert lasts 3 days)
                covered = [a for a in info["prod"] if age <= a <= age + 2
                           and day + (a - age) <= LAST_DAY]
                if covered and age <= max(info["prod"]):
                    tasks.append(mk(T_FERT, len(covered) * price, x, y,
                                    ["FERTILIZE"], need=("FERTILIZER", 3)))
            elif have_fert > 40 and age <= info["bonus_start"]:
                gain = info["yield_fert"] - info["yield_water"]
                if gain > 0 and day + info["maxday"] <= LAST_DAY:
                    tasks.append(mk(T_FERT, gain * price, x, y,
                                    ["FERTILIZE"], need=("FERTILIZER", 3)))

        # ---- clear a spent plant ----
        if units == 0 and decaying:
            tasks.append(mk(T_DIG, 5.0, x, y, ["DIG"]))


def animal_tasks(farm, tasks):
    day = farm.day
    fert_price = farm.price("FERTILIZER")
    for (x, y, tile) in farm.structs:
        animal = tile.get("animal")
        if not animal:
            continue
        info = ANIMALS.get(animal)
        if info is None:
            continue
        price = farm.price(info["product"])
        age = day - int(_num(tile.get("placed_day", day)))
        units = int(_num(tile.get("yield_units", 0)))
        fed = bool(tile.get("fed_today", False))
        cared = bool(tile.get("cared_today", False))
        unfed = int(_num(tile.get("consecutive_unfed", 0)))
        banked = int(_num(tile.get("pending_care_bonus", 0)))
        dung = bool(tile.get("fertilizer_available", False))
        cap = info["max_held"]

        # ---- harvest ----
        if units > 0:
            incoming = 1 + banked
            if units + incoming > cap or day >= LAST_DAY:
                tasks.append(mk(T_RESCUE, units * price, x, y, ["HARVEST"]))
            elif units >= 2:
                tasks.append(mk(T_HARVEST, units * price, x, y, ["HARVEST"]))
            else:
                tasks.append(mk(T_HARVEST - 8, units * price, x, y, ["HARVEST"]))

        # ---- fertilizer ----
        if dung:
            tasks.append(mk(T_DUNG, fert_price, x, y, ["COLLECT_FERTILIZER"]))

        if day >= LAST_DAY:
            continue

        # ---- feed ----
        if not fed:
            if unfed >= 1:
                tasks.append(mk(T_SURVIVE, info["cost"] + 6 * price, x, y,
                                ["FEED"], need=("WHEAT", 5)))
            else:
                # missing a feed day forfeits the banked care bonus
                tasks.append(mk(T_FEED, price * (1 + banked * 0.6), x, y,
                                ["FEED"], need=("WHEAT", 5)))

        # ---- care (the single most valuable repeatable action) ----
        if not cared:
            nxt = next_prod_age(info, age, strictly_after=True)
            pays_out = (day - age) + nxt <= LAST_DAY
            room = units + 1 + banked < cap + info["interval"]
            if pays_out and room:
                tasks.append(mk(T_CARE, price, x, y, ["CARE"]))


def build_and_plant_tasks(farm, tasks, plan):
    seeds = dict((k, int(_num(v, 0))) for k, v in farm.seeds.items())
    # a fresh seed starts at consecutive_unwatered = 1, so it must be watered
    # the same day or it weeds overnight -- leave room to walk over and water
    too_late = farm.hour >= TURNS_PER_DAY - 4
    for (x, y, what) in plan:
        if what in ("BUILD_COOP", "BUILD_PASTURE"):
            tasks.append(mk(T_BUILD, 300.0, x, y, [what]))
        elif what in CROPS and not too_late:
            if seeds.get(what, 0) > 0:
                seeds[what] -= 1
                info = CROPS[what]
                val = crop_score(farm, what) * info["maxday"]
                tasks.append(mk(T_PLANT, max(5.0, val), x, y, ["PLANT", what]))


def place_tasks(farm, tasks):
    """Move purchased animals from the shed onto empty structures."""
    if not farm.spare_animals:
        return
    pool = dict(farm.spare_animals)
    for kind in ("PASTURE", "COOP"):
        for (x, y) in farm.free_struct.get(kind, []):
            for animal in STRUCT_ANIMALS[kind]:
                if pool.get(animal, 0) > 0:
                    pool[animal] -= 1
                    price = farm.price(ANIMALS[animal]["product"])
                    tasks.append(mk(T_PLACE, ANIMALS[animal]["cost"] + 10 * price,
                                    x, y, ["PLACE", animal],
                                    need=(animal, 1)))
                    break


def weed_tasks(farm, tasks):
    if farm.day >= LAST_DAY:
        return
    for (x, y) in farm.weeds:
        tasks.append(mk(T_DIG, 30.0, x, y, ["DIG"]))


def mk(tier, value, x, y, op, need=None):
    return {"tier": tier, "value": float(value), "x": x, "y": y,
            "op": list(op), "need": need}


def build_tasks(farm, plan):
    tasks = []
    plant_tasks(farm, tasks)
    animal_tasks(farm, tasks)
    place_tasks(farm, tasks)
    build_and_plant_tasks(farm, tasks, plan)
    weed_tasks(farm, tasks)
    tasks.sort(key=lambda t: (-t["tier"], -t["value"], t["y"], t["x"]))
    return tasks


# ----------------------------------------------------------------------------
# Assigning units to tasks
# ----------------------------------------------------------------------------

def total_item(farm, item):
    n = int(_num(farm.shed.get(item, 0)))
    for inv in farm.inventories:
        n += int(_num(inv.get(item, 0)))
    return n


def assign(farm, tasks):
    out, claimed = {}, set()
    free = set(range(len(farm.units)))
    for relaxed in (False, True):
        if not free:
            break
        _assign_pass(farm, tasks, free, claimed, out, relaxed)
    return out


def _assign_pass(farm, tasks, free, claimed, out, relaxed):
    wheat_pool = total_item(farm, "WHEAT")
    fert_pool = total_item(farm, "FERTILIZER")

    for t in tasks:
        if not free:
            break
        key = (t["x"], t["y"], t["op"][0])
        if key in claimed:
            continue
        need = t["need"]
        if need:
            if need[0] == "WHEAT" and wheat_pool <= 0:
                continue
            if need[0] == "FERTILIZER" and fert_pool <= 0:
                continue

        best, best_key = None, None
        for u in free:
            ux, uy = farm.units[u]
            lacks = 0
            d = dist(ux, uy, t["x"], t["y"])
            if need and int(_num(farm.inventories[u].get(need[0], 0))) <= 0:
                sx, sy = farm.nearest_shed(ux, uy)
                d = dist(ux, uy, sx, sy) + dist(sx, sy, t["x"], t["y"])
                lacks = 1
            k = (d, lacks, u)
            if best_key is None or k < best_key:
                best, best_key = u, k
        if best is None:
            break
        if best_key[0] > cap_for(t["tier"], relaxed):
            continue
        free.discard(best)
        claimed.add(key)
        out[best] = t
        if need:
            if need[0] == "WHEAT":
                wheat_pool -= 1
            elif need[0] == "FERTILIZER":
                fert_pool -= 1
    return out


# how far a unit will walk for a task of a given tier
TRAVEL_CAP = {T_SURVIVE: 99, T_RESCUE: 99, T_PLACE: 99, T_FEED: 7, T_FERT: 5,
              T_WATER: 6, T_WATER_SAFE: 3, T_CARE: 6, T_BUILD: 6, T_HARVEST: 5,
              T_PLANT: 5,
              T_DUNG: 4, T_DIG: 4}


def cap_for(tier, relaxed):
    if relaxed:
        return 99
    return TRAVEL_CAP.get(tier, 5)


def carried_value(farm, inv):
    """Coin value of what a unit is carrying (ignores feed/seed stock)."""
    total = 0.0
    for k, v in inv.items():
        n = int(_num(v, 0))
        if n <= 0 or k in ANIMALS:
            continue
        if k in ("WHEAT", "FERTILIZER"):
            continue          # these are working stock, not cargo
        total += n * farm.price(k)
    return total


def unit_action(farm, u, task, st):
    """Turn an assigned task into a concrete op for this unit."""
    x, y = farm.units[u]
    inv = farm.inventories[u]
    carried = sum(int(_num(v, 0)) for v in inv.values())
    value = carried_value(farm, inv)
    dumping = (farm.day >= LAST_DAY and farm.hour >= TURNS_PER_DAY - 12)

    # Nothing in a pocket can be sold, and everything in a pocket risks being
    # discarded at the end-of-day shed drop. Ferry it back promptly.
    if carried >= 7 or value >= 350 or (dumping and carried > 0):
        sx, sy = farm.nearest_shed(x, y)
        if (x, y) == (sx, sy):
            return ["DROP"]
        mv = move_toward(x, y, sx, sy, st)
        if mv:
            return [mv]

    if task is not None:
        need = task["need"]
        if need:
            item, qty = need
            if int(_num(inv.get(item, 0))) <= 0:
                sx, sy = farm.nearest_shed(x, y)
                if (x, y) == (sx, sy):
                    avail = int(_num(farm.shed.get(item, 0)))
                    if avail > 0:
                        return ["PICKUP", item, int(min(qty, avail))]
                else:
                    mv = move_toward(x, y, sx, sy, st)
                    if mv:
                        return [mv]
        else:
            if (x, y) != (task["x"], task["y"]):
                mv = move_toward(x, y, task["x"], task["y"], st)
                if mv:
                    return [mv]
            else:
                return task["op"]
        if (x, y) == (task["x"], task["y"]):
            return task["op"]
        mv = move_toward(x, y, task["x"], task["y"], st)
        if mv:
            return [mv]

    # idle: ferry produce back so it can be sold, otherwise wait near the shed
    if carried >= 2 or value > 0:
        sx, sy = farm.nearest_shed(x, y)
        if (x, y) == (sx, sy):
            return ["DROP"]
        mv = move_toward(x, y, sx, sy, st)
        if mv:
            return [mv]
    return ["PASS"]


# ----------------------------------------------------------------------------
# Market
# ----------------------------------------------------------------------------

SELL_ORDER = ["MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "EGG",
              "FERTILIZER", "CARROT", "WHEAT"]


def hands_target(farm):
    workload = len(farm.plants) + len(farm.empty)
    need = workload / 9.0 + farm.n_animals / 2.0 + 1.0
    k = int(round(need))
    budget = max(60.0, farm.money * 0.05)
    while k > 2 and fib_sum(k) > budget:
        k -= 1
    if farm.day >= LAST_DAY:
        k = min(k, 6)
    return max(2, min(MAX_HANDS, k))


def sell_floor_fraction(farm):
    """How far below base price we are willing to sell today."""
    day = farm.day
    if day < 20:
        frac = 0.55
    else:
        frac = 0.55 * max(0.0, (LAST_DAY - day) / float(LAST_DAY - 20 + 1))
    held = farm.shed_total()
    if held > 60:
        frac *= 0.5
    if held > 85 or day >= LAST_DAY:
        frac = 0.0
    return frac


def market_orders(farm, want_seeds):
    orders = []
    money = farm.money
    day, hour = farm.day, farm.hour
    n_animals = farm.n_animals + sum(farm.spare_animals.values())
    reserve = max(350, 150 + 25 * n_animals)

    # --- 1. hire hands (they reset every day; cheapest big lever in the game)
    if hour <= 1 and day < LAST_DAY:
        todo = max(0, hands_target(farm) - farm.hires_today)
        for i in range(min(todo, 10)):
            idx = min(len(FIB) - 1, farm.hires_today + i)
            cost = FIB[idx]
            if money - cost < 20:
                break
            orders.append(["HIRE"])
            money -= cost

    # --- 2. keep wheat on hand so no animal ever starves
    if n_animals > 0 and day < LAST_DAY - 1:
        stock = total_item(farm, "WHEAT")
        want = n_animals * 3 + 2
        if stock < want:
            wprice = farm.price("WHEAT")
            if wprice <= 95:
                qty = int(min(want - stock, max(0, (money - 100) // max(1, wprice))))
                if qty > 0:
                    orders.append(["BUY_PRODUCT", "WHEAT", int(qty)])
                    money -= qty * wprice

    # --- 3. sell produce, paced so we don't crash our own prices
    frac = sell_floor_fraction(farm)
    held_total = farm.shed_total()
    days_left = max(0, LAST_DAY - day)
    wheat_reserve = n_animals * min(3, days_left) if n_animals else 0
    fert_reserve = 0
    if days_left > 3 and any(CROPS[c]["ongoing"] for c in farm.crop_counts
                             if c in CROPS):
        fert_reserve = 12
    sells = 0
    for product in SELL_ORDER:
        if sells >= 6:
            break
        have = int(_num(farm.shed.get(product, 0)))
        if product == "WHEAT":
            have -= wheat_reserve
        elif product == "FERTILIZER":
            have -= fert_reserve
        if have <= 0:
            continue
        base = MARKET_PARAMS.get(product, (1.0,))[0]
        inv = farm.inv_of(product)
        n = sellable_count(product, inv, have, base * frac, cap=40)
        if n <= 0:
            if held_total > 85 or day >= LAST_DAY:
                n = have
            elif have > 25:
                # slow bleed rather than being stuck with dead stock at the end
                n = max(1, have // max(1, LAST_DAY - day + 1))
        if n > 0:
            orders.append(["SELL", product, int(n)])
            sells += 1

    if day >= LAST_DAY:
        return orders[:10]

    # --- 4. land: it gates everything else, so take it as soon as affordable
    n_quads = len(farm.unlocked_quads)
    if n_quads < MAX_QUADS and day <= 22:
        cost = LAND_COSTS[min(len(LAND_COSTS) - 1, n_quads - 1)]
        if money >= cost + reserve + 400:
            orders.append(["BUY_LAND"])
            money -= cost

    # --- 5. livestock, best value first, only where a home exists.
    # A cared-for sheep or cow returns its purchase price in ~2 days, so this
    # outranks seed spending.
    save_for_animal = 0
    if day <= 22:
        tgt = target_animals(farm)
        free_p = len(farm.free_struct["PASTURE"]) - (
            farm.spare_animals.get("SHEEP", 0) + farm.spare_animals.get("COW", 0))
        free_c = len(farm.free_struct["COOP"]) - farm.spare_animals.get("GOOSE", 0)
        for animal in buy_order(day):
            info = ANIMALS[animal]
            owned = (farm.animal_counts.get(animal, 0)
                     + farm.spare_animals.get(animal, 0))
            if owned >= tgt.get(animal, 0):
                continue
            slots = free_c if info["struct"] == "COOP" else free_p
            if slots <= 0:
                continue
            qty = int(min(slots, tgt[animal] - owned,
                          max(0, (money - reserve - 200)) // info["cost"]))
            if qty > 0:
                orders.append(["BUY_ANIMAL", animal, int(qty)])
                money -= qty * info["cost"]
                if info["struct"] == "COOP":
                    free_c -= qty
                else:
                    free_p -= qty
            elif not save_for_animal and money > info["cost"] * 0.55:
                save_for_animal = info["cost"]

    # --- 6. seeds for the tiles we intend to plant
    seed_orders = 0
    seed_reserve = reserve + save_for_animal
    for crop, want in sorted(want_seeds.items(),
                             key=lambda kv: -crop_score(farm, kv[0])):
        if seed_orders >= 3:
            break
        have = int(_num(farm.seeds.get(crop, 0)))
        short = want - have
        if short <= 0:
            continue
        cost = CROPS[crop]["seed"]
        afford = int(max(0, (money - seed_reserve)) // cost)
        qty = min(short, afford, 30)
        if qty > 0:
            orders.append(["BUY_SEED", crop, int(qty)])
            money -= qty * cost
            seed_orders += 1

    return orders[:10]


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def _decide(obs, config):
    farm = Farm(obs)
    st = compass(farm.player)
    if farm.hour > 0:
        _calibrate(st, farm.step, farm.units[0][0], farm.units[0][1])
    else:
        st["probe"] = None

    plan, want_seeds = plan_empty_tiles(farm)
    tasks = build_tasks(farm, plan)
    assigned = assign(farm, tasks)

    ops = []
    for u in range(len(farm.units)):
        try:
            ops.append(unit_action(farm, u, assigned.get(u), st))
        except Exception:
            ops.append(["PASS"])

    farmer_op = ops[0] if ops else ["PASS"]
    if farmer_op and farmer_op[0] in ("NORTH", "SOUTH", "EAST", "WEST"):
        st["probe"] = {"step": farm.step, "dir": farmer_op[0],
                       "x": farm.units[0][0], "y": farm.units[0][1]}

    try:
        market = market_orders(farm, want_seeds)
    except Exception:
        market = []

    return {"farmer": farmer_op,
            "hands": ops[1:],
            "market": market}


def agent(obs, config=None):
    try:
        return _decide(obs, config)
    except Exception:
        n = 0
        try:
            n = len(_get(_get(obs, "farms")[int(_num(_get(obs, "player", 0)))],
                         "hands", []) or [])
        except Exception:
            n = 0
        return {"farmer": ["PASS"],
                "hands": [["PASS"] for _ in range(n)],
                "market": []}

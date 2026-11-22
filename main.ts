// main.ts -- Sokoban solver: room-by-room push-optimal A*.
//
// Every level this generator draws is a hallway with independent rooms off
// it: a box pushed out of a door into the hallway can never be pushed back,
// so each room is its own little puzzle and the level's push optimum is the
// sum of the rooms' optima. Each room is solved on its own with A* over
// pushes: a state is the boxes plus the player's reachable region (named by
// its lowest square), the cost is pushes so far, and the heuristic is the
// cheapest assignment of boxes to goals by walking distance (a lower bound,
// so the first solved state popped is push-optimal). Two deadlocks are
// pruned: a dead square, from which no sequence of pushes can reach a goal,
// and a frozen 2x2 block of walls and boxes. The search budget is counted
// in expansions, never the clock, so two runs write identical bytes.
//
// Usage: MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=<dir> node main.ts
// Reads <input>/<id>/level.txt (XSB), writes <output>/<id>/solution.txt
// (moves in LURD: a lower letter is a step, an upper letter a push).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// [dx, dy, step letter, push letter]
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// Work budgets (expansions), never the clock: deterministic by construction.
const ASTAR_POPS = 200_000;
const BFS_POPS = 250_000;

type Level = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(lv: Level): Uint8Array {
  const live = new Uint8Array(lv.w * lv.h);
  const queue = [...lv.goals];
  for (const g of lv.goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * lv.w + dx;
      const y = x - d;
      if (!lv.wall[y] && !lv.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Walking distances from `from` over open floor, ignoring boxes (-1 unreachable). */
function distFrom(lv: Level, from: number): Int32Array {
  const dist = new Int32Array(lv.w * lv.h).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * lv.w + dx;
      if (!lv.wall[q] && dist[q] < 0) {
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/**
 * The squares the player reaches from `from` with `occupied` in the way,
 * marked in `seen` with `stamp`; returns the region's lowest square.
 * `stack` is scratch space of at least w*h.
 */
function regionLow(
  lv: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const [dx, dy] of DIRS) {
      const q = p + dy * lv.w + dx;
      if (lv.wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(lv: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const w = lv.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(lv.wall[a] || occupied[a])) continue;
    if (!(lv.wall[b] || occupied[b])) continue;
    if (!(lv.wall[c] || occupied[c])) continue;
    if (!(lv.wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) {
      return true;
    }
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkPath(lv: Level, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(lv.w * lv.h).fill(-1);
  const how = new Array<string>(lv.w * lv.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * lv.w + dx;
      if (lv.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkPath: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

type HeapEntry = { f: number; g: number; seq: number; idx: number };

/** A binary heap on (f asc, g desc, seq asc): the smallest f, then the deepest, then the earliest. */
class Heap {
  a: HeapEntry[] = [];
  get size(): number {
    return this.a.length;
  }
  push(e: HeapEntry): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i]!, a[p]!)) break;
      [a[i], a[p]] = [a[p]!, a[i]!];
      i = p;
    }
  }
  pop(): HeapEntry {
    const a = this.a;
    const top = a[0]!;
    const last = a.pop()!;
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1]!, a[c]!)) c++;
        if (!Heap.less(a[c]!, a[i]!)) break;
        [a[i], a[c]] = [a[c]!, a[i]!];
        i = c;
      }
    }
    return top;
  }
  static less(x: HeapEntry, y: HeapEntry): boolean {
    return x.f < y.f || (x.f === y.f && (x.g > y.g || (x.g === y.g && x.seq < y.seq)));
  }
}

/**
 * Push-optimal solution of `lv` within `maxPops` expansions: {pushes, moves},
 * or null when the search gives up (or the level has no solution).
 * With `heuristic` true the A* estimate is the cheapest assignment of boxes
 * to goals by walking distance (admissible, so the first solved state popped
 * is optimal); false is plain breadth-first search over pushes.
 */
function solvePushes(lv: Level, maxPops: number, heuristic: boolean): { pushes: number; moves: string } | null {
  const { w, h } = lv;
  const size = w * h;
  if (lv.boxes.length !== lv.goals.length || lv.player < 0) return null;
  const goal = new Uint8Array(size);
  for (const g of lv.goals) goal[g] = 1;
  const live = liveSquares(lv);
  if (lv.boxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (solved(lv.boxes)) return { pushes: 0, moves: '' };

  // Heuristic: cheapest assignment of boxes to goals (exact for few boxes,
  // sum of nearest distances otherwise); Infinity when some box is stranded.
  const n = lv.boxes.length;
  const goalDist = lv.goals.map((g) => distFrom(lv, g));
  const estimate = (boxes: number[]): number => {
    if (!heuristic) return 0;
    if (n <= 8 && n === lv.goals.length) {
      const m = lv.goals.length;
      const dp = new Array<number>(1 << m).fill(Infinity);
      dp[0] = 0;
      for (let mask = 0; mask < (1 << m); mask++) {
        const i = popcount(mask);
        if (i >= n || dp[mask] === Infinity) continue;
        for (let j = 0; j < m; j++) {
          if (mask & (1 << j)) continue;
          const d = goalDist[j]![boxes[i]!]!;
          if (d < 0) continue;
          const nm = mask | (1 << j);
          if (dp[mask]! + d < dp[nm]!) dp[nm] = dp[mask]! + d;
        }
      }
      return dp[(1 << m) - 1]!;
    }
    let s = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let j = 0; j < lv.goals.length; j++) {
        const d = goalDist[j]![b]!;
        if (d >= 0 && d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      s += best;
    }
    return s;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size); // this state's player region
  const mark = new Int32Array(size); // a successor's player region
  const stack = new Int32Array(size);
  let stamp = 0;

  const keyOf = (boxes: number[], low: number): string => low + ':' + boxes.join(',');
  const best = new Map<string, number>(); // state key -> best pushes so far
  const sBoxes: number[][] = [lv.boxes.slice()];
  const sPlayer: number[] = [lv.player];
  const sParent: number[] = [-1];
  const sG: number[] = [0];
  const sLow: number[] = [0];
  for (const b of lv.boxes) occupied[b] = 1;
  sLow[0] = regionLow(lv, occupied, lv.player, seen, ++stamp, stack);
  for (const b of lv.boxes) occupied[b] = 0;
  const h0 = estimate(lv.boxes);
  if (h0 === Infinity) return null;
  best.set(keyOf(lv.boxes, sLow[0]!), 0);
  const heap = new Heap();
  let seq = 0;
  heap.push({ f: h0, g: 0, seq: seq++, idx: 0 });

  let pops = 0;
  let found = -1;
  while (heap.size > 0) {
    const cur = heap.pop();
    const idx = cur.idx;
    if (cur.g !== best.get(keyOf(sBoxes[idx]!, sLow[idx]!))) continue; // stale entry
    if (solved(sBoxes[idx]!)) {
      found = idx;
      break;
    }
    if (++pops > maxPops) return null;
    const boxes = sBoxes[idx]!;
    for (const b of boxes) occupied[b] = 1;
    regionLow(lv, occupied, sPlayer[idx]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || lv.wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(lv, occupied, goal, t);
        let next: number[] | null = null;
        let nlow = -1;
        let nh = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          nlow = regionLow(lv, occupied, b, mark, ++stamp, stack);
          nh = estimate(next);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null || nh === Infinity) continue;
        const ng = sG[idx]! + 1;
        const nkey = keyOf(next, nlow);
        const prevBest = best.get(nkey);
        if (prevBest !== undefined && prevBest <= ng) continue;
        best.set(nkey, ng);
        const nidx = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(idx);
        sG.push(ng);
        sLow.push(nlow);
        heap.push({ f: ng + nh, g: ng, seq: seq++, idx: nidx });
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes.slice();
  for (const s of path) {
    const nextBoxes = sBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkPath(lv, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

function popcount(x: number): number {
  let c = 0;
  while (x > 0) {
    c += x & 1;
    x >>= 1;
  }
  return c;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway:
 * a box there reaches no goal) out of the floor, and each part left that
 * holds a box is a room. Null when some part holds more or fewer boxes than
 * goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(lv: Level): { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] | null {
  const live = liveSquares(lv);
  const row = Math.floor(lv.player / lv.w);
  const cut = new Uint8Array(lv.w * lv.h);
  for (let i = row * lv.w; i < (row + 1) * lv.w; i++) if (!lv.wall[i] && !live[i]) cut[i] = 1;
  if (lv.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(lv.w * lv.h).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < lv.w * lv.h; s++) {
    if (lv.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * lv.w + dx;
        if (!lv.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of lv.boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of lv.goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (lv.goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn,
 * the other rooms' boxes standing as walls (on their goals once their room
 * is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's.
 * Falls back to a search over the whole level when it is no set of rooms,
 * and to plain breadth-first search when A* gives up on a room.
 */
function solveLevel(lv: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(lv);
  if (rooms === null) {
    return solvePushes(lv, ASTAR_POPS, true) ?? solvePushes(lv, BFS_POPS, false);
  }
  let player = lv.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = lv.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: lv.w,
      h: lv.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushes(sub, ASTAR_POPS, true) ?? solvePushes(sub, BFS_POPS, false);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(lv.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

// Every instance gets an answer file, even one the search gives up on
// (an empty answer is valid, worth no points, and keeps the entry passing).
for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(INPUT, id, 'level.txt'), 'utf8');
    const solved = solveLevel(parse(text));
    if (solved !== null) moves = solved.moves;
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}

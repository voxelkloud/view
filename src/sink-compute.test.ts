import { describe, expect, it } from "vitest";
import {
  BlockAllocator,
  DEAD_META,
  DEAD_SLOT,
  MAX_SLOTS,
  SlotPool,
  buildVisibleBlocks,
  packNodeMeta,
} from "./sink-compute.js";

/**
 * The allocator behind the compute sink. These are the cases that corrupt
 * silently rather than throw: a free run that fails to merge fragments the
 * arena until an allocation that SHOULD fit does not, and a wrong `start` hands
 * out a range overlapping a live node — neither of which any frame counter
 * would report.
 */
describe("BlockAllocator", () => {
  it("bumps while there is room and refuses past the cap", () => {
    const a = new BlockAllocator(100);
    expect(a.allocate(40)).toBe(0);
    expect(a.allocate(40)).toBe(40);
    expect(a.used).toBe(80);
    expect(a.allocate(40)).toBe(-1);
    expect(a.used).toBe(80);
  });

  it("reuses a freed run rather than growing", () => {
    const a = new BlockAllocator(100);
    a.allocate(30);
    const mid = a.allocate(30);
    a.allocate(30);
    a.release(mid, 30);
    expect(a.freePoints).toBe(30);
    expect(a.allocate(30)).toBe(mid);
    expect(a.used).toBe(90);
    expect(a.freeRunCount).toBe(0);
  });

  it("splits a run larger than the request and keeps the remainder", () => {
    const a = new BlockAllocator(100);
    a.allocate(10);
    const mid = a.allocate(50);
    a.allocate(10);
    a.release(mid, 50);
    expect(a.allocate(20)).toBe(mid);
    expect(a.freePoints).toBe(30);
    expect(a.allocate(30)).toBe(mid + 20);
    expect(a.freeRunCount).toBe(0);
  });

  it("coalesces with the run before, the run after, and both at once", () => {
    const a = new BlockAllocator(100);
    for (let i = 0; i < 4; i++) a.allocate(10);
    a.release(0, 10);
    a.release(10, 10); // merges FORWARD into the one before
    expect(a.freeRunCount).toBe(1);
    expect(a.freePoints).toBe(20);

    // FIVE blocks, not four, so the pair released below is INTERIOR. With four
    // the second release would reach the high-water mark and be given back to
    // the bump pointer instead of coalesced, which is a different behaviour
    // and has its own test.
    const b = new BlockAllocator(100);
    for (let i = 0; i < 5; i++) b.allocate(10);
    b.release(30, 10);
    b.release(20, 10); // merges BACKWARD into the one after
    expect(b.freeRunCount).toBe(1);
    expect(b.freePoints).toBe(20);

    const c = new BlockAllocator(100);
    for (let i = 0; i < 5; i++) c.allocate(10);
    c.release(0, 10);
    c.release(20, 10);
    expect(c.freeRunCount).toBe(2);
    c.release(10, 10); // fills the hole between them: three runs become one
    expect(c.freeRunCount).toBe(1);
    expect(c.freePoints).toBe(30);
    // And the merged run is usable AS ONE, which is the whole point of merging.
    expect(c.allocate(30)).toBe(0);
    expect(c.freeRunCount).toBe(0);
  });

  it("gives the tail back to the bump pointer instead of to the free list", () => {
    // WHY THIS MATTERS AND `freePoints` DOES NOT SAY IT: `used` is what a frame
    // dispatches over — one GPU thread per slot below it, twice a frame,
    // whether or not a live node is there. A high-water that only rose made a
    // frame cost the session's PEAK residency for ever, so orbiting away from
    // a region and never coming back still paid for it.
    const a = new BlockAllocator(100);
    const first = a.allocate(10);
    const second = a.allocate(10);
    const third = a.allocate(10);
    expect(a.used).toBe(30);

    a.release(third, 10);
    expect(a.used).toBe(20);
    expect(a.freeRunCount).toBe(0);
    expect(a.livePoints).toBe(20);

    // Coalesce first, THEN trim: releasing the new tail must take the run
    // behind it with it, not leave a free run adjacent to the high-water mark.
    a.release(second, 10);
    expect(a.used).toBe(10);
    expect(a.freeRunCount).toBe(0);
    expect(a.livePoints).toBe(10);

    a.release(first, 10);
    expect(a.used).toBe(0);
    expect(a.livePoints).toBe(0);

    // A hole in the middle stays a hole: reclaiming it would mean moving a
    // live range, and every block index that addresses it would be wrong.
    const b = new BlockAllocator(100);
    b.allocate(10);
    const hole = b.allocate(10);
    b.allocate(10);
    b.release(hole, 10);
    expect(b.used).toBe(30);
    expect(b.freeRunCount).toBe(1);
    expect(b.livePoints).toBe(20);
  });

  it("never hands out a range that overlaps a live one", () => {
    // Deterministic churn: allocate a varying size each round, retire the
    // oldest every third round, and assert after every allocation that the new
    // range intersects no live one. Overlap is the failure that would silently
    // draw one node's points with another node's colour.
    const a = new BlockAllocator(10_000);
    const live: { start: number; end: number }[] = [];
    const sizes = [7, 13, 5, 21, 3, 11, 17, 2];
    for (let round = 0; round < 200; round++) {
      const n = sizes[round % sizes.length]!;
      const start = a.allocate(n);
      expect(start).toBeGreaterThanOrEqual(0);
      for (const r of live) {
        expect(start >= r.end || start + n <= r.start).toBe(true);
      }
      live.push({ start, end: start + n });
      // Steady state after a short warm-up: retire one per round, so the live
      // set stops growing and the high-water mark is a statement about REUSE
      // rather than about how fast the test allocates.
      if (live.length > 20) {
        const victim = live.shift()!;
        a.release(victim.start, victim.end - victim.start);
      }
    }
    // At most 20 live blocks of at most 21 points is 420 live. A free list that
    // reuses keeps the high-water near that; one that leaks marches toward the
    // ~2000 points these 200 rounds allocated in total. The bound is 3x live,
    // which is loose enough for first-fit fragmentation and tight enough that a
    // leak fails it.
    expect(a.used).toBeLessThan(3 * 20 * 21);
  });

  it("raises the ceiling only upward", () => {
    const a = new BlockAllocator(50);
    expect(a.allocate(60)).toBe(-1);
    a.setCapacity(100);
    expect(a.allocate(60)).toBe(0);
    a.setCapacity(10);
    expect(a.capacity).toBe(100);
  });
});

/**
 * O empacotamento do `nmeta`. Estes são os erros que não lançam nada: uma
 * classe que invade o slot faz o ponto consultar a liveness de OUTRO nó, e o
 * sintoma é um nó que some ou pisca — nunca uma exceção, nunca um contador
 * errado. O shader lê estes mesmos bits em `slotOf` e `classOff`.
 */
describe("packNodeMeta", () => {
  /** O que o WGSL faz: `(nmeta[i] >> 8u) & 0xffffu`. */
  const slotOf = (v: number) => (v >>> 8) & 0xffff;
  const levelOf = (v: number) => v & 0xff;
  const classOf = (v: number) => v >>> 24;

  it("mantém nível, slot e classe sem se pisarem no extremo", () => {
    // MAX_SLOTS - 1 com a classe mais alta: se o slot ainda ocupasse 24 bits,
    // a classe entraria nele e este é o caso em que se veria.
    const meta = packNodeMeta(255, 65_535, 1, [255]);
    expect(levelOf(meta[0]!)).toBe(255);
    expect(slotOf(meta[0]!)).toBe(65_535);
    expect(classOf(meta[0]!)).toBe(255);
  });

  it("carrega a classe de cada ponto, não a do nó", () => {
    const meta = packNodeMeta(3, 7, 4, [2, 6, 2, 18]);
    expect([...meta].map(classOf)).toEqual([2, 6, 2, 18]);
    expect([...meta].map(slotOf)).toEqual([7, 7, 7, 7]);
    expect([...meta].map(levelOf)).toEqual([3, 3, 3, 3]);
  });

  it("deixa a classe em zero quando a nuvem não tem o atributo", () => {
    const meta = packNodeMeta(1, 2, 3, undefined);
    expect([...meta].map(classOf)).toEqual([0, 0, 0]);
    expect([...meta].map(slotOf)).toEqual([2, 2, 2]);
  });

  it("manda código fora de faixa para o balde 255, não para a classe 0", () => {
    // 0 é uma classe REAL ("created, never classified"); mandar lixo para lá
    // faria pontos corrompidos desaparecerem junto com pontos legítimos.
    const meta = packNodeMeta(0, 0, 3, [-1, 300, 1e9]);
    expect([...meta].map(classOf)).toEqual([255, 255, 255]);
  });

  it("reserva exactamente 16 bits ao slot", () => {
    // A trava da invariante. Subir MAX_SLOTS acima disto faz `slot & 0xffff`
    // dobrar dois nós no mesmo slot: cada um passa a ler a liveness do outro,
    // e o sintoma é um nó que pisca — nunca um erro, nunca um contador errado.
    expect(MAX_SLOTS).toBeLessThanOrEqual(0x1_0000);
  });

  it("registra só as classes que viu", () => {
    const present = new Uint8Array(256);
    packNodeMeta(0, 0, 4, [2, 6, 2, 2], present);
    expect([...present.keys()].filter((c) => present[c] === 1)).toEqual([2, 6]);
  });
});

/**
 * O pool de slots do sink de compute. O caso que corrompe em silêncio é o
 * reuso: um slot devolvido e emprestado outra vez sem o `nmeta` do bloco velho
 * ter sido carimbado faz pontos de um nó desanexado acenderem-se com a
 * liveness do nó novo — nenhum contador de frame diz nada. Os testes aqui
 * trancam a aritmética; o carimbo em si vive no `detach`.
 */
describe("SlotPool", () => {
  it("empresta em sequência enquanto há capacidade e recusa depois", () => {
    const p = new SlotPool(3);
    expect(p.acquire()).toBe(0);
    expect(p.acquire()).toBe(1);
    expect(p.acquire()).toBe(2);
    expect(p.acquire()).toBe(-1);
    expect(p.used).toBe(3);
  });

  it("reaproveita o devolvido em vez de subir a marca de água", () => {
    const p = new SlotPool(3);
    p.acquire();
    const mid = p.acquire();
    p.acquire();
    p.release(mid);
    expect(p.freeCount).toBe(1);
    expect(p.acquire()).toBe(mid);
    expect(p.highWater).toBe(3);
  });

  it("devolve o mais recente primeiro", () => {
    const p = new SlotPool(8);
    const a = p.acquire();
    const b = p.acquire();
    p.release(a);
    p.release(b);
    expect(p.acquire()).toBe(b);
    expect(p.acquire()).toBe(a);
  });

  it("sobrevive a um churn maior que a capacidade, que é o vazamento que isto fecha", () => {
    const p = new SlotPool(4);
    for (let i = 0; i < 1000; i++) {
      const s = p.acquire();
      expect(s).toBeGreaterThanOrEqual(0);
      p.release(s);
    }
    expect(p.highWater).toBe(1);
  });

  it("ignora uma devolução fora do intervalo em vez de envenenar o pool", () => {
    const p = new SlotPool(2);
    p.release(-1);
    p.release(2);
    expect(p.freeCount).toBe(0);
    expect(p.acquire()).toBe(0);
  });

  it("a marca de água é o que o commit varre, não os emprestados", () => {
    const p = new SlotPool(8);
    p.acquire();
    const b = p.acquire();
    p.release(b);
    expect(p.used).toBe(1);
    expect(p.highWater).toBe(2);
  });

  it("volta ao início no reset", () => {
    const p = new SlotPool(2);
    p.acquire();
    p.acquire();
    p.reset();
    expect(p.acquire()).toBe(0);
    expect(p.highWater).toBe(1);
  });
});

describe("DEAD_SLOT", () => {
  it("fica fora do que o pool empresta, logo nenhum nó pode acendê-lo", () => {
    const p = new SlotPool(DEAD_SLOT);
    let last = -1;
    for (let i = 0; i < DEAD_SLOT; i++) last = p.acquire();
    expect(last).toBe(DEAD_SLOT - 1);
    expect(p.acquire()).toBe(-1);
  });

  it("cabe nos 16 bits que packNodeMeta lhe reserva", () => {
    expect(DEAD_SLOT).toBeLessThan(MAX_SLOTS);
    expect(DEAD_META >>> 8).toBe(DEAD_SLOT);
    // Nível e classe a zero: um ponto órfão não tem nem um nem outro.
    expect(DEAD_META & 0xff).toBe(0);
    expect(DEAD_META >>> 24).toBe(0);
  });

  it("o carimbo é distinto de qualquer nmeta que o attach escreva", () => {
    const real = packNodeMeta(6, 0, 1, undefined)[0]!;
    expect(real).not.toBe(DEAD_META);
  });
});

describe("buildVisibleBlocks", () => {
  const table = (
    blocks: Record<number, { start: number; count: number; level: number }>,
    order: number[],
    capacityRows = 64,
  ) => {
    const out = new Uint32Array(capacityRows * 2);
    const result = buildVisibleBlocks(
      Int32Array.from(order),
      order.length,
      (i) => blocks[i],
      out,
    );
    return { result, out };
  };

  it("writes a running prefix, not the slot order", () => {
    // The nodes are drawn in an order that has nothing to do with where they
    // sit in the arena — which is the normal case, because the draw list is
    // the scheduler's priority order and the arena's is allocation order.
    const { result, out } = table(
      { 7: { start: 900, count: 30, level: 4 }, 3: { start: 10, count: 50, level: 2 } },
      [7, 3],
    );
    expect(result.entries).toBe(2);
    expect(result.points).toBe(80);
    expect(result.deepest).toBe(4);
    expect(Array.from(out.slice(0, 4))).toEqual([900, 0, 10, 30]);
  });

  it("maps every thread back to the slot it came from", () => {
    // THE PROPERTY THE SHADER DEPENDS ON, checked exhaustively rather than at
    // the boundaries: thread t must land on the t-th drawn point, and the
    // binary search only finds it if the prefix column is sorted and exact.
    const blocks = {
      1: { start: 500, count: 3, level: 1 },
      2: { start: 0, count: 1, level: 2 },
      3: { start: 64, count: 7, level: 3 },
    };
    const { result, out } = table(blocks, [1, 2, 3]);
    const expected = [500, 501, 502, 0, 64, 65, 66, 67, 68, 69, 70];
    expect(result.points).toBe(expected.length);

    // The shader's search, in TypeScript, over the table it will be handed.
    const slotForThread = (t: number): number => {
      let lo = 0;
      let hi = result.entries - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (out[mid * 2 + 1]! <= t) lo = mid;
        else hi = mid - 1;
      }
      return out[lo * 2]! + (t - out[lo * 2 + 1]!);
    };
    expect(Array.from({ length: result.points }, (_, t) => slotForThread(t))).toEqual(
      expected,
    );
  });

  it("skips a selected node that has nothing attached yet", () => {
    // The ordinary state of a node still streaming. It must not take a row: a
    // row claiming points that are not resident would map threads onto a
    // neighbour's memory.
    const { result, out } = table(
      {
        1: { start: 0, count: 4, level: 1 },
        3: { start: 40, count: 6, level: 2 },
      },
      [1, 2, 3],
    );
    expect(result.entries).toBe(2);
    expect(result.points).toBe(10);
    expect(Array.from(out.slice(0, 4))).toEqual([0, 0, 40, 4]);
  });

  it("skips a zero-point block rather than emitting an empty row", () => {
    const { result } = table(
      { 1: { start: 0, count: 0, level: 1 }, 2: { start: 8, count: 5, level: 1 } },
      [1, 2],
    );
    expect(result.entries).toBe(1);
    expect(result.points).toBe(5);
  });

  it("reports truncation instead of dropping nodes in silence", () => {
    // The caller falls back to the flag path on this, because the alternative
    // is a cloud with a piece missing and nothing anywhere saying so.
    const blocks: Record<number, { start: number; count: number; level: number }> = {};
    const order: number[] = [];
    for (let i = 0; i < 5; i++) {
      blocks[i] = { start: i * 10, count: 10, level: 1 };
      order.push(i);
    }
    const { result } = table(blocks, order, 3);
    expect(result.truncated).toBe(true);
    expect(result.entries).toBe(3);
  });

  it("is empty when nothing is drawn", () => {
    const { result } = table({}, []);
    expect(result).toEqual({ entries: 0, points: 0, deepest: 0, truncated: false });
  });
});

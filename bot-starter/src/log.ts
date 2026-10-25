// JSONL logging, one file per match plus a "bot" file for everything outside
// a match. Logging must never throw into the caller.

import * as fs from "node:fs";
import * as path from "node:path";

let dir = "logs";
const streams = new Map<string, fs.WriteStream>();

export function initLog(logDir: string) {
  dir = logDir;
  try {
    fs.mkdirSync(dir, { recursive: true });
  } catch {
    /* ignore */
  }
}

function streamFor(name: string): fs.WriteStream | null {
  let s = streams.get(name);
  if (s) return s;
  try {
    s = fs.createWriteStream(path.join(dir, `${name}.jsonl`), { flags: "a" });
    s.on("error", () => {});
    streams.set(name, s);
    return s;
  } catch {
    return null;
  }
}

// BigInt, Identity-like objects and Timestamps all need to survive JSON.
function replacer(_k: string, v: unknown): unknown {
  if (typeof v === "bigint") return v.toString();
  if (v instanceof Uint8Array || v instanceof Float64Array)
    return Array.from(v as ArrayLike<number>);
  if (v instanceof Map) return Object.fromEntries(v);
  if (v && typeof v === "object") {
    const o = v as { toHexString?: () => string; microsSinceUnixEpoch?: bigint };
    if (typeof o.toHexString === "function") return o.toHexString();
    if (o.microsSinceUnixEpoch !== undefined)
      return o.microsSinceUnixEpoch.toString();
  }
  return v;
}

export function logEvent(
  matchId: bigint | string | null,
  type: string,
  data: Record<string, unknown> = {},
) {
  try {
    const s = streamFor(matchId === null ? "bot" : `match-${matchId}`);
    s?.write(JSON.stringify({ t: Date.now(), type, ...data }, replacer) + "\n");
  } catch {
    /* never throw from logging */
  }
}

export function closeMatchLog(matchId: bigint | string) {
  const name = `match-${matchId}`;
  const s = streams.get(name);
  if (s) {
    s.end();
    streams.delete(name);
  }
}

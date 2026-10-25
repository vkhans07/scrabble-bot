import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('native cash-in and cumulative refinement preserve scenario accounting', () => {
  const dir = mkdtempSync(join(tmpdir(), 'scrabble-native-test-'));
  try {
    const bin = join(dir, 'test');
    const words = join(dir, 'words');
    writeFileSync(words, 'AAAAAAA\nZZ\n');
    execFileSync('c++', ['-std=c++17', '-O2', '-pthread', fileURLToPath(new URL('../native/engine.test.cpp', import.meta.url)), '-o', bin]);
    const output = execFileSync(bin, [words], { encoding: 'utf8' });
    assert.match(output, /RES fast 7 test auction 4 /);
    assert.match(output, /RES refine 7 test auction 10 /);
    assert.match(output, /RES full 7 test auction 6 /);
    assert.ok(output.indexOf('RES fast') < output.indexOf('RES refine'));
    assert.ok(output.indexOf('RES refine') < output.indexOf('RES full'));
    const lines = output.split('\n').filter(l => l.startsWith('RES ')).map(l => l.split(' '));
    assert.match(output, /RES fast 8 test auction 0 /);
    assert.match(output, /RES refine 8 test auction 10 /);
    for (const seq of ['7', '8']) {
      const refined = lines.find(l => l[1] === 'refine' && l[2] === seq)!;
      const full = lines.find(l => l[1] === 'full' && l[2] === seq)!;
      assert.equal(full[7], refined[7], 'broad pass preserves coin value');
      assert.equal(full[9 + 25], refined[9 + 25], 'broad pass preserves current letter value');
    }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

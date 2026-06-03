import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { Tiktoken } from 'js-tiktoken/lite';

const repo = 'moonshotai/Kimi-K2.6';
const rev = 'main';
const baseRaw = `https://huggingface.co/${repo}/raw/${rev}`;
const baseResolve = `https://huggingface.co/${repo}/resolve/${rev}`;

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  return await res.text();
}

function getKimiCompatiblePatStr() {
  return [
    String.raw`[\p{Script=Han}]+`,
    String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sStTrReEvVmMlLdD])?`,
    String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sStTrReEvVmMlLdD])?`,
    String.raw`\p{N}{1,3}`,
    String.raw` ?[^\s\p{L}\p{N}]+[\r\n]*`,
    String.raw`\s*[\r\n]+`,
    String.raw`\s+(?!\S)`,
    String.raw`\s+`,
  ].join('|');
}

function buildSpecialTokensFromConfig(cfg) {
  const specialTokens = {};
  for (const [id, tokenInfo] of Object.entries(cfg.added_tokens_decoder || {})) {
    const content = typeof tokenInfo === 'string' ? tokenInfo : tokenInfo.content;
    const isSpecial = typeof tokenInfo === 'string' ? true : tokenInfo.special !== false;
    if (content && isSpecial) specialTokens[content] = Number(id);
  }
  return specialTokens;
}

function tiktokenModelToJsTiktokenRanks(modelText, patStr) {
  const rows = modelText.trim().split(/\r?\n/)
    .map(line => line.trim().split(/\s+/))
    .filter(parts => parts.length >= 2)
    .map(([token, rank]) => ({ token, rank: Number(rank) }))
    .filter(row => Number.isFinite(row.rank))
    .sort((a, b) => a.rank - b.rank);

  const compressedLines = [];
  let groupStartRank = null;
  let groupTokens = [];
  let expectedRank = null;
  const flush = () => {
    if (groupTokens.length) compressedLines.push(`x ${groupStartRank} ${groupTokens.join(' ')}`);
    groupStartRank = null;
    groupTokens = [];
    expectedRank = null;
  };

  for (const row of rows) {
    if (expectedRank === null || row.rank !== expectedRank) {
      flush();
      groupStartRank = row.rank;
      expectedRank = row.rank;
    }
    groupTokens.push(row.token);
    expectedRank += 1;
  }
  flush();

  return { pat_str: patStr, bpe_ranks: compressedLines.join('\n'), special_tokens: {} };
}

const cfg = JSON.parse(await fetchText(`${baseRaw}/tokenizer_config.json`));
const pointer = await fetchText(`${baseRaw}/tiktoken.model`);
assert.match(pointer, /^version https:\/\/git-lfs\.github\.com\/spec\//, 'raw URL should be an LFS pointer; app must not use it');

const modelText = await fetchText(`${baseResolve}/tiktoken.model`);
assert.ok(modelText.length > 1_000_000, 'resolve URL should fetch real tiktoken.model');
assert.doesNotMatch(modelText, /^version https:\/\/git-lfs\.github\.com\/spec\//);

const enc = new Tiktoken(
  tiktokenModelToJsTiktokenRanks(modelText, getKimiCompatiblePatStr()),
  buildSpecialTokensFromConfig(cfg),
);

const hello = enc.encode('Hello world', 'all', []);
assert.deepEqual(hello, [19180, 2695]);
assert.equal(enc.decode(hello), 'Hello world');

const prompt = '<|im_system|>system<|im_middle|>You are concise.<|im_end|><|im_user|>user<|im_middle|>What is the weather in Montreal?<|im_end|><|im_assistant|>assistant<|im_middle|><think>';
const ids = enc.encode(prompt, 'all', []);
assert.ok(ids.length > 20, `expected full prompt tokenization, got ${ids.length}`);
assert.equal(enc.decode(ids), prompt);
assert.ok(ids.some(id => id < 163584), 'should include normal text token IDs, not only special tokens');

console.log('Kimi tiktoken adapter tests passed');

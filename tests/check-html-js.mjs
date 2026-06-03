import fs from 'node:fs';

const html = fs.readFileSync('preview (1).html', 'utf8');
const startMarker = '<script type="module">';
const start = html.indexOf(startMarker);
if (start === -1) throw new Error('No module script found');
const end = html.indexOf('</script>', start);
if (end === -1) throw new Error('No closing script tag found');
const script = html.slice(start + startMarker.length, end);
const out = '/tmp/promptloom-script.mjs';
fs.writeFileSync(out, script);
console.log(out);

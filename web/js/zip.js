// A zip of named files, stored without compression: enough to bundle an eval's artifacts into one
// download without loading a zip library.

const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (bytes) => {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
// little-endian fields at fixed offsets: [offset, value, width in bytes]
const record = (size, fields) => {
  const view = new DataView(new ArrayBuffer(size));
  for (const [at, value, width] of fields) width === 4 ? view.setUint32(at, value, true) : view.setUint16(at, value, true);
  return new Uint8Array(view.buffer);
};

// files: [{ name, bytes: Uint8Array }] -> a Blob holding the zip
export function zip(files) {
  const now = new Date();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  const body = [], directory = [];
  let offset = 0;
  for (const { name, bytes } of files) {
    const path = new TextEncoder().encode(name);
    const crc = crc32(bytes);
    // version 2.0, flag 0x0800 (UTF-8 names), method 0 (stored)
    const local = record(30, [[0, 0x04034b50, 4], [4, 20, 2], [6, 0x0800, 2], [10, time, 2], [12, date, 2],
      [14, crc, 4], [18, bytes.length, 4], [22, bytes.length, 4], [26, path.length, 2]]);
    directory.push(record(46, [[0, 0x02014b50, 4], [4, 20, 2], [6, 20, 2], [8, 0x0800, 2], [12, time, 2], [14, date, 2],
      [16, crc, 4], [20, bytes.length, 4], [24, bytes.length, 4], [28, path.length, 2], [42, offset, 4]]), path);
    body.push(local, path, bytes);
    offset += local.length + path.length + bytes.length;
  }
  const size = directory.reduce((n, part) => n + part.length, 0);
  const end = record(22, [[0, 0x06054b50, 4], [8, files.length, 2], [10, files.length, 2], [12, size, 4], [16, offset, 4]]);
  return new Blob([...body, ...directory, end], { type: "application/zip" });
}

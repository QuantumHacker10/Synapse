// MegascansImageDecoder.cs — on-disk texture decode for Megascans / MaterialTextureSystem.
// Formats: PNG (8/16-bit, non-interlaced), baseline JPEG, BMP, TGA.
// No NuGet image packages; zlib via System.IO.Compression.ZLibStream.
using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;

namespace GDNN.Rendering
{
    /// <summary>Decoded RGBA8 bitmap ready for Vulkan upload.</summary>
    public sealed class DecodedImage
    {
        public int Width { get; init; }
        public int Height { get; init; }
        public byte[] Rgba8 { get; init; } = Array.Empty<byte>();
    }

    /// <summary>
    /// Minimal Megascans texture decoder (PNG / JPEG / BMP / TGA).
    /// EXR/TIFF are discovered by MegascansBridge but not decoded here — callers fall back procedurally.
    /// </summary>
    public static class MegascansImageDecoder
    {
        public const int MaxDimension = 4096;

        public static bool TryDecodeFile(string path, out DecodedImage image)
        {
            image = null!;
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                return false;

            try
            {
                var data = File.ReadAllBytes(path);
                return TryDecode(data, Path.GetExtension(path), out image);
            }
            catch
            {
                return false;
            }
        }

        public static bool TryDecode(ReadOnlySpan<byte> data, string? extensionHint, out DecodedImage image)
        {
            image = null!;
            if (data.Length < 8)
                return false;

            try
            {
                if (IsPng(data))
                    return TryDecodePng(data, out image);
                if (IsJpeg(data))
                    return TryDecodeJpeg(data, out image);
                if (IsBmp(data))
                    return TryDecodeBmp(data, out image);
                if (IsTga(data, extensionHint))
                    return TryDecodeTga(data, out image);
            }
            catch
            {
                return false;
            }

            return false;
        }

        private static bool IsPng(ReadOnlySpan<byte> d) =>
            d.Length >= 8 && d[0] == 0x89 && d[1] == 0x50 && d[2] == 0x4E && d[3] == 0x47
            && d[4] == 0x0D && d[5] == 0x0A && d[6] == 0x1A && d[7] == 0x0A;

        private static bool IsJpeg(ReadOnlySpan<byte> d) =>
            d.Length >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF;

        private static bool IsBmp(ReadOnlySpan<byte> d) =>
            d.Length >= 14 && d[0] == (byte)'B' && d[1] == (byte)'M';

        private static bool IsTga(ReadOnlySpan<byte> d, string? ext)
        {
            if (ext != null)
            {
                var e = ext.ToLowerInvariant();
                if (e is ".tga" or ".targa")
                    return d.Length >= 18;
            }
            // Heuristic: uncompressed truecolor / grayscale without relying on footer.
            if (d.Length < 18) return false;
            byte type = d[2];
            return type is 2 or 3 or 10;
        }

        private static DecodedImage ClampAndPack(int width, int height, byte[] rgba)
        {
            if (width <= 0 || height <= 0 || rgba.Length < width * height * 4)
                throw new InvalidDataException("Invalid image dimensions.");

            if (width <= MaxDimension && height <= MaxDimension)
                return new DecodedImage { Width = width, Height = height, Rgba8 = rgba };

            int nw = Math.Min(width, MaxDimension);
            int nh = Math.Min(height, MaxDimension);
            // Preserve aspect when only one axis exceeds.
            if (width > MaxDimension || height > MaxDimension)
            {
                float scale = Math.Min(MaxDimension / (float)width, MaxDimension / (float)height);
                nw = Math.Max(1, (int)(width * scale));
                nh = Math.Max(1, (int)(height * scale));
            }

            var dst = new byte[nw * nh * 4];
            for (int y = 0; y < nh; y++)
            {
                int sy = y * height / nh;
                for (int x = 0; x < nw; x++)
                {
                    int sx = x * width / nw;
                    int si = (sy * width + sx) * 4;
                    int di = (y * nw + x) * 4;
                    dst[di] = rgba[si];
                    dst[di + 1] = rgba[si + 1];
                    dst[di + 2] = rgba[si + 2];
                    dst[di + 3] = rgba[si + 3];
                }
            }
            return new DecodedImage { Width = nw, Height = nh, Rgba8 = dst };
        }

        #region PNG

        private static bool TryDecodePng(ReadOnlySpan<byte> data, out DecodedImage image)
        {
            image = null!;
            int offset = 8;
            int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
            byte[]? palette = null;
            byte[]? trns = null;
            using var idat = new MemoryStream();

            while (offset + 12 <= data.Length)
            {
                int length = BinaryPrimitives.ReadInt32BigEndian(data.Slice(offset, 4));
                offset += 4;
                if (length < 0 || offset + length + 8 > data.Length)
                    return false;

                uint type = BinaryPrimitives.ReadUInt32BigEndian(data.Slice(offset, 4));
                offset += 4;
                var chunk = data.Slice(offset, length);
                offset += length;
                offset += 4; // CRC

                switch (type)
                {
                    case 0x49484452: // IHDR
                        if (length < 13) return false;
                        width = BinaryPrimitives.ReadInt32BigEndian(chunk);
                        height = BinaryPrimitives.ReadInt32BigEndian(chunk.Slice(4));
                        bitDepth = chunk[8];
                        colorType = chunk[9];
                        if (chunk[10] != 0 || chunk[11] != 0) return false; // compression/filter
                        interlace = chunk[12];
                        if (interlace != 0) return false; // Adam7 not supported
                        if (width <= 0 || height <= 0) return false;
                        break;
                    case 0x504C5445: // PLTE
                        palette = chunk.ToArray();
                        break;
                    case 0x74524E53: // tRNS
                        trns = chunk.ToArray();
                        break;
                    case 0x49444154: // IDAT
                        idat.Write(chunk);
                        break;
                    case 0x49454E44: // IEND
                        goto Done;
                }
            }

        Done:
            if (width == 0 || height == 0 || idat.Length == 0)
                return false;

            byte[] raw = InflateZlib(idat.ToArray());
            int channels = colorType switch
            {
                0 => 1,
                2 => 3,
                3 => 1,
                4 => 2,
                6 => 4,
                _ => 0
            };
            if (channels == 0) return false;
            if (bitDepth is not (8 or 16)) return false;
            if (colorType == 3 && bitDepth != 8) return false;

            int bytesPerPixel = channels * (bitDepth / 8);
            int stride = width * bytesPerPixel;
            int expected = (stride + 1) * height;
            if (raw.Length < expected) return false;

            var unfiltered = new byte[stride * height];
            var prev = new byte[stride];
            int src = 0;
            for (int y = 0; y < height; y++)
            {
                byte filter = raw[src++];
                var row = new byte[stride];
                Buffer.BlockCopy(raw, src, row, 0, stride);
                src += stride;
                ApplyPngFilter(filter, row, prev, bytesPerPixel);
                Buffer.BlockCopy(row, 0, unfiltered, y * stride, stride);
                Buffer.BlockCopy(row, 0, prev, 0, stride);
            }

            var rgba = new byte[width * height * 4];
            for (int i = 0; i < width * height; i++)
            {
                int o = i * 4;
                switch (colorType)
                {
                    case 0: // Gray
                    {
                        byte g = SamplePngChannel(unfiltered, i, 0, channels, bitDepth);
                        rgba[o] = g; rgba[o + 1] = g; rgba[o + 2] = g; rgba[o + 3] = 255;
                        break;
                    }
                    case 2: // RGB
                    {
                        rgba[o] = SamplePngChannel(unfiltered, i, 0, channels, bitDepth);
                        rgba[o + 1] = SamplePngChannel(unfiltered, i, 1, channels, bitDepth);
                        rgba[o + 2] = SamplePngChannel(unfiltered, i, 2, channels, bitDepth);
                        rgba[o + 3] = 255;
                        break;
                    }
                    case 3: // Indexed
                    {
                        byte idx = unfiltered[i];
                        if (palette == null || idx * 3 + 2 >= palette.Length) return false;
                        rgba[o] = palette[idx * 3];
                        rgba[o + 1] = palette[idx * 3 + 1];
                        rgba[o + 2] = palette[idx * 3 + 2];
                        rgba[o + 3] = trns != null && idx < trns.Length ? trns[idx] : (byte)255;
                        break;
                    }
                    case 4: // Gray+A
                    {
                        byte g = SamplePngChannel(unfiltered, i, 0, channels, bitDepth);
                        byte a = SamplePngChannel(unfiltered, i, 1, channels, bitDepth);
                        rgba[o] = g; rgba[o + 1] = g; rgba[o + 2] = g; rgba[o + 3] = a;
                        break;
                    }
                    case 6: // RGBA
                    {
                        rgba[o] = SamplePngChannel(unfiltered, i, 0, channels, bitDepth);
                        rgba[o + 1] = SamplePngChannel(unfiltered, i, 1, channels, bitDepth);
                        rgba[o + 2] = SamplePngChannel(unfiltered, i, 2, channels, bitDepth);
                        rgba[o + 3] = SamplePngChannel(unfiltered, i, 3, channels, bitDepth);
                        break;
                    }
                }
            }

            image = ClampAndPack(width, height, rgba);
            return true;
        }

        private static byte SamplePngChannel(byte[] data, int pixel, int channel, int channels, int bitDepth)
        {
            if (bitDepth == 8)
                return data[pixel * channels + channel];
            int idx = (pixel * channels + channel) * 2;
            // Take high byte of big-endian 16-bit sample.
            return data[idx];
        }

        private static void ApplyPngFilter(byte filter, byte[] row, byte[] prev, int bpp)
        {
            switch (filter)
            {
                case 0: break;
                case 1: // Sub
                    for (int i = bpp; i < row.Length; i++)
                        row[i] = (byte)(row[i] + row[i - bpp]);
                    break;
                case 2: // Up
                    for (int i = 0; i < row.Length; i++)
                        row[i] = (byte)(row[i] + prev[i]);
                    break;
                case 3: // Average
                    for (int i = 0; i < row.Length; i++)
                    {
                        int left = i >= bpp ? row[i - bpp] : 0;
                        row[i] = (byte)(row[i] + ((left + prev[i]) >> 1));
                    }
                    break;
                case 4: // Paeth
                    for (int i = 0; i < row.Length; i++)
                    {
                        int a = i >= bpp ? row[i - bpp] : 0;
                        int b = prev[i];
                        int c = i >= bpp ? prev[i - bpp] : 0;
                        row[i] = (byte)(row[i] + Paeth(a, b, c));
                    }
                    break;
                default:
                    throw new InvalidDataException($"Unsupported PNG filter {filter}");
            }
        }

        private static int Paeth(int a, int b, int c)
        {
            int p = a + b - c;
            int pa = Math.Abs(p - a);
            int pb = Math.Abs(p - b);
            int pc = Math.Abs(p - c);
            if (pa <= pb && pa <= pc) return a;
            if (pb <= pc) return b;
            return c;
        }

        private static byte[] InflateZlib(byte[] zlib)
        {
            using var input = new MemoryStream(zlib);
            using var zs = new ZLibStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            zs.CopyTo(output);
            return output.ToArray();
        }

        #endregion

        #region BMP

        private static bool TryDecodeBmp(ReadOnlySpan<byte> data, out DecodedImage image)
        {
            image = null!;
            if (data.Length < 54) return false;
            int pixelOffset = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(10, 4));
            int dibSize = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(14, 4));
            if (dibSize < 40) return false;
            int width = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(18, 4));
            int height = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(22, 4));
            bool topDown = height < 0;
            height = Math.Abs(height);
            ushort planes = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(26, 2));
            ushort bpp = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(28, 2));
            int compression = BinaryPrimitives.ReadInt32LittleEndian(data.Slice(30, 4));
            if (planes != 1 || compression != 0 || width <= 0 || height <= 0)
                return false;
            if (bpp is not (24 or 32)) return false;

            int rowBytes = ((width * bpp + 31) / 32) * 4;
            if (pixelOffset < 0 || pixelOffset + rowBytes * (long)height > data.Length)
                return false;

            var rgba = new byte[width * height * 4];
            for (int y = 0; y < height; y++)
            {
                int srcY = topDown ? y : height - 1 - y;
                int srcRow = pixelOffset + srcY * rowBytes;
                for (int x = 0; x < width; x++)
                {
                    int si = srcRow + x * (bpp / 8);
                    int di = (y * width + x) * 4;
                    rgba[di] = data[si + 2];
                    rgba[di + 1] = data[si + 1];
                    rgba[di + 2] = data[si];
                    rgba[di + 3] = bpp == 32 ? data[si + 3] : (byte)255;
                }
            }

            image = ClampAndPack(width, height, rgba);
            return true;
        }

        #endregion

        #region TGA

        private static bool TryDecodeTga(ReadOnlySpan<byte> data, out DecodedImage image)
        {
            image = null!;
            if (data.Length < 18) return false;
            byte idLength = data[0];
            byte colorMapType = data[1];
            byte imageType = data[2];
            if (colorMapType != 0) return false;
            if (imageType is not (2 or 3 or 10)) return false;

            int width = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(12, 2));
            int height = BinaryPrimitives.ReadUInt16LittleEndian(data.Slice(14, 2));
            byte bpp = data[16];
            byte descriptor = data[17];
            if (width <= 0 || height <= 0) return false;
            if (bpp is not (8 or 24 or 32)) return false;

            int dataOffset = 18 + idLength;
            bool topOrigin = (descriptor & 0x20) != 0;
            var pixels = new byte[width * height * (bpp / 8)];
            int needed = pixels.Length;

            if (imageType is 2 or 3)
            {
                if (dataOffset + needed > data.Length) return false;
                data.Slice(dataOffset, needed).CopyTo(pixels);
            }
            else // RLE truecolor
            {
                if (!DecodeTgaRle(data.Slice(dataOffset), pixels, bpp / 8))
                    return false;
            }

            var rgba = new byte[width * height * 4];
            int cpp = bpp / 8;
            for (int y = 0; y < height; y++)
            {
                int srcY = topOrigin ? y : height - 1 - y;
                for (int x = 0; x < width; x++)
                {
                    int si = (srcY * width + x) * cpp;
                    int di = (y * width + x) * 4;
                    if (cpp == 1)
                    {
                        byte g = pixels[si];
                        rgba[di] = g; rgba[di + 1] = g; rgba[di + 2] = g; rgba[di + 3] = 255;
                    }
                    else
                    {
                        rgba[di] = pixels[si + 2];
                        rgba[di + 1] = pixels[si + 1];
                        rgba[di + 2] = pixels[si];
                        rgba[di + 3] = cpp == 4 ? pixels[si + 3] : (byte)255;
                    }
                }
            }

            image = ClampAndPack(width, height, rgba);
            return true;
        }

        private static bool DecodeTgaRle(ReadOnlySpan<byte> src, byte[] dst, int pixelSize)
        {
            int si = 0, di = 0;
            while (di < dst.Length)
            {
                if (si >= src.Length) return false;
                byte packet = src[si++];
                int count = (packet & 0x7F) + 1;
                if ((packet & 0x80) != 0)
                {
                    if (si + pixelSize > src.Length) return false;
                    for (int i = 0; i < count; i++)
                    {
                        if (di + pixelSize > dst.Length) return false;
                        src.Slice(si, pixelSize).CopyTo(dst.AsSpan(di, pixelSize));
                        di += pixelSize;
                    }
                    si += pixelSize;
                }
                else
                {
                    int bytes = count * pixelSize;
                    if (si + bytes > src.Length || di + bytes > dst.Length) return false;
                    src.Slice(si, bytes).CopyTo(dst.AsSpan(di, bytes));
                    si += bytes;
                    di += bytes;
                }
            }
            return true;
        }

        #endregion

        #region JPEG (baseline SOF0)

        private static bool TryDecodeJpeg(ReadOnlySpan<byte> data, out DecodedImage image)
        {
            image = null!;
            var jpeg = new JpegDecoder();
            if (!jpeg.Decode(data, out int w, out int h, out byte[] rgba))
                return false;
            image = ClampAndPack(w, h, rgba);
            return true;
        }

        private sealed class JpegDecoder
        {
            private readonly int[][] _quant = { new int[64], new int[64] };
            private readonly HuffmanTable[] _dcTables = new HuffmanTable[4];
            private readonly HuffmanTable[] _acTables = new HuffmanTable[4];
            private Component[] _components = Array.Empty<Component>();
            private int _width, _height;
            private int _maxH, _maxV;
            private byte[] _data = Array.Empty<byte>();
            private int _pos;
            private int _bits;
            private int _bitCount;
            private readonly int[] _zigzag =
            {
                0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,
                57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
            };

            private struct Component
            {
                public int Id, H, V, QuantId, DcId, AcId;
                public short[] Pred;
                public float[]? Coeff;
                public int BlocksW, BlocksH;
            }

            private sealed class HuffmanTable
            {
                public readonly byte[] Codes = new byte[256];
                public readonly byte[] Sizes = new byte[256];
                public readonly short[] MinCode = new short[17];
                public readonly short[] MaxCode = new short[17];
                public readonly short[] ValPtr = new short[17];
                public byte[] Values = Array.Empty<byte>();
            }

            public bool Decode(ReadOnlySpan<byte> data, out int width, out int height, out byte[] rgba)
            {
                width = height = 0;
                rgba = Array.Empty<byte>();
                _data = data.ToArray();
                _pos = 0;
                if (!NextMarker(out byte m) || m != 0xD8) return false;

                bool gotSof = false;
                while (_pos < _data.Length)
                {
                    if (!NextMarker(out m)) return false;
                    if (m == 0xD9) break; // EOI
                    if (m == 0xDA) // SOS
                    {
                        if (!gotSof || !ReadSos()) return false;
                        if (!DecodeScan()) return false;
                        break;
                    }

                    int len = ReadMarkerLength();
                    if (len < 2 || _pos + len - 2 > _data.Length) return false;
                    int start = _pos;
                    int end = _pos + len - 2;

                    switch (m)
                    {
                        case 0xC0: // SOF0 baseline
                            if (!ReadSof(start, end)) return false;
                            gotSof = true;
                            break;
                        case 0xC4: // DHT
                            if (!ReadDht(start, end)) return false;
                            break;
                        case 0xDB: // DQT
                            if (!ReadDqt(start, end)) return false;
                            break;
                        case 0xDD: // DRI — ignore
                            break;
                        default:
                            // APP/COM/etc.
                            break;
                    }
                    _pos = end;
                }

                if (!gotSof || _components.Length == 0) return false;
                rgba = YCbCrToRgba();
                width = _width;
                height = _height;
                return true;
            }

            private bool NextMarker(out byte marker)
            {
                marker = 0;
                while (_pos < _data.Length)
                {
                    if (_data[_pos++] != 0xFF) continue;
                    while (_pos < _data.Length && _data[_pos] == 0xFF) _pos++;
                    if (_pos >= _data.Length) return false;
                    marker = _data[_pos++];
                    if (marker != 0) return true;
                }
                return false;
            }

            private int ReadMarkerLength()
            {
                if (_pos + 2 > _data.Length) return 0;
                int len = (_data[_pos] << 8) | _data[_pos + 1];
                _pos += 2;
                return len;
            }

            private bool ReadSof(int start, int end)
            {
                if (end - start < 6) return false;
                int p = start;
                byte precision = _data[p++];
                if (precision != 8) return false;
                _height = (_data[p] << 8) | _data[p + 1]; p += 2;
                _width = (_data[p] << 8) | _data[p + 1]; p += 2;
                int n = _data[p++];
                if (n is < 1 or > 4 || _width <= 0 || _height <= 0) return false;
                _components = new Component[n];
                _maxH = _maxV = 0;
                for (int i = 0; i < n; i++)
                {
                    if (p + 3 > end) return false;
                    var c = new Component
                    {
                        Id = _data[p++],
                        H = _data[p] >> 4,
                        V = _data[p] & 0x0F,
                        QuantId = _data[p + 1],
                        Pred = new short[1]
                    };
                    p += 2;
                    if (c.H == 0 || c.V == 0) return false;
                    _maxH = Math.Max(_maxH, c.H);
                    _maxV = Math.Max(_maxV, c.V);
                    _components[i] = c;
                }
                return true;
            }

            private bool ReadDqt(int start, int end)
            {
                int p = start;
                while (p < end)
                {
                    int info = _data[p++];
                    int id = info & 0x0F;
                    int prec = info >> 4;
                    if (id > 1) return false;
                    for (int i = 0; i < 64; i++)
                    {
                        int v;
                        if (prec != 0)
                        {
                            if (p + 1 >= end) return false;
                            v = (_data[p] << 8) | _data[p + 1];
                            p += 2;
                        }
                        else
                        {
                            if (p >= end) return false;
                            v = _data[p++];
                        }
                        _quant[id][_zigzag[i]] = v;
                    }
                }
                return true;
            }

            private bool ReadDht(int start, int end)
            {
                int p = start;
                while (p < end)
                {
                    if (p >= end) return false;
                    int info = _data[p++];
                    int tc = info >> 4;
                    int th = info & 0x0F;
                    if (th > 3) return false;
                    if (p + 16 > end) return false;
                    var counts = new byte[17];
                    int total = 0;
                    for (int i = 1; i <= 16; i++)
                    {
                        counts[i] = _data[p++];
                        total += counts[i];
                    }
                    if (p + total > end) return false;
                    var values = new byte[total];
                    for (int i = 0; i < total; i++)
                        values[i] = _data[p++];

                    var table = BuildHuffman(counts, values);
                    if (tc == 0) _dcTables[th] = table;
                    else _acTables[th] = table;
                }
                return true;
            }

            private static HuffmanTable BuildHuffman(byte[] counts, byte[] values)
            {
                var t = new HuffmanTable { Values = values };
                int code = 0, k = 0;
                for (int i = 1; i <= 16; i++)
                {
                    t.ValPtr[i] = (short)k;
                    t.MinCode[i] = (short)code;
                    for (int j = 0; j < counts[i]; j++)
                    {
                        if (k < 256)
                        {
                            t.Codes[k] = (byte)code;
                            t.Sizes[k] = (byte)i;
                        }
                        code++;
                        k++;
                    }
                    t.MaxCode[i] = (short)(code - 1);
                    code <<= 1;
                }
                for (int i = 1; i <= 16; i++)
                {
                    if (counts[i] == 0)
                    {
                        t.MaxCode[i] = -1;
                        t.MinCode[i] = -1;
                    }
                }
                return t;
            }

            private bool ReadSos()
            {
                int len = ReadMarkerLength();
                if (len < 6) return false;
                int end = _pos + len - 2;
                int n = _data[_pos++];
                if (n != _components.Length) return false;
                for (int i = 0; i < n; i++)
                {
                    if (_pos + 2 > end) return false;
                    int id = _data[_pos++];
                    int tdta = _data[_pos++];
                    int idx = Array.FindIndex(_components, c => c.Id == id);
                    if (idx < 0) return false;
                    var c = _components[idx];
                    c.DcId = tdta >> 4;
                    c.AcId = tdta & 0x0F;
                    _components[idx] = c;
                }
                // Ss, Se, AhAl
                if (_pos + 3 > end) return false;
                _pos = end;

                int mcuW = (_width + _maxH * 8 - 1) / (_maxH * 8);
                int mcuH = (_height + _maxV * 8 - 1) / (_maxV * 8);
                for (int i = 0; i < _components.Length; i++)
                {
                    var c = _components[i];
                    c.BlocksW = mcuW * c.H;
                    c.BlocksH = mcuH * c.V;
                    c.Coeff = new float[c.BlocksW * c.BlocksH * 64];
                    c.Pred = new short[1];
                    _components[i] = c;
                }
                _bits = 0;
                _bitCount = 0;
                return true;
            }

            private bool DecodeScan()
            {
                int mcuW = (_width + _maxH * 8 - 1) / (_maxH * 8);
                int mcuH = (_height + _maxV * 8 - 1) / (_maxV * 8);

                for (int my = 0; my < mcuH; my++)
                {
                    for (int mx = 0; mx < mcuW; mx++)
                    {
                        for (int ci = 0; ci < _components.Length; ci++)
                        {
                            var c = _components[ci];
                            for (int v = 0; v < c.V; v++)
                            {
                                for (int h = 0; h < c.H; h++)
                                {
                                    int bx = mx * c.H + h;
                                    int by = my * c.V + v;
                                    if (!DecodeBlock(ci, bx, by))
                                        return false;
                                }
                            }
                        }
                    }
                }
                return true;
            }

            private bool DecodeBlock(int ci, int bx, int by)
            {
                var c = _components[ci];
                var dcTable = _dcTables[c.DcId] ?? throw new InvalidDataException("Missing DC Huffman table");
                var acTable = _acTables[c.AcId] ?? throw new InvalidDataException("Missing AC Huffman table");
                var quant = _quant[c.QuantId];

                int[] block = new int[64];
                int t = DecodeHuffman(dcTable);
                if (t < 0) return false;
                int diff = t == 0 ? 0 : ReceiveExtend(t);
                c.Pred[0] = (short)(c.Pred[0] + diff);
                block[0] = c.Pred[0];

                int k = 1;
                while (k < 64)
                {
                    int rs = DecodeHuffman(acTable);
                    if (rs < 0) return false;
                    int s = rs & 0x0F;
                    int r = rs >> 4;
                    if (s == 0)
                    {
                        if (r == 15) { k += 16; continue; }
                        break;
                    }
                    k += r;
                    if (k >= 64) break;
                    block[k] = ReceiveExtend(s);
                    k++;
                }

                // Dequantize + zigzag inverse into natural order, then IDCT into Coeff as spatial.
                float[] natural = new float[64];
                for (int i = 0; i < 64; i++)
                    natural[_zigzag[i]] = block[i] * quant[_zigzag[i]];

                float[] spatial = new float[64];
                Idct(natural, spatial);

                int baseIdx = (by * c.BlocksW + bx) * 64;
                Array.Copy(spatial, 0, c.Coeff!, baseIdx, 64);
                _components[ci] = c;
                return true;
            }

            private int DecodeHuffman(HuffmanTable table)
            {
                int code = 0;
                for (int i = 1; i <= 16; i++)
                {
                    code = (code << 1) | GetBit();
                    if (table.MaxCode[i] >= 0 && code <= table.MaxCode[i] && code >= table.MinCode[i])
                    {
                        int j = table.ValPtr[i] + code - table.MinCode[i];
                        if (j < 0 || j >= table.Values.Length) return -1;
                        return table.Values[j];
                    }
                }
                return -1;
            }

            private int ReceiveExtend(int s)
            {
                int v = GetBits(s);
                int vt = 1 << (s - 1);
                if (v < vt)
                    v += (-1 << s) + 1;
                return v;
            }

            private int GetBit()
            {
                if (_bitCount == 0)
                {
                    int b = NextEntropyByte();
                    _bits = b;
                    _bitCount = 8;
                }
                _bitCount--;
                return (_bits >> _bitCount) & 1;
            }

            private int GetBits(int n)
            {
                int v = 0;
                for (int i = 0; i < n; i++)
                    v = (v << 1) | GetBit();
                return v;
            }

            private int NextEntropyByte()
            {
                if (_pos >= _data.Length) return 0;
                int b = _data[_pos++];
                if (b == 0xFF)
                {
                    if (_pos >= _data.Length) return 0;
                    int n = _data[_pos++];
                    if (n != 0)
                    {
                        // Marker inside scan — push back and return 0 fill.
                        _pos -= 2;
                        return 0;
                    }
                }
                return b;
            }

            private static void Idct(float[] src, float[] dst)
            {
                // Separable AAN-ish IDCT (float), good enough for albedo maps.
                float[] tmp = new float[64];
                for (int y = 0; y < 8; y++)
                {
                    for (int x = 0; x < 8; x++)
                    {
                        float sum = 0;
                        for (int u = 0; u < 8; u++)
                        {
                            float cu = u == 0 ? 0.70710678118f : 1f;
                            sum += cu * src[y * 8 + u] * MathF.Cos((2 * x + 1) * u * MathF.PI / 16f);
                        }
                        tmp[y * 8 + x] = sum * 0.5f;
                    }
                }
                for (int x = 0; x < 8; x++)
                {
                    for (int y = 0; y < 8; y++)
                    {
                        float sum = 0;
                        for (int v = 0; v < 8; v++)
                        {
                            float cv = v == 0 ? 0.70710678118f : 1f;
                            sum += cv * tmp[v * 8 + x] * MathF.Cos((2 * y + 1) * v * MathF.PI / 16f);
                        }
                        dst[y * 8 + x] = sum * 0.5f;
                    }
                }
            }

            private byte[] YCbCrToRgba()
            {
                var rgba = new byte[_width * _height * 4];
                var yComp = _components[0];
                Component? cbComp = _components.Length > 1 ? _components[1] : null;
                Component? crComp = _components.Length > 2 ? _components[2] : null;

                for (int py = 0; py < _height; py++)
                {
                    for (int px = 0; px < _width; px++)
                    {
                        float Y = SampleComponent(yComp, px, py) + 128f;
                        float Cb = cbComp.HasValue ? SampleComponent(cbComp.Value, px, py) : 0f;
                        float Cr = crComp.HasValue ? SampleComponent(crComp.Value, px, py) : 0f;

                        float r = Y + 1.402f * Cr;
                        float g = Y - 0.344136f * Cb - 0.714136f * Cr;
                        float b = Y + 1.772f * Cb;

                        int di = (py * _width + px) * 4;
                        rgba[di] = ClampByte(r);
                        rgba[di + 1] = ClampByte(g);
                        rgba[di + 2] = ClampByte(b);
                        rgba[di + 3] = 255;
                    }
                }
                return rgba;
            }

            private float SampleComponent(Component c, int px, int py)
            {
                // Map pixel to component sample space.
                float sx = px * c.H / (float)_maxH;
                float sy = py * c.V / (float)_maxV;
                int bx = Math.Min((int)sx / 8, c.BlocksW - 1);
                int by = Math.Min((int)sy / 8, c.BlocksH - 1);
                int lx = Math.Min((int)sx % 8, 7);
                int ly = Math.Min((int)sy % 8, 7);
                // Fix modulo for float:
                lx = ((int)sx) - bx * 8;
                ly = ((int)sy) - by * 8;
                if (lx < 0) lx = 0; if (lx > 7) lx = 7;
                if (ly < 0) ly = 0; if (ly > 7) ly = 7;
                return c.Coeff![(by * c.BlocksW + bx) * 64 + ly * 8 + lx];
            }

            private static byte ClampByte(float v)
            {
                if (v < 0) return 0;
                if (v > 255) return 255;
                return (byte)(v + 0.5f);
            }
        }

        #endregion
    }
}

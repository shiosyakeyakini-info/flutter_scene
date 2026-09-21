/// Covers CPU mip-chain generation: chain sizing, and content-aware
/// downsampling (sRGB color averaged in linear light, data averaged directly,
/// normals averaged as vectors and renormalized).
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_scene/src/texture/mipmap.dart';
import 'package:test/test.dart';

Uint8List _solid(int w, int h, int r, int g, int b, int a) {
  final p = Uint8List(w * h * 4);
  for (var i = 0; i < w * h; i++) {
    p[i * 4] = r;
    p[i * 4 + 1] = g;
    p[i * 4 + 2] = b;
    p[i * 4 + 3] = a;
  }
  return p;
}

void main() {
  test('mipLevelCountFor is floor(log2(max)) + 1', () {
    expect(mipLevelCountFor(256, 256), 9);
    expect(mipLevelCountFor(1, 1), 1);
    expect(mipLevelCountFor(8, 2), 4);
  });

  test('chain halves down to 1x1 with level 0 first', () {
    final chain = generateMipChain(
      _solid(4, 4, 10, 20, 30, 255),
      4,
      4,
      TextureContent.data,
    );
    expect(chain.map((l) => '${l.width}x${l.height}'), ['4x4', '2x2', '1x1']);
    // A solid image stays solid at every level.
    expect(chain.last.pixels, [10, 20, 30, 255]);
  });

  test('color content averages in linear light, not naively', () {
    // A 2x2 checker of black and white. Naive byte average = 127; correct
    // linear average is 0.5 in linear -> ~188 in sRGB.
    final pixels = Uint8List.fromList([
      0, 0, 0, 255, // black
      255, 255, 255, 255, // white
      255, 255, 255, 255, // white
      0, 0, 0, 255, // black
    ]);
    final chain = generateMipChain(pixels, 2, 2, TextureContent.color);
    final mip = chain[1].pixels; // 1x1
    expect(mip[0], greaterThan(180));
    expect(mip[0], lessThan(195));
  });

  test('data content averages bytes directly', () {
    final pixels = Uint8List.fromList([
      0, 0, 0, 0, //
      255, 255, 255, 255, //
      255, 255, 255, 255, //
      0, 0, 0, 0, //
    ]);
    final chain = generateMipChain(pixels, 2, 2, TextureContent.data);
    expect(chain[1].pixels, [128, 128, 128, 128]);
  });

  test('normal content renormalizes to a unit vector', () {
    // Flat normals (0,0,1) encoded as (128,128,255) stay flat.
    final chain = generateMipChain(
      _solid(2, 2, 128, 128, 255, 255),
      2,
      2,
      TextureContent.normal,
    );
    final mip = chain[1].pixels;
    expect(mip[0], closeTo(128, 1));
    expect(mip[1], closeTo(128, 1));
    expect(mip[2], closeTo(255, 1));
  });

  // The sRGB transfer functions are tabulated (decode) and bisected over
  // their own rounding boundaries (encode) rather than calling `pow` 15
  // times per output pixel. That is a pure speedup and must not shift a
  // single byte, so check it against the reference formulas directly.
  test('color downsampling matches the reference sRGB transfer exactly', () {
    double refToLinear(int byte) {
      final c = byte / 255.0;
      return c <= 0.04045
          ? c / 12.92
          : math.pow((c + 0.055) / 1.055, 2.4).toDouble();
    }

    int refToSrgb(double linear) {
      final c = linear <= 0.0031308
          ? linear * 12.92
          : 1.055 * math.pow(linear, 1 / 2.4).toDouble() - 0.055;
      return (c * 255.0).round().clamp(0, 255);
    }

    // Deterministic pseudo-random source: every 2x2 block is a different
    // quadruple, so one pass covers a wide spread of averages.
    const w = 256;
    const h = 256;
    final pixels = Uint8List(w * h * 4);
    var seed = 12345;
    for (var i = 0; i < pixels.length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
      pixels[i] = (seed >> 16) & 0xFF;
    }

    final chain = generateMipChain(pixels, w, h, TextureContent.color);

    // Recompute every level from its predecessor with the reference math.
    var src = pixels;
    var sw = w;
    var sh = h;
    for (var level = 1; level < chain.length; level++) {
      final dw = math.max(1, sw >> 1);
      final dh = math.max(1, sh >> 1);
      final got = chain[level];
      expect(got.width, dw);
      expect(got.height, dh);
      for (var y = 0; y < dh; y++) {
        final y0 = math.min(y * 2, sh - 1);
        final y1 = math.min(y0 + 1, sh - 1);
        for (var x = 0; x < dw; x++) {
          final x0 = math.min(x * 2, sw - 1);
          final x1 = math.min(x0 + 1, sw - 1);
          final a = (y0 * sw + x0) * 4;
          final b = (y0 * sw + x1) * 4;
          final c = (y1 * sw + x0) * 4;
          final d = (y1 * sw + x1) * 4;
          final o = (y * dw + x) * 4;
          for (var ch = 0; ch < 3; ch++) {
            final avg =
                (refToLinear(src[a + ch]) +
                    refToLinear(src[b + ch]) +
                    refToLinear(src[c + ch]) +
                    refToLinear(src[d + ch])) *
                0.25;
            expect(
              got.pixels[o + ch],
              refToSrgb(avg),
              reason: 'level $level at ($x,$y) channel $ch',
            );
          }
        }
      }
      src = got.pixels;
      sw = dw;
      sh = dh;
    }
  });
}

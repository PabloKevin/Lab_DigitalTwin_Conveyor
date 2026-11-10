// facefind.cpp - C++ port of classifier/preprocess.py. Keep the two in step: the parity test
// (classifier/tests/test_facefind_parity.py) runs both on the dataset and compares status, corners and crop.
#pragma GCC optimize("O3")             // Arduino builds with -Os; this file is the hot path on the ESP32
#include "facefind.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

int64_t (*ff_clock_us)(void) = nullptr;
int32_t ff_timing_us[FF_N_TIMING];

namespace {

typedef std::vector<int32_t> I32;
typedef std::vector<int16_t> I16;       // image-sized buffers: half the memory traffic (PSRAM on the ESP32)

inline int64_t now_us() { return ff_clock_us ? ff_clock_us() : 0; }
typedef std::vector<uint8_t> U8;

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int pyround(double x) { return (int)nearbyint(x); }          // Python round(): half to even

// ── integer primitives (same arithmetic as preprocess.py) ─────────────────────────────────
void blur5(const I16 &ch, int w, int h, I16 &out) {
  static const int K[5] = {1, 4, 6, 4, 1};
  I16 t(w * h);                                                    // |sum| <= 16 * 255 fits int16
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int s = 0;
      for (int i = 0; i < 5; i++) s += K[i] * ch[y * w + clampi(x + i - 2, 0, w - 1)];
      t[y * w + x] = s;
    }
  out.assign(w * h, 0);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int s = 0;
      for (int i = 0; i < 5; i++) s += K[i] * t[clampi(y + i - 2, 0, h - 1) * w + x];
      out[y * w + x] = (s + 128) >> 8;                               // arithmetic shift = floor, like Python
    }
}

void sobel(const I16 &c, int w, int h, I16 &gx, I16 &gy) {
  gx.assign(w * h, 0);
  gy.assign(w * h, 0);
  for (int y = 0; y < h; y++) {
    int ym = clampi(y - 1, 0, h - 1) * w, y0 = y * w, yp = clampi(y + 1, 0, h - 1) * w;
    for (int x = 0; x < w; x++) {
      int xm = clampi(x - 1, 0, w - 1), xp = clampi(x + 1, 0, w - 1);
      gx[y0 + x] = (c[ym + xp] + 2 * c[y0 + xp] + c[yp + xp]) - (c[ym + xm] + 2 * c[y0 + xm] + c[yp + xm]);
      gy[y0 + x] = (c[yp + xm] + 2 * c[yp + x] + c[yp + xp]) - (c[ym + xm] + 2 * c[ym + x] + c[ym + xp]);
    }
  }
}

void dilate(U8 &m, int w, int h, int r) {                          // square (2r+1), out-of-image ignored
  if (r <= 0) return;
  U8 t(w * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint8_t v = 0;
      for (int k = std::max(0, x - r); k <= std::min(w - 1, x + r) && !v; k++) v = m[y * w + k];
      t[y * w + x] = v;
    }
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint8_t v = 0;
      for (int k = std::max(0, y - r); k <= std::min(h - 1, y + r) && !v; k++) v = t[k * w + x];
      m[y * w + x] = v;
    }
}

void erode(U8 &m, int w, int h, int r) {                           // out-of-image counts as foreground
  if (r <= 0) return;
  U8 t(w * h);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint8_t v = 255;
      for (int k = std::max(0, x - r); k <= std::min(w - 1, x + r) && v; k++) v = m[y * w + k];
      t[y * w + x] = v;
    }
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint8_t v = 255;
      for (int k = std::max(0, y - r); k <= std::min(h - 1, y + r) && v; k++) v = t[k * w + x];
      m[y * w + x] = v;
    }
}

void fill_holes(U8 &m, int w, int h) {                             // flood the background from top/bottom rows
  U8 reached(w * h, 0);
  std::vector<int> st;
  for (int x = 0; x < w; x++) {
    st.push_back(x);
    st.push_back((h - 1) * w + x);
  }
  while (!st.empty()) {
    int i = st.back();
    st.pop_back();
    if (m[i] || reached[i]) continue;
    reached[i] = 1;
    int x = i % w, y = i / w;
    if (x > 0) st.push_back(i - 1);
    if (x < w - 1) st.push_back(i + 1);
    if (y > 0) st.push_back(i - w);
    if (y < h - 1) st.push_back(i + w);
  }
  for (int i = 0; i < w * h; i++)
    if (!m[i] && !reached[i]) m[i] = 255;
}

// 8-connected labelling in raster order (same label order as OpenCV). Returns number of labels (>= 1, 0 = bg).
int label8(const U8 &m, int w, int h, I32 &lab) {
  lab.assign(w * h, 0);
  int n = 1;
  std::vector<int> st;
  for (int i = 0; i < w * h; i++) {
    if (!m[i] || lab[i]) continue;
    lab[i] = n;
    st.push_back(i);
    while (!st.empty()) {
      int j = st.back();
      st.pop_back();
      int x = j % w, y = j / w;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          int k = yy * w + xx;
          if (m[k] && !lab[k]) {
            lab[k] = n;
            st.push_back(k);
          }
        }
    }
    n++;
  }
  return n;
}

// ── geometry ────────────────────────────────────────────────────────────────────────────────
struct Pt { float x, y; };
struct RRect { float cx, cy, w, h, ux, uy; };                       // centre, sides along u and v = (-uy, ux)

long cross(const Pt &o, const Pt &a, const Pt &b) {
  return (long)((a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x));
}

std::vector<Pt> hull(std::vector<Pt> p) {                          // Andrew's monotone chain, counter-clockwise
  std::sort(p.begin(), p.end(), [](const Pt &a, const Pt &b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });
  p.erase(std::unique(p.begin(), p.end(), [](const Pt &a, const Pt &b) { return a.x == b.x && a.y == b.y; }), p.end());
  if (p.size() < 3) return p;
  std::vector<Pt> H(2 * p.size());
  size_t k = 0;
  for (size_t i = 0; i < p.size(); i++) {
    while (k >= 2 && cross(H[k - 2], H[k - 1], p[i]) <= 0) k--;
    H[k++] = p[i];
  }
  for (size_t i = p.size() - 1, t = k + 1; i > 0; i--) {
    while (k >= t && cross(H[k - 2], H[k - 1], p[i - 1]) <= 0) k--;
    H[k++] = p[i - 1];
  }
  H.resize(k - 1);
  return H;
}

RRect min_area_rect(const std::vector<Pt> &pts) {                  // rotating calipers over the hull edges
  std::vector<Pt> H = hull(pts);
  RRect best = {0, 0, 0, 0, 1, 0};
  if (H.empty()) return best;
  if (H.size() == 1) { best.cx = H[0].x; best.cy = H[0].y; return best; }
  float bestA = 1e30f;
  for (size_t i = 0; i < H.size(); i++) {
    const Pt &a = H[i], &b = H[(i + 1) % H.size()];
    float ex = b.x - a.x, ey = b.y - a.y, L = sqrtf(ex * ex + ey * ey);
    if (L == 0) continue;
    float ux = ex / L, uy = ey / L;
    float mnu = 1e30f, mxu = -1e30f, mnv = 1e30f, mxv = -1e30f;
    for (const Pt &q : H) {
      float pu = q.x * ux + q.y * uy, pv = -q.x * uy + q.y * ux;
      mnu = std::min(mnu, pu); mxu = std::max(mxu, pu); mnv = std::min(mnv, pv); mxv = std::max(mxv, pv);
    }
    float A = (mxu - mnu) * (mxv - mnv);
    if (A < bestA - 1e-4f) {
      bestA = A;
      float cu = (mnu + mxu) / 2, cv = (mnv + mxv) / 2;
      best = {cu * ux - cv * uy, cu * uy + cv * ux, mxu - mnu, mxv - mnv, ux, uy};
    }
  }
  return best;
}

void box_points(const RRect &r, float w, float h, Pt out[4]) {
  float vx = -r.uy, vy = r.ux;
  const int s[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int i = 0; i < 4; i++) {
    out[i].x = r.cx + s[i][0] * w / 2 * r.ux + s[i][1] * h / 2 * vx;
    out[i].y = r.cy + s[i][0] * w / 2 * r.uy + s[i][1] * h / 2 * vy;
  }
}

float angle90(const RRect &r) {
  float a = atan2f(r.uy, r.ux) * 57.29578f;
  a = fmodf(a, 90.0f);
  return a < 0 ? a + 90.0f : a;
}

bool in_convex(const std::vector<Pt> &H, float x, float y) {       // inside or on the edge (pointPolygonTest >= 0)
  if (H.size() < 3) return false;
  for (size_t i = 0; i < H.size(); i++) {
    const Pt &a = H[i], &b = H[(i + 1) % H.size()];
    if ((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x) < 0) return false;   // CCW hull: left side = inside
  }
  return true;
}

struct Comp {
  int label;                    // > 0 component, 0 merged group
  int x1, y1, x2, y2, area;
  bool touches, has_rect;
  int why;                      // 0 ok, 1 small, 2 big, 3 rect, 4 shape, 5 size, 6 corners
  RRect rect;
  std::vector<Pt> pts, H;       // pixel points (components) / hull (groups)
};

template <class Inside>
int square_test(Comp &c, const ff_params *p, float ppcm, float min_inset, Inside inside) {
  c.rect = min_area_rect(c.H.empty() ? c.pts : c.H);
  c.has_rect = true;
  float s1 = c.rect.w + 1, s2 = c.rect.h + 1;
  if (std::max(s1, s2) / std::max(1.0f, std::min(s1, s2)) > p->max_side_ratio) return 3;
  if (c.area / (s1 * s2) < p->min_rect_fill) return 4;
  float side = (s1 + s2) / 2 / ppcm;
  if (side < p->side_cm_min || side > p->side_cm_max) return 5;
  float frac = std::min(0.25f, std::max(p->corner_probe_frac, min_inset / std::min(s1, s2)));
  Pt q[4];
  box_points(c.rect, c.rect.w, c.rect.h, q);
  int hits = 0;
  for (int i = 0; i < 4; i++) {
    float x = c.rect.cx + (q[i].x - c.rect.cx) * (1 - 2 * frac), y = c.rect.cy + (q[i].y - c.rect.cy) * (1 - 2 * frac);
    hits += inside(pyround(x), pyround(y)) ? 1 : 0;
  }
  return hits >= p->min_corner_hits ? 0 : 6;
}

}  // namespace

// ── pipeline ────────────────────────────────────────────────────────────────────────────────
int ff_process(const uint8_t *img, int W, int H, int stride, const ff_params *p, ff_result *r, uint8_t *crop) {
  memset(r, 0, sizeof(*r));
  const int f = p->downsample;
  int wx1 = 0, wy1 = 0, wx2 = W, wy2 = H;
  if (p->win_x2 > 0) {
    wx1 = std::max(0, p->win_x1); wy1 = std::max(0, p->win_y1);
    wx2 = std::min(W, p->win_x2); wy2 = std::min(H, p->win_y2);
  }
  const int w = (wx2 - wx1) / f, h = (wy2 - wy1) / f, n = w * h;
  if (w < 3 || h < 3) return FF_NONE;

  int64_t tm = now_us();
  // downsample (block mean, rounded) + channels Y, RG, BY
  I16 ch[3];
  for (int c = 0; c < 3; c++) ch[c].assign(n, 0);
  for (int k = 0; k < FF_N_TIMING; k++) ff_timing_us[k] = 0;
  const int ff2 = f * f;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int s[3] = {0, 0, 0};
      for (int dy = 0; dy < f; dy++) {
        const uint8_t *row = img + (size_t)(wy1 + y * f + dy) * stride + (size_t)(wx1 + x * f) * 3;
        for (int dx = 0; dx < f; dx++)
          for (int c = 0; c < 3; c++) s[c] += row[dx * 3 + c];
      }
      int b = (s[0] + ff2 / 2) / ff2, g = (s[1] + ff2 / 2) / ff2, rr = (s[2] + ff2 / 2) / ff2;
      ch[0][y * w + x] = (rr + 2 * g + b + 2) >> 2;
      ch[1][y * w + x] = rr - g;
      ch[2][y * w + x] = b - ((rr + g) >> 1);
    }

  // gradient: per pixel the channel with the strongest gain-weighted L1 magnitude
  const int gq[3] = {16, pyround(p->chroma_gain * 16.0), pyround(p->chroma_gain * 16.0)};
  I16 m(n, 0), bgx(n, 0), bgy(n, 0);
  {
    I16 bl, gx, gy;
    for (int c = 0; c < 3; c++) {
      blur5(ch[c], w, h, bl);
      sobel(bl, w, h, gx, gy);
      for (int i = 0; i < n; i++) {
        int mm = std::min(32767, ((abs(gx[i]) + abs(gy[i])) * gq[c]) >> 4);
        if (c == 0 || mm > m[i]) { m[i] = (int16_t)mm; bgx[i] = gx[i]; bgy[i] = gy[i]; }
      }
    }
    for (int c = 0; c < 3; c++) I16().swap(ch[c]);
  }
  ff_timing_us[0] = (int32_t)(now_us() - tm); tm = now_us();

  // non-maximum suppression
  I16 nms(n, 0);
  for (int y = 1; y < h - 1; y++)
    for (int x = 1; x < w - 1; x++) {
      int i = y * w + x, c = m[i], ax = abs(bgx[i]), ay = abs(bgy[i]), n1, n2;
      if (ay * 1000 < ax * 414) { n1 = m[i - 1]; n2 = m[i + 1]; }
      else if (ax * 1000 < ay * 414) { n1 = m[i - w]; n2 = m[i + w]; }
      else if ((long)bgx[i] * bgy[i] > 0) { n1 = m[i - w - 1]; n2 = m[i + w + 1]; }
      else { n1 = m[i - w + 1]; n2 = m[i + w - 1]; }
      nms[i] = (c > n1 && c >= n2) ? c : 0;
    }
  I16().swap(m); I16().swap(bgx); I16().swap(bgy);

  // thresholds
  int lo = p->lo, hi = p->hi;
  if (p->mode_percentile) {
    std::vector<int> v;
    for (int i = 0; i < n; i++) if (nms[i] > 0) v.push_back(nms[i]);
    if (v.empty()) return FF_NONE;
    std::sort(v.begin(), v.end());
    int k = std::max(0, (int)ceil(p->hi_percentile / 100.0 * v.size()) - 1);
    hi = std::max(v[k], p->lo);
    lo = (int)(hi * p->lo_ratio);
  }

  // hysteresis + short fragments
  U8 mask(n, 0);
  {
    U8 cand(n, 0);
    for (int i = 0; i < n; i++) cand[i] = (nms[i] >= lo && nms[i] > 0) ? 255 : 0;
    I32 lab;
    int nl = label8(cand, w, h, lab);
    std::vector<int> size(nl, 0);
    std::vector<char> strong(nl, 0);
    for (int i = 0; i < n; i++)
      if (lab[i]) { size[lab[i]]++; if (nms[i] >= hi) strong[lab[i]] = 1; }
    for (int i = 0; i < n; i++)
      if (lab[i] && strong[lab[i]] && size[lab[i]] >= p->min_edge_px) mask[i] = 255;
  }
  I16().swap(nms);
  ff_timing_us[1] = (int32_t)(now_us() - tm); tm = now_us();

  // edges -> solid mask
  dilate(mask, w, h, p->close_r);
  fill_holes(mask, w, h);
  erode(mask, w, h, p->close_r);
  if (p->open_r > 0) { erode(mask, w, h, p->open_r); dilate(mask, w, h, p->open_r); }

  ff_timing_us[2] = (int32_t)(now_us() - tm); tm = now_us();
  // components + square filter
  I32 lab;
  int nl = label8(mask, w, h, lab);
  std::vector<Comp> comps(nl > 1 ? nl - 1 : 0);
  for (int i = 0; i < (int)comps.size(); i++) {
    comps[i].label = i + 1; comps[i].x1 = w; comps[i].y1 = h; comps[i].x2 = -1; comps[i].y2 = -1;
  }
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      int l = lab[y * w + x];
      if (!l) continue;
      Comp &c = comps[l - 1];
      c.x1 = std::min(c.x1, x); c.y1 = std::min(c.y1, y); c.x2 = std::max(c.x2, x + 1); c.y2 = std::max(c.y2, y + 1);
      c.area++;
      c.pts.push_back({(float)x, (float)y});
    }
  const float ppcm = p->px_per_cm / f;
  for (Comp &c : comps) {
    c.touches = c.x1 <= p->edge_margin_px || c.x2 >= w - p->edge_margin_px;
    c.has_rect = false;
    auto inside = [&](int x, int y) { return x >= 0 && y >= 0 && x < w && y < h && lab[y * w + x] == c.label; };
    if (c.area < p->min_area_px) {
      c.why = 1;
      if (c.area >= p->merge_min_px && !c.touches) { c.rect = min_area_rect(c.pts); c.has_rect = true; }
    } else if (c.area > p->max_area_px) {
      c.why = 2;
    } else if (!c.touches) {
      c.why = square_test(c, p, ppcm, (float)(p->open_r + 1), inside);
    } else {
      c.why = 0;
    }
  }

  ff_timing_us[3] = (int32_t)(now_us() - tm); tm = now_us();
  // fragment merging (union-find over close, aligned pieces; each group's hull tested as a square)
  if (p->merge_gap_cm > 0) {
    std::vector<int> idx;
    for (int i = 0; i < (int)comps.size(); i++) if (comps[i].has_rect && !comps[i].touches) idx.push_back(i);
    std::vector<int> parent(comps.size());
    for (int i = 0; i < (int)parent.size(); i++) parent[i] = i;
    auto root = [&](int k) { while (parent[k] != k) k = parent[k]; return k; };
    const float gap = p->merge_gap_cm * ppcm;
    for (size_t a = 0; a < idx.size(); a++)
      for (size_t b = a + 1; b < idx.size(); b++) {
        const Comp &A = comps[idx[a]], &B = comps[idx[b]];
        int g = std::max(std::max(std::max(A.x1, B.x1) - std::min(A.x2, B.x2), std::max(A.y1, B.y1) - std::min(A.y2, B.y2)), 0);
        float d = fabsf(angle90(A.rect) - angle90(B.rect));
        if (g <= gap && std::min(d, 90.0f - d) <= p->merge_max_deg) parent[root(idx[b])] = root(idx[a]);
      }
    std::vector<int> roots;
    for (int k : idx) {
      int rt = root(k);
      if (std::find(roots.begin(), roots.end(), rt) == roots.end()) roots.push_back(rt);
    }
    for (int rt : roots) {
      Comp g;
      int members = 0;
      for (int k : idx)
        if (root(k) == rt) { members++; g.pts.insert(g.pts.end(), comps[k].pts.begin(), comps[k].pts.end()); }
      if (members < 2) continue;
      g.H = hull(g.pts);
      std::vector<Pt>().swap(g.pts);
      double A2 = 0;
      g.x1 = w; g.y1 = h; g.x2 = -1; g.y2 = -1;
      for (size_t i = 0; i < g.H.size(); i++) {
        const Pt &a = g.H[i], &b = g.H[(i + 1) % g.H.size()];
        A2 += (double)a.x * b.y - (double)b.x * a.y;
        g.x1 = std::min(g.x1, (int)a.x); g.y1 = std::min(g.y1, (int)a.y);
        g.x2 = std::max(g.x2, (int)a.x + 1); g.y2 = std::max(g.y2, (int)a.y + 1);
      }
      g.area = (int)(fabs(A2) / 2) + 1;
      g.label = 0;
      g.touches = false;
      const std::vector<Pt> &HH = g.H;
      g.why = square_test(g, p, ppcm, (float)(p->open_r + 1), [&](int x, int y) { return in_convex(HH, x, y); });
      comps.push_back(g);
    }
  }
  r->n_blobs = (int)comps.size();
  ff_timing_us[4] = (int32_t)(now_us() - tm); tm = now_us();

  // best: complete squares first, then the largest
  int best = -1;
  for (int i = 0; i < (int)comps.size(); i++) {
    const Comp &c = comps[i];
    if (c.why) continue;
    if (best < 0 || (!c.touches && comps[best].touches) ||
        (c.touches == comps[best].touches && c.area > comps[best].area)) best = i;
  }
  if (best < 0) return FF_NONE;
  const Comp &b = comps[best];
  r->bbox[0] = wx1 + b.x1 * f; r->bbox[1] = wy1 + b.y1 * f; r->bbox[2] = wx1 + b.x2 * f; r->bbox[3] = wy1 + b.y2 * f;
  r->area_px = b.area * f * f;
  if (b.touches) return r->status = FF_PARTIAL;

  // corners (full-res, shrunk by inset) -> upright face
  r->size_cm[0] = (b.rect.w + 1) / ppcm;
  r->size_cm[1] = (b.rect.h + 1) / ppcm;
  r->angle = angle90(b.rect);
  Pt q[4];
  box_points(b.rect, b.rect.w + 1, b.rect.h + 1, q);
  float mx = 0, my = 0;
  for (int i = 0; i < 4; i++) { mx += q[i].x / 4; my += q[i].y / 4; }
  for (int i = 0; i < 4; i++) {
    float x = mx + (q[i].x - mx) * (1 - 2 * p->inset_frac), y = my + (q[i].y - my) * (1 - 2 * p->inset_frac);
    q[i].x = (x + 0.5f) * f - 0.5f + wx1;
    q[i].y = (y + 0.5f) * f - 0.5f + wy1;
  }
  // order: sort by angle around the centre (clockwise on screen), start at the min x+y corner
  float cx = 0, cy = 0;
  for (int i = 0; i < 4; i++) { cx += q[i].x / 4; cy += q[i].y / 4; }
  std::sort(q, q + 4, [&](const Pt &a, const Pt &c) { return atan2f(a.y - cy, a.x - cx) < atan2f(c.y - cy, c.x - cx); });
  int s = 0;
  for (int i = 1; i < 4; i++) if (q[i].x + q[i].y < q[s].x + q[s].y) s = i;
  for (int i = 0; i < 4; i++) { r->corners[2 * i] = q[(s + i) % 4].x; r->corners[2 * i + 1] = q[(s + i) % 4].y; }

  if (crop) {
    struct CropTimer { int64_t t0; ~CropTimer() { ff_timing_us[5] = (int32_t)(now_us() - t0); } } ct{now_us()};
    // affine: tl -> (-0.5,-0.5), tr -> (N-0.5,-0.5), br -> (N-0.5,N-0.5); sample the source with the inverse map
    const int N = p->input_size;
    float x0 = r->corners[0], y0 = r->corners[1];
    float ax = (r->corners[2] - x0) / N, ay = (r->corners[3] - y0) / N;          // source step per output x
    float bx = (r->corners[4] - r->corners[2]) / N, by = (r->corners[5] - r->corners[3]) / N;   // per output y
    for (int v = 0; v < N; v++)
      for (int u = 0; u < N; u++) {
        float su = u + 0.5f, sv = v + 0.5f;                     // output pixel centre, relative to (-0.5,-0.5)
        float sx = x0 + su * ax + sv * bx, sy = y0 + su * ay + sv * by;
        int ix = (int)floorf(sx), iy = (int)floorf(sy);
        float fx = sx - ix, fy = sy - iy;
        int xa = clampi(ix, 0, W - 1), xb = clampi(ix + 1, 0, W - 1), ya = clampi(iy, 0, H - 1), yb = clampi(iy + 1, 0, H - 1);
        for (int c = 0; c < 3; c++) {
          float pa = img[(size_t)ya * stride + xa * 3 + c], pb = img[(size_t)ya * stride + xb * 3 + c];
          float pc = img[(size_t)yb * stride + xa * 3 + c], pd = img[(size_t)yb * stride + xb * 3 + c];
          float val = (pa * (1 - fx) + pb * fx) * (1 - fy) + (pc * (1 - fx) + pd * fx) * fy;
          crop[(v * N + u) * 3 + c] = (uint8_t)clampi((int)(val + 0.5f), 0, 255);
        }
      }
  }
  return r->status = FF_OK;
}

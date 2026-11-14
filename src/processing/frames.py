"""Общая сетка неба, сложение экспозиций по эпохам, разностные кадры и «след» движения."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS

from Exposure import PIXEL_SCALE_ARCSEC, Exposure


# ---------------------------------------------------------------- сетка
def make_grid(center: SkyCoord, fov_arcmin: float,
              pix_arcsec: float = PIXEL_SCALE_ARCSEC) -> tuple[WCS, tuple[int, int]]:
    """Квадратная TAN-сетка: север вверх, восток влево, пиксель = родной пиксель SPHEREx.

    Мельче пиксель не делаем: это создало бы фальшивое разрешение.
    """
    n = max(5, int(round(fov_arcmin * 60 / pix_arcsec)))
    if n % 2 == 0:
        n += 1  # нечётный размер — центр ровно в пикселе
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]  # FITS 1-based
    w.wcs.cdelt = [-pix_arcsec / 3600, pix_arcsec / 3600]
    return w, (n, n)


def grid_header_json(wcs: WCS, shape) -> dict:
    return {"width_px": shape[1], "height_px": shape[0], "pixel_scale_arcsec": abs(wcs.wcs.cdelt[1]) * 3600,
            "wcs": {"CTYPE1": wcs.wcs.ctype[0], "CTYPE2": wcs.wcs.ctype[1],
                    "CRVAL1": float(wcs.wcs.crval[0]), "CRVAL2": float(wcs.wcs.crval[1]),
                    "CRPIX1": float(wcs.wcs.crpix[0]), "CRPIX2": float(wcs.wcs.crpix[1]),
                    "CDELT1": float(wcs.wcs.cdelt[0]), "CDELT2": float(wcs.wcs.cdelt[1])}}


def reproject_exposure(exp: Exposure, wcs: WCS, shape):
    """Перенос одной экспозиции на общую сетку.

    Возвращает (image, variance, bad, contam): яркость минус фон, дисперсию,
    маску «нельзя использовать» и маску «загрязнение» (блики, засветки — для слоя флагов).
    Маски переносятся ближайшим соседом: биты нельзя интерполировать.
    """
    from reproject import reproject_interp

    bad = exp.bad_mask()
    img = np.where(bad, np.nan, exp.background_subtracted()).astype(np.float64)
    var = np.where(bad, np.nan, exp.variance).astype(np.float64)
    # Слой FLAGS должен показывать и загрязнения, и пиксели, исключённые из науки
    # (например OVERFLOW / HOT / OUTLIER). SOURCE намеренно НЕ включён:
    # это информационный бит о наличии источника, а не плохой пиксель.
    from Exposure import CONTAMINATION_FLAGS, HARD_BAD_FLAGS
    contam = exp.flag_mask(CONTAMINATION_FLAGS + HARD_BAD_FLAGS)

    r_img, foot = reproject_interp((img, exp.wcs), wcs, shape_out=shape, order="bilinear")
    r_var, _ = reproject_interp((var, exp.wcs), wcs, shape_out=shape, order="bilinear")
    r_bad, _ = reproject_interp((bad.astype(float), exp.wcs), wcs, shape_out=shape,
                                order="nearest-neighbor")
    r_con, _ = reproject_interp((contam.astype(float), exp.wcs), wcs, shape_out=shape,
                                order="nearest-neighbor")
    outside = foot <= 0
    r_bad = (np.nan_to_num(r_bad, nan=1.0) > 0.5) | outside | ~np.isfinite(r_img) | ~np.isfinite(r_var)
    r_img[r_bad] = np.nan
    r_var[r_bad] = np.nan
    return r_img, r_var, r_bad, np.nan_to_num(r_con) > 0.5


# ---------------------------------------------------------------- кадры
@dataclass
class Frame:
    image: np.ndarray            # сумма эпохи на общей сетке, MJy/sr, фон вычтен; NaN = нет данных
    variance: np.ndarray
    contam: np.ndarray           # bool: загрязнённые/насыщенные пиксели (слой флагов)
    mjd: float                   # средняя дата эпохи
    mjd_min: float
    mjd_max: float
    detector: int
    wave_um: float               # эффективная λ в центре поля
    n_exp: int
    obs_ids: list[str] = field(default_factory=list)

    @property
    def date(self) -> str:
        return Time(self.mjd, format="mjd").to_value("iso", subfmt="date")


def coadd(exps: list[Exposure], wcs: WCS, shape, center: SkyCoord) -> Frame | None:
    """Взвешенное (1/σ²) среднее экспозиций одной эпохи на общей сетке."""
    num = np.zeros(shape)
    den = np.zeros(shape)
    contam = np.zeros(shape, bool)
    waves, used = [], []
    for e in exps:
        img, var, bad, con = reproject_exposure(e, wcs, shape)
        if bad.all():
            continue
        # остаточный фон (модель зодиакального света неидеальна) — убираем медианой по кадру
        _, med, _ = sigma_clipped_stats(img[~bad], sigma=3)
        img = img - med
        w = np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, var))
        num += np.where(bad, 0.0, img) * w
        den += w
        contam |= con  # включая насыщенные пиксели, которые выброшены из суммы
        if e.contains(center):
            waves.append(e.wavelength_at_sky(center)[0])
        used.append(e)
    if not used:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(den > 0, num / den, np.nan)
        variance = np.where(den > 0, 1.0 / den, np.nan)
    mjds = [e.mjd for e in used]
    return Frame(image=image, variance=variance, contam=contam,
                 mjd=float(np.mean(mjds)), mjd_min=float(min(mjds)), mjd_max=float(max(mjds)),
                 detector=used[0].detector,
                 wave_um=float(np.mean(waves)) if waves else float("nan"),
                 n_exp=len(used), obs_ids=[e.obs_id for e in used])


def difference(frames: list[Frame]) -> tuple[list[np.ndarray], np.ndarray]:
    """Разностные кадры в единицах σ: (кадр − медианный шаблон) / шум.

    Движущийся объект даёт пару: «+» там, где он сейчас, «−» там, где был в среднем.
    """
    cube = np.array([f.image for f in frames])
    template = np.nanmedian(cube, axis=0)
    out = []
    for f in frames:
        with np.errstate(invalid="ignore"):
            # шум шаблона медианы ≈ 1.25·σ/√N; добавляем его к шуму кадра
            var_t = np.nanmedian([g.variance for g in frames], axis=0) * (1.57 / max(len(frames), 1))
            out.append((f.image - template) / np.sqrt(f.variance + var_t))
    return out, template


def trail(diff_sigma: list[np.ndarray], clip: float = 8.0) -> np.ndarray:
    """«След» движения на одной картинке (RGB 0..1).

    Положительные остатки каждой эпохи окрашиваются цветом по времени (viridis: старые —
    фиолетовые, новые — жёлтые) с прозрачностью α от 0.35 до 1.0 и СКЛАДЫВАЮТСЯ.
    Обычное наложение слоёв не годится: слой с α=1 полностью закрыл бы все предыдущие.
    """
    from matplotlib import colormaps

    cmap = colormaps["viridis"]
    n = len(diff_sigma)
    rgb = np.zeros((*diff_sigma[0].shape, 3))
    for i, d in enumerate(diff_sigma):
        t = i / max(n - 1, 1)
        alpha = 0.35 + 0.65 * t
        pos = np.clip(np.nan_to_num(d), 0, clip) / clip
        rgb += alpha * pos[..., None] * np.array(cmap(t)[:3])
    return np.clip(rgb, 0, 1)
  

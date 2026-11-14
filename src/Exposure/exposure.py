"""Exposure: один кадр SPHEREx с одного детектора (Level 2 spectral image).

Формат файла (по документации IRSA, SPHEREx QR3/DR1):
  HDU 0  PRIMARY   — без данных
  HDU 1  IMAGE     — поверхностная яркость, MJy/sr, 2040x2040, ЗОДИАКАЛЬНЫЙ ФОН НЕ ВЫЧТЕН
  HDU 2  FLAGS     — битовая маска (int32); имена битов в заголовке: MSKNnn / MSKDnn
  HDU 3  VARIANCE  — (MJy/sr)^2
  HDU 4  ZODI      — модель зодиакального фона, MJy/sr
  HDU 5  EPSF/PSF  — PSF (имя зависит от релиза)
  HDU 6  WCS-WAVE  — таблица «пиксель -> (λ, ширина полосы)», читается через WCS key='W'
Серверные вырезки IRSA (?center=..&size=..) имеют те же HDU, поэтому этот класс
одинаково работает и с полными файлами, и с вырезками.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

PIXEL_SCALE_ARCSEC = 6.15  # размер пикселя спектрального изображения SPHEREx

# Диапазоны детекторов (мкм) и спектральное разрешение R = λ/Δλ.
DETECTOR_BANDS: dict[int, tuple[float, float, int]] = {
    1: (0.75, 1.09, 39),
    2: (1.10, 1.62, 41),
    3: (1.63, 2.41, 41),
    4: (2.42, 3.82, 35),
    5: (3.83, 4.41, 112),
    6: (4.42, 5.00, 128),
}

# Запасная таблица битов на случай, если в заголовке FLAGS нет ключей MSKN*.
# Основной источник — заголовок конкретного файла (см. parse_flag_bits).
DEFAULT_FLAG_BITS: dict[str, int] = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "CROSSTALK": 20, "SOURCE": 21,
    "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24, "STREAK": 25, "BLOOM": 26,
    "SNOWBALL": 27, "HALO": 28,
}

# Пиксели с этими флагами не используем НИКОГДА (данных нет или они испорчены).
HARD_BAD_FLAGS = (
    "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA",
    "HOT", "COLD", "NONLINEAR", "PERSIST", "OUTLIER",
)
# Загрязнения: данные есть, но им нельзя доверять для точной фотометрии/центроида.
# Показываем пользователю отдельным слоем.
CONTAMINATION_FLAGS = (
    "CROSSTALK", "GHOST", "GHOST_FPA", "GHOST_EXT", "STREAK", "BLOOM", "SNOWBALL", "HALO",
)
# ВНИМАНИЕ: "SOURCE" означает «здесь есть источник» — его НЕЛЬЗЯ маскировать,
# иначе мы выкинем саму цель.


def parse_flag_bits(flags_header: fits.Header) -> dict[str, int]:
    """Читает имена битов из заголовка FLAGS: MSKN<бит> = имя."""
    bits: dict[str, int] = {}
    for key in flags_header:
        if key.startswith("MSKN"):
            try:
                bits[str(flags_header[key]).strip()] = int(key[4:])
            except ValueError:
                continue
    return bits or dict(DEFAULT_FLAG_BITS)


@dataclass
class Exposure:
    """Контейнер одного кадра. Ничего не обрабатывает — только хранит и отвечает на вопросы."""

    image: np.ndarray            # MJy/sr, фон НЕ вычтен
    variance: np.ndarray         # (MJy/sr)^2
    flags: np.ndarray            # int32, битовая маска
    zodi: np.ndarray | None      # модель зодиакального фона, MJy/sr
    wcs: WCS                     # пиксель <-> (RA, Dec)
    wave_wcs: WCS | None         # пиксель -> (λ, ширина полосы); None если не удалось прочитать
    detector: int                # 1..6
    mjd: float                   # время середины экспозиции (MJD)
    obs_id: str
    flag_bits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FLAG_BITS))
    source: str = ""             # путь или URL, откуда загружено
    header: fits.Header | None = field(default=None, repr=False)  # заголовок IMAGE

    # ---------- геометрия ----------
    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape

    def sky_to_pixel(self, coord: SkyCoord) -> tuple[float, float]:
        """(RA, Dec) -> (x, y), 0-based пиксели numpy: image[y, x]."""
        x, y = self.wcs.world_to_pixel(coord)
        return float(x), float(y)

    def pixel_to_sky(self, x: float, y: float) -> SkyCoord:
        return self.wcs.pixel_to_world(x, y)

    def contains(self, coord: SkyCoord, margin_px: float = 0.0) -> bool:
        """Попадает ли точка неба в кадр (с отступом от края)."""
        x, y = self.sky_to_pixel(coord)
        ny, nx = self.shape
        return (margin_px - 0.5 <= x <= nx - 0.5 - margin_px) and (
            margin_px - 0.5 <= y <= ny - 0.5 - margin_px)

    # ---------- длина волны ----------
    def wavelength_at(self, x: float, y: float) -> tuple[float, float]:
        """Центральная длина волны и ширина полосы (мкм) в пикселе (x, y)."""
        if self.wave_wcs is None:
            lo, hi, _ = DETECTOR_BANDS[self.detector]
            return 0.5 * (lo + hi), hi - lo  # грубая оценка: середина диапазона детектора
        wl, bw = self.wave_wcs.pixel_to_world(x, y)
        return float(wl.to_value(u.um)), float(bw.to_value(u.um))

    def wavelength_at_sky(self, coord: SkyCoord) -> tuple[float, float]:
        return self.wavelength_at(*self.sky_to_pixel(coord))

    def wave_map(self) -> np.ndarray:
        """λ (мкм) для каждого пикселя. Тяжело для 2040x2040 — вызывать только на вырезках."""
        if self.wave_wcs is None:
            lo, hi, _ = DETECTOR_BANDS[self.detector]
            return np.full(self.shape, 0.5 * (lo + hi), dtype=np.float32)
        y, x = np.indices(self.shape)
        wl, _ = self.wave_wcs.pixel_to_world(x, y)
        return wl.to_value(u.um).astype(np.float32)

    # ---------- флаги ----------
    def flag_mask(self, names) -> np.ndarray:
        """True там, где поднят хотя бы один из флагов `names` (неизвестные имена игнорируются)."""
        bits = 0
        for n in names:
            if n in self.flag_bits:
                bits |= 1 << self.flag_bits[n]
        return (self.flags & bits) != 0

    def bad_mask(self, include_contamination: bool = False) -> np.ndarray:
        """True = пиксель нельзя использовать. Плюс NaN/inf и неположительная дисперсия."""
        names = HARD_BAD_FLAGS + (CONTAMINATION_FLAGS if include_contamination else ())
        return (self.flag_mask(names) | ~np.isfinite(self.image)
                | ~np.isfinite(self.variance) | (self.variance <= 0))

    def flag_names_at(self, x: int, y: int) -> list[str]:
        """Какие флаги подняты в пикселе — для подсказки в UI."""
        v = int(self.flags[y, x])
        return [n for n, b in sorted(self.flag_bits.items(), key=lambda t: t[1]) if v >> b & 1]

    # ---------- яркость ----------
    def background_subtracted(self) -> np.ndarray:
        """Снимок минус модель зодиакального света (если её нет — минус медиана)."""
        if self.zodi is not None:
            # Модель ZODI задаёт фон для каждого пикселя; не заменяем её медианой.
            return self.image - self.zodi
        # Если ZODI отсутствует, оцениваем фон по конечным пикселям,
        # исключая hard-bad флаги. Не вызываем bad_mask(): VARIANCE может
        # отсутствовать в допустимом cutout и тогда вся маска стала бы True.
        usable = np.isfinite(self.image) & ~self.flag_mask(HARD_BAD_FLAGS)
        if not np.any(usable):
            return np.full(self.shape, np.nan, dtype=np.float32)
        return self.image - np.nanmedian(self.image[usable])

    @staticmethod
    def mjysr_to_ujy_per_pixel() -> float:
        """Множитель MJy/sr -> μJy для одного пикселя (умножить на телесный угол пикселя)."""
        omega_sr = (PIXEL_SCALE_ARCSEC * u.arcsec).to_value(u.rad) ** 2
        return 1e6 * 1e6 * omega_sr  # MJy -> Jy (1e6), Jy -> μJy (1e6)

    def __repr__(self) -> str:  # компактно, без массивов
        return (f"Exposure(obs_id={self.obs_id!r}, detector={self.detector}, "
                f"mjd={self.mjd:.5f}, shape={self.shape}, wave={'yes' if self.wave_wcs else 'no'})")

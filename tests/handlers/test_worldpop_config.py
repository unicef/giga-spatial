"""
Tests for resolve_dataset_category and DemographicPathFilter.

Run with:
    pytest tests/handlers/test_worldpop_config.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gigaspatial.handlers.worldpop import (
    AVAILABLE_YEARS_GR1,
    AVAILABLE_YEARS_GR2,
    AVAILABLE_RESOLUTIONS,
    DemographicPathFilter,
    resolve_dataset_category,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(
    release="GR1",
    project="pop",
    year=2020,
    resolution=100,
    un_adjusted=False,
    constrained=False,
    school_age=False,
    under_18=False,
    dug_level="L1",
):
    """Thin wrapper so tests only override what they care about."""
    return resolve_dataset_category(
        release=release,
        project=project,
        year=year,
        resolution=resolution,
        un_adjusted=un_adjusted,
        constrained=constrained,
        school_age=school_age,
        under_18=under_18,
        dug_level=dug_level,
    )


def _paths(*names: str) -> list[Path]:
    """Create fake Path objects from bare filenames."""
    return [Path(n) for n in names]


# ─────────────────────────────────────────────────────────────────────────────
# resolve_dataset_category — happy-path table
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveCategoryHappyPath:
    """Every valid combination resolves to the expected category string."""

    # ── DUG ──────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("year", [2015, 2020, 2030])
    def test_dug_returns_correct_category(self, year):
        category, _, _ = _resolve(project="degree_of_urbanization", year=year)
        assert category == "dug_g2_v1"

    # ── GR2 / pop ────────────────────────────────────────────────────────────

    def test_gr2_pop_100m(self):
        category, _, _ = _resolve(
            release="GR2", project="pop", year=2020, resolution=100, constrained=True
        )
        assert category == "G2_CN_POP_R25A_100m"

    def test_gr2_pop_1km(self):
        category, _, _ = _resolve(
            release="GR2", project="pop", year=2020, resolution=1000, constrained=True
        )
        assert category == "G2_CN_POP_R25A_1km"

    # ── GR2 / age_structures ─────────────────────────────────────────────────

    def test_gr2_age_100m(self):
        category, _, _ = _resolve(
            release="GR2",
            project="age_structures",
            year=2020,
            resolution=100,
            constrained=True,
        )
        assert category == "G2_CN_Age_R25A_100m"

    def test_gr2_age_1km(self):
        category, _, _ = _resolve(
            release="GR2",
            project="age_structures",
            year=2020,
            resolution=1000,
            constrained=True,
        )
        assert category == "G2_CN_Age_R25A_1km"

    def test_gr2_under_18(self):
        category, _, _ = _resolve(
            release="GR2",
            project="age_structures",
            year=2020,
            resolution=100,
            constrained=True,
            under_18=True,
        )
        assert category == "G2_Age_U18_R25A_100m"

    # ── GR1 / pop — unconstrained ─────────────────────────────────────────────

    @pytest.mark.parametrize(
        "resolution,un_adjusted,expected",
        [
            (100, False, "wpgp"),
            (100, True, "wpgpunadj"),
            (1000, False, "wpic1km"),
            (1000, True, "wpicuadj1km"),
        ],
    )
    def test_gr1_pop_unconstrained(self, resolution, un_adjusted, expected):
        category, _, _ = _resolve(resolution=resolution, un_adjusted=un_adjusted)
        assert category == expected

    # ── GR1 / pop — constrained ───────────────────────────────────────────────

    @pytest.mark.parametrize(
        "un_adjusted,expected",
        [
            (False, "cic2020_100m"),
            (True, "cic2020_UNadj_100m"),
        ],
    )
    def test_gr1_pop_constrained(self, un_adjusted, expected):
        category, _, _ = _resolve(
            year=2020, resolution=100, constrained=True, un_adjusted=un_adjusted
        )
        assert category == expected

    # ── GR1 / age_structures — school_age ─────────────────────────────────────

    def test_gr1_school_age(self):
        category, _, _ = _resolve(project="age_structures", school_age=True)
        assert category == "sapya1km"

    # ── GR1 / age_structures — unconstrained ──────────────────────────────────

    def test_gr1_age_unconstrained(self):
        category, _, _ = _resolve(project="age_structures")
        assert category == "aswpgp"

    # ── GR1 / age_structures — constrained + UN-adjusted ─────────────────────

    def test_gr1_age_constrained_unadj(self):
        category, _, _ = _resolve(
            project="age_structures", year=2020, constrained=True, un_adjusted=True
        )
        assert category == "ascicua_2020"

    # ── GR1 / age_structures — constrained, no UN adjustment ─────────────────

    def test_gr1_age_constrained_no_unadj(self):
        category, _, _ = _resolve(
            project="age_structures", year=2020, constrained=True, un_adjusted=False
        )
        assert category == "ascic_2020"


# ─────────────────────────────────────────────────────────────────────────────
# resolve_dataset_category — normalised field overrides
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveCategoryNormalization:
    """Verify that the normalized_fields dict carries the right corrections."""

    def test_gr2_under_18_forces_100m(self):
        _, normalized, _ = _resolve(
            release="GR2",
            project="age_structures",
            year=2020,
            resolution=1000,
            constrained=True,
            under_18=True,
        )
        assert normalized.get("resolution") == 100

    def test_gr2_un_adjusted_cleared(self):
        _, normalized, _ = _resolve(
            release="GR2",
            project="pop",
            year=2020,
            resolution=100,
            constrained=True,
            un_adjusted=True,
        )
        assert normalized.get("un_adjusted") is False

    def test_gr1_pop_constrained_forces_2020(self):
        _, normalized, _ = _resolve(year=2018, resolution=100, constrained=True)
        assert normalized.get("year") == 2020

    def test_gr1_pop_constrained_forces_100m(self):
        _, normalized, _ = _resolve(year=2020, resolution=1000, constrained=True)
        assert normalized.get("resolution") == 100

    def test_gr1_school_age_forces_1km(self):
        _, normalized, _ = _resolve(
            project="age_structures", year=2020, resolution=100, school_age=True
        )
        assert normalized.get("resolution") == 1000

    def test_gr1_school_age_forces_2020(self):
        _, normalized, _ = _resolve(
            project="age_structures", year=2015, resolution=1000, school_age=True
        )
        assert normalized.get("year") == 2020

    def test_gr1_school_age_clears_constrained(self):
        _, normalized, _ = _resolve(
            project="age_structures",
            year=2020,
            resolution=1000,
            school_age=True,
            constrained=True,
        )
        assert normalized.get("constrained") is False

    def test_gr1_school_age_clears_un_adjusted(self):
        _, normalized, _ = _resolve(
            project="age_structures",
            year=2020,
            resolution=1000,
            school_age=True,
            un_adjusted=True,
        )
        assert normalized.get("un_adjusted") is False

    def test_gr1_age_constrained_unadj_forces_2020(self):
        _, normalized, _ = _resolve(
            project="age_structures",
            year=2018,
            constrained=True,
            un_adjusted=True,
        )
        assert normalized.get("year") == 2020

    def test_gr1_age_unconstrained_clears_un_adjusted(self):
        _, normalized, _ = _resolve(
            project="age_structures", un_adjusted=True, constrained=False
        )
        assert normalized.get("un_adjusted") is False

    def test_dug_clears_school_age(self):
        _, normalized, _ = _resolve(
            project="degree_of_urbanization", year=2020, school_age=True
        )
        assert normalized.get("school_age") is False

    def test_dug_clears_under_18(self):
        _, normalized, _ = _resolve(
            project="degree_of_urbanization", year=2020, under_18=True
        )
        assert normalized.get("under_18") is False

    def test_dug_clears_un_adjusted(self):
        _, normalized, _ = _resolve(
            project="degree_of_urbanization", year=2020, un_adjusted=True
        )
        assert normalized.get("un_adjusted") is False

    def test_no_spurious_overrides(self):
        """A clean GR2 pop request should produce no normalised field updates."""
        _, normalized, _ = _resolve(
            release="GR2",
            project="pop",
            year=2020,
            resolution=100,
            constrained=True,
        )
        assert normalized == {}


# ─────────────────────────────────────────────────────────────────────────────
# resolve_dataset_category — warnings
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveCategoryWarnings:
    """Warnings are emitted as strings, not silently swallowed."""

    def test_gr2_un_adjusted_warns(self):
        _, _, warnings = _resolve(
            release="GR2",
            project="pop",
            year=2020,
            resolution=100,
            constrained=True,
            un_adjusted=True,
        )
        assert any("un_adjusted" in w for w in warnings)

    def test_gr2_under_18_resolution_warns(self):
        _, _, warnings = _resolve(
            release="GR2",
            project="age_structures",
            year=2020,
            resolution=1000,
            constrained=True,
            under_18=True,
        )
        assert any("100m" in w for w in warnings)

    def test_dug_school_age_warns(self):
        _, _, warnings = _resolve(
            project="degree_of_urbanization", year=2020, school_age=True
        )
        assert any("school_age" in w for w in warnings)

    def test_gr1_school_age_resolution_warns(self):
        _, _, warnings = _resolve(
            project="age_structures", year=2020, resolution=100, school_age=True
        )
        assert any("1km" in w for w in warnings)

    def test_gr1_constrained_pop_year_warns(self):
        _, _, warnings = _resolve(year=2018, resolution=100, constrained=True)
        assert any("2020" in w for w in warnings)

    def test_clean_config_no_warnings(self):
        _, _, warnings = _resolve()
        assert warnings == []


# ─────────────────────────────────────────────────────────────────────────────
# resolve_dataset_category — invalid combinations raise ValueError
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveCategoryErrors:
    """Invalid parameter combinations must raise ValueError, not silently corrupt."""

    def test_gr1_year_out_of_range(self):
        with pytest.raises(ValueError, match="GR1"):
            _resolve(year=2025)

    def test_gr2_year_out_of_range(self):
        with pytest.raises(ValueError, match="GR2"):
            _resolve(release="GR2", project="pop", year=2010, constrained=True)

    def test_gr2_unconstrained_raises(self):
        with pytest.raises(ValueError, match="constrained"):
            _resolve(
                release="GR2",
                project="pop",
                year=2020,
                resolution=100,
                constrained=False,
            )

    def test_gr2_school_age_raises(self):
        with pytest.raises(ValueError, match="GR2"):
            _resolve(
                release="GR2",
                project="age_structures",
                year=2020,
                resolution=1000,
                constrained=True,
                school_age=True,
            )

    def test_gr2_pop_under_18_raises(self):
        with pytest.raises(ValueError, match="age_structures"):
            _resolve(
                release="GR2",
                project="pop",
                year=2020,
                resolution=100,
                constrained=True,
                under_18=True,
            )

    def test_gr1_pop_school_age_raises(self):
        with pytest.raises(ValueError, match="age_structures"):
            _resolve(project="pop", school_age=True)

    def test_gr1_pop_under_18_raises(self):
        with pytest.raises(ValueError, match="GR2"):
            _resolve(project="pop", under_18=True)

    def test_gr1_age_constrained_not_2020_raises(self):
        with pytest.raises(ValueError, match="2020"):
            _resolve(
                project="age_structures", year=2018, constrained=True, un_adjusted=False
            )

    def test_dug_year_out_of_range_raises(self):
        with pytest.raises(ValueError, match="2015-2030"):
            _resolve(project="degree_of_urbanization", year=2010)


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — construction
# ─────────────────────────────────────────────────────────────────────────────


class TestDemographicPathFilterConstruction:

    def test_from_kwargs_no_args(self):
        f = DemographicPathFilter.from_kwargs()
        assert f.sex_filters is None
        assert f.level_filters is None
        assert f.ages_filter is None
        assert f.min_age is None
        assert f.max_age is None

    def test_from_kwargs_sex_string(self):
        f = DemographicPathFilter.from_kwargs(sex="f")
        assert f.sex_filters == frozenset({"F"})

    def test_from_kwargs_sex_list(self):
        f = DemographicPathFilter.from_kwargs(sex=["M", "f"])
        assert f.sex_filters == frozenset({"M", "F"})

    def test_from_kwargs_education_level_alias(self):
        f = DemographicPathFilter.from_kwargs(level="primary")
        assert f.level_filters == frozenset({"PRIMARY"})

    def test_from_kwargs_education_level_kwarg(self):
        f = DemographicPathFilter.from_kwargs(education_level=["PRIMARY", "secondary"])
        assert f.level_filters == frozenset({"PRIMARY", "SECONDARY"})

    def test_from_kwargs_ages_set(self):
        f = DemographicPathFilter.from_kwargs(ages={5, 10, 15})
        assert f.ages_filter == frozenset({5, 10, 15})

    def test_from_kwargs_min_max_age(self):
        f = DemographicPathFilter.from_kwargs(min_age=5, max_age=17)
        assert f.min_age == 5
        assert f.max_age == 17

    def test_from_kwargs_min_max_age_as_string(self):
        """Accepts string-typed age bounds (common from CLI/config sources)."""
        f = DemographicPathFilter.from_kwargs(min_age="5", max_age="17")
        assert f.min_age == 5
        assert f.max_age == 17

    def test_frozen_is_immutable(self):
        f = DemographicPathFilter.from_kwargs(sex="M")
        with pytest.raises((AttributeError, TypeError)):
            f.sex_filters = frozenset({"F"})  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — has_filters
# ─────────────────────────────────────────────────────────────────────────────


class TestDemographicPathFilterHasFilters:

    def test_empty_has_no_filters(self):
        assert DemographicPathFilter.from_kwargs().has_filters is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sex": "M"},
            {"level": "PRIMARY"},
            {"ages": [5, 10]},
            {"min_age": 5},
            {"max_age": 18},
        ],
    )
    def test_any_criterion_returns_true(self, kwargs):
        assert DemographicPathFilter.from_kwargs(**kwargs).has_filters is True


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — filter_paths: under-18 files
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterPathsUnder18:
    """Pattern: ISO3_SEX_Under_18_YEAR.tif  (T = total, F = female, M = male)"""

    PATHS = _paths(
        "RWA_T_Under_18_2020.tif",
        "RWA_F_Under_18_2020.tif",
        "RWA_M_Under_18_2020.tif",
    )

    def _filter(self, **kwargs):
        return DemographicPathFilter.from_kwargs(**kwargs).filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )

    def test_default_no_sex_filter_returns_total_only(self):
        result = DemographicPathFilter().filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )
        assert [p.name for p in result] == ["RWA_T_Under_18_2020.tif"]

    def test_explicit_sex_T(self):
        result = self._filter(sex="T")
        assert [p.name for p in result] == ["RWA_T_Under_18_2020.tif"]

    def test_explicit_sex_F(self):
        result = self._filter(sex="F")
        assert [p.name for p in result] == ["RWA_F_Under_18_2020.tif"]

    def test_explicit_sex_M(self):
        result = self._filter(sex="M")
        assert [p.name for p in result] == ["RWA_M_Under_18_2020.tif"]

    def test_multi_sex_filter(self):
        result = self._filter(sex=["F", "M"])
        names = {p.name for p in result}
        assert names == {"RWA_F_Under_18_2020.tif", "RWA_M_Under_18_2020.tif"}

    def test_unknown_sex_returns_empty(self):
        result = self._filter(sex="X")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — filter_paths: school-age files
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterPathsSchoolAge:
    """
    Patterns:
      Combined:  ISO3_F_M_PRIMARY_YEAR.tif   (sex=F_M)
      Gendered:  ISO3_F_PRIMARY_YEAR.tif     (sex=F)
                 ISO3_M_SECONDARY_YEAR.tif   (sex=M)
    """

    PATHS = _paths(
        "RWA_F_M_PRIMARY_2020.tif",
        "RWA_F_M_SECONDARY_2020.tif",
        "RWA_F_PRIMARY_2020.tif",
        "RWA_M_PRIMARY_2020.tif",
        "RWA_F_SECONDARY_2020.tif",
        "RWA_M_SECONDARY_2020.tif",
    )

    def _filter(self, **kwargs):
        return DemographicPathFilter.from_kwargs(**kwargs).filter_paths(
            self.PATHS, project="age_structures", school_age=True
        )

    def test_default_no_sex_returns_fm_combined_only(self):
        result = DemographicPathFilter().filter_paths(
            self.PATHS, project="age_structures", school_age=True
        )
        names = {p.name for p in result}
        assert names == {"RWA_F_M_PRIMARY_2020.tif", "RWA_F_M_SECONDARY_2020.tif"}

    def test_sex_F_M(self):
        result = self._filter(sex="F_M")
        names = {p.name for p in result}
        assert names == {"RWA_F_M_PRIMARY_2020.tif", "RWA_F_M_SECONDARY_2020.tif"}

    def test_sex_F(self):
        result = self._filter(sex="F")
        names = {p.name for p in result}
        assert names == {"RWA_F_PRIMARY_2020.tif", "RWA_F_SECONDARY_2020.tif"}

    def test_sex_M(self):
        result = self._filter(sex="M")
        names = {p.name for p in result}
        assert names == {"RWA_M_PRIMARY_2020.tif", "RWA_M_SECONDARY_2020.tif"}

    def test_level_filter_primary(self):
        result = self._filter(level="PRIMARY")
        names = {p.name for p in result}
        assert names == {
            "RWA_F_M_PRIMARY_2020.tif",
        }

    def test_level_filter_secondary(self):
        result = self._filter(level="SECONDARY")
        names = {p.name for p in result}
        assert names == {
            "RWA_F_M_SECONDARY_2020.tif",
        }

    def test_sex_and_level_combined(self):
        result = self._filter(sex="F_M", level="PRIMARY")
        assert [p.name for p in result] == ["RWA_F_M_PRIMARY_2020.tif"]

    def test_unknown_level_returns_empty(self):
        result = self._filter(level="TERTIARY")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — filter_paths: standard age-band files
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterPathsAgeBand:
    """Standard age-band pattern: ISO3_SEX_AGE_YEAR.tif"""

    PATHS = _paths(
        "RWA_F_0_2020.tif",
        "RWA_M_0_2020.tif",
        "RWA_F_5_2020.tif",
        "RWA_M_5_2020.tif",
        "RWA_F_10_2020.tif",
        "RWA_M_10_2020.tif",
        "RWA_F_15_2020.tif",
        "RWA_M_15_2020.tif",
        "RWA_F_20_2020.tif",
        "RWA_M_20_2020.tif",
    )

    def _filter(self, **kwargs):
        return DemographicPathFilter.from_kwargs(**kwargs).filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )

    def test_no_filter_returns_all(self):
        result = DemographicPathFilter().filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )
        assert len(result) == len(self.PATHS)

    def test_sex_filter_female(self):
        result = self._filter(sex="F")
        assert all("_F_" in p.name for p in result)
        assert len(result) == 5

    def test_sex_filter_male(self):
        result = self._filter(sex="M")
        assert all("_M_" in p.name for p in result)
        assert len(result) == 5

    def test_ages_filter_exact(self):
        result = self._filter(ages=[5, 10])
        ages = {int(p.name.split("_")[2]) for p in result}
        assert ages == {5, 10}

    def test_min_age(self):
        result = self._filter(min_age=10)
        ages = {int(p.name.split("_")[2]) for p in result}
        assert all(a >= 10 for a in ages)

    def test_max_age(self):
        result = self._filter(max_age=10)
        ages = {int(p.name.split("_")[2]) for p in result}
        assert all(a <= 10 for a in ages)

    def test_min_max_age_range(self):
        result = self._filter(min_age=5, max_age=15)
        ages = {int(p.name.split("_")[2]) for p in result}
        assert ages == {5, 10, 15}

    def test_sex_and_age_range_combined(self):
        result = self._filter(sex="F", min_age=5, max_age=10)
        names = {p.name for p in result}
        assert names == {"RWA_F_5_2020.tif", "RWA_F_10_2020.tif"}

    def test_no_overlap_ages_returns_empty(self):
        result = self._filter(ages=[99])
        assert result == []

    def test_ages_filter_as_string_values(self):
        """ages values that come in as strings should still match int file stems."""
        result = self._filter(ages=["5", "10"])
        ages = {int(p.name.split("_")[2]) for p in result}
        assert ages == {5, 10}

    def test_unparseable_age_skipped_with_age_filter(self, caplog):
        """Files whose age can't be parsed are skipped when age filters are active."""
        bad_paths = _paths("RWA_F_unknown_2020.tif")
        import logging

        with caplog.at_level(logging.WARNING):
            result = DemographicPathFilter.from_kwargs(min_age=0).filter_paths(
                bad_paths, project="age_structures", school_age=False
            )
        assert result == []
        assert any("age" in msg.lower() for msg in caplog.messages)


# ─────────────────────────────────────────────────────────────────────────────
# DemographicPathFilter — filter_paths: mixed corpus
# ─────────────────────────────────────────────────────────────────────────────


class TestFilterPathsMixedCorpus:
    """
    When a directory contains both school-age and standard age-band files,
    each file type should be classified independently.
    """

    PATHS = _paths(
        "RWA_F_M_PRIMARY_2020.tif",  # school-age combined
        "RWA_F_5_2020.tif",  # standard age band
        "RWA_T_Under_18_2020.tif",  # under-18 total
    )

    def test_no_filter_returns_all_standard_and_under18_total(self):
        result = DemographicPathFilter().filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )
        names = {p.name for p in result}
        # Standard files pass through; under-18 defaults to T; school-age gets F_M default
        # but school_age=False means school-age files' sex check against F_M still runs
        assert "RWA_F_5_2020.tif" in names
        assert "RWA_T_Under_18_2020.tif" in names

    def test_sex_F_excludes_under18_total_and_fm_combined(self):
        result = DemographicPathFilter.from_kwargs(sex="F").filter_paths(
            self.PATHS, project="age_structures", school_age=False
        )
        names = {p.name for p in result}
        assert "RWA_F_5_2020.tif" in names
        assert "RWA_T_Under_18_2020.tif" not in names
        assert "RWA_F_M_PRIMARY_2020.tif" not in names

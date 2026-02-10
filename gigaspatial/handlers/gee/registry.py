# gigaspatial/handlers/gee/registry.py

from typing import Dict, Optional, List
from pathlib import Path
import json
from .datasets.base import GEEDatasetEntry
from .datasets.builtin import BUILTIN_DATASETS


class GEEDatasetRegistry:
    """
    Manages GEE dataset registry with built-in and custom datasets.

    Automatically loads built-in datasets and allows users to add custom ones.
    """

    def __init__(self, custom_registry: Optional[Dict] = None):
        """
        Initialize registry with built-in datasets.

        Parameters
        ----------
        custom_registry : dict, optional
            User-provided custom dataset registry to merge with built-in
        """
        self._registry: Dict[str, GEEDatasetEntry] = {}

        # Load built-in datasets
        self._load_builtin()

        # Load custom datasets if provided
        if custom_registry:
            self.add_custom_datasets(custom_registry)

    def _load_builtin(self):
        """Load built-in datasets from package."""
        self._registry.update(BUILTIN_DATASETS)

    def add_custom_datasets(self, custom_datasets: Dict):
        """
        Add or override datasets with custom definitions.

        Parameters
        ----------
        custom_datasets : dict
            Dictionary of dataset_id -> GEEDatasetEntry or dict
        """
        for dataset_id, dataset_info in custom_datasets.items():
            if isinstance(dataset_info, GEEDatasetEntry):
                self._registry[dataset_id] = dataset_info
            elif isinstance(dataset_info, dict):
                # Convert dict to GEEDatasetEntry
                self._registry[dataset_id] = GEEDatasetEntry(**dataset_info)
            else:
                raise TypeError(
                    f"Dataset entry must be GEEDatasetEntry or dict, got {type(dataset_info)}"
                )

    def get(self, dataset_id: str) -> GEEDatasetEntry:
        """
        Get dataset entry by ID.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier

        Returns
        -------
        GEEDatasetEntry
            Dataset metadata entry
        """
        if dataset_id not in self._registry:
            raise ValueError(
                f"Dataset '{dataset_id}' not found in registry. "
                f"Available: {self.list_datasets()}"
            )
        return self._registry[dataset_id]

    def list_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        return sorted(self._registry.keys())

    def search(self, keyword: str) -> List[str]:
        """
        Search datasets by keyword in name or description.

        Parameters
        ----------
        keyword : str
            Search keyword

        Returns
        -------
        list
            Matching dataset IDs
        """
        keyword = keyword.lower()
        matches = []
        for dataset_id, entry in self._registry.items():
            if keyword in dataset_id.lower():
                matches.append(dataset_id)
            elif entry.description and keyword in entry.description.lower():
                matches.append(dataset_id)
        return matches

    def get_datasets_by_source(self, source: str) -> List[str]:
        """Get all datasets from a specific source (e.g., 'NOAA', 'NASA')."""
        return [
            dataset_id
            for dataset_id, entry in self._registry.items()
            if entry.source and entry.source.lower() == source.lower()
        ]

    def get_datasets_by_cadence(self, cadence: str) -> List[str]:
        """Get datasets by temporal cadence."""
        return [
            dataset_id
            for dataset_id, entry in self._registry.items()
            if entry.temporal_cadence == cadence
        ]

    def to_dict(self) -> Dict:
        """Export registry as dictionary."""
        return {
            dataset_id: entry.to_dict() for dataset_id, entry in self._registry.items()
        }

    def save_to_json(self, filepath: str):
        """Save registry to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "GEEDatasetRegistry":
        """Load registry from JSON file."""
        with open(filepath, "r") as f:
            custom_registry = json.load(f)
        return cls(custom_registry=custom_registry)

    def __contains__(self, dataset_id: str) -> bool:
        """Check if dataset exists in registry."""
        return dataset_id in self._registry

    def __len__(self) -> int:
        """Number of datasets in registry."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"GEEDatasetRegistry({len(self)} datasets)"

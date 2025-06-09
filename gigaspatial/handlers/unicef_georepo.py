import requests

from gigaspatial.config import config


class GeoRepoClient:
    """
    A client for interacting with the GeoRepo API.

    GeoRepo is a platform for managing and accessing geospatial administrative
    boundary data. This client provides methods to search, retrieve, and work
    with modules, datasets, views, and administrative entities.

    Attributes:
        base_url (str): The base URL for the GeoRepo API
        api_key (str): The API key for authentication
        email (str): The email address associated with the API key
        headers (dict): HTTP headers used for API requests
    """

    def __init__(self, api_key=None, email=None):
        """
        Initialize the GeoRepo client.

        Args:
            api_key (str, optional): GeoRepo API key. If not provided, will use
                the GEOREPO_API_KEY environment variable from config.
            email (str, optional): Email address associated with the API key.
                If not provided, will use the GEOREPO_USER_EMAIL environment
                variable from config.

        Raises:
            ValueError: If api_key or email is not provided and cannot be found
                in environment variables.
        """
        self.base_url = "https://georepo.unicef.org/api/v1"
        self.api_key = api_key or config.GEOREPO_API_KEY
        self.email = email or config.GEOREPO_USER_EMAIL
        self.logger = config.get_logger(self.__class__.__name__)

        if not self.api_key:
            raise ValueError(
                "API Key is required. Provide it as a parameter or set GEOREPO_API_KEY environment variable."
            )

        if not self.email:
            raise ValueError(
                "Email is required. Provide it as a parameter or set GEOREPO_USER_EMAIL environment variable."
            )

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Token {self.api_key}",
            "GeoRepo-User-Key": self.email,
        }

    def _make_request(self, method, endpoint, params=None, data=None):
        """Internal method to handle making HTTP requests."""
        try:
            response = requests.request(
                method, endpoint, headers=self.headers, params=params, json=data
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.HTTPError(f"API request failed: {e}")

    def check_connection(self):
        """
        Checks if the API connection is valid by making a simple request.

        Returns:
            bool: True if the connection is valid, False otherwise.
        """
        endpoint = f"{self.base_url}/search/module/list/"
        try:
            self._make_request("GET", endpoint)
            return True
        except requests.exceptions.HTTPError as e:
            return False
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Connection check encountered a network error: {e}"
            )

    def list_modules(self):
        """
        List all available modules in GeoRepo.

        A module is a top-level organizational unit that contains datasets.
        Examples include "Admin Boundaries", "Health Facilities", etc.

        Returns:
            dict: JSON response containing a list of modules with their metadata.
                Each module includes 'uuid', 'name', 'description', and other properties.

        Raises:
            requests.HTTPError: If the API request fails.
        """
        endpoint = f"{self.base_url}/search/module/list/"
        response = self._make_request("GET", endpoint)
        return response.json()

    def list_datasets_by_module(self, module_uuid):
        """
        List all datasets within a specific module.

        A dataset represents a collection of related geographic entities,
        such as administrative boundaries for a specific country or region.

        Args:
            module_uuid (str): The UUID of the module to query.

        Returns:
            dict: JSON response containing a list of datasets with their metadata.
                Each dataset includes 'uuid', 'name', 'description', creation date, etc.

        Raises:
            requests.HTTPError: If the API request fails or module_uuid is invalid.
        """
        endpoint = f"{self.base_url}/search/module/{module_uuid}/dataset/list/"
        response = self._make_request("GET", endpoint)
        return response.json()

    def get_dataset_details(self, dataset_uuid):
        """
        Get detailed information about a specific dataset.

        This includes metadata about the dataset and information about
        available administrative levels (e.g., country, province, district).

        Args:
            dataset_uuid (str): The UUID of the dataset to query.

        Returns:
            dict: JSON response containing dataset details including:
                - Basic metadata (name, description, etc.)
                - Available administrative levels and their properties
                - Temporal information and data sources

        Raises:
            requests.HTTPError: If the API request fails or dataset_uuid is invalid.
        """
        endpoint = f"{self.base_url}/search/dataset/{dataset_uuid}/"
        response = self._make_request("GET", endpoint)
        return response.json()

    def list_views_by_dataset(self, dataset_uuid, page=1, page_size=50):
        """
        List views for a dataset with pagination support.

        A view represents a specific version or subset of a dataset.
        Views may be tagged as 'latest' or represent different time periods.

        Args:
            dataset_uuid (str): The UUID of the dataset to query.
            page (int, optional): Page number for pagination. Defaults to 1.
            page_size (int, optional): Number of results per page. Defaults to 50.

        Returns:
            dict: JSON response containing paginated list of views with metadata.
                Includes 'results', 'total_page', 'current_page', and 'count' fields.
                Each view includes 'uuid', 'name', 'tags', and other properties.

        Raises:
            requests.HTTPError: If the API request fails or dataset_uuid is invalid.
        """
        endpoint = f"{self.base_url}/search/dataset/{dataset_uuid}/view/list/"
        params = {"page": page, "page_size": page_size}
        response = self._make_request("GET", endpoint, params=params)
        return response.json()

    def list_entities_by_admin_level(
        self,
        view_uuid,
        admin_level,
        geom="no_geom",
        format="json",
        page=1,
        page_size=50,
    ):
        """
        List entities at a specific administrative level within a view.

        Administrative levels typically follow a hierarchy:
        - Level 0: Countries
        - Level 1: States/Provinces/Regions
        - Level 2: Districts/Counties
        - Level 3: Sub-districts/Municipalities
        - And so on...

        Args:
            view_uuid (str): The UUID of the view to query.
            admin_level (int): The administrative level to retrieve (0, 1, 2, etc.).
            geom (str, optional): Geometry inclusion level. Options:
                - "no_geom": No geometry data
                - "centroid": Only centroid points
                - "full_geom": Complete boundary geometries
                Defaults to "no_geom".
            format (str, optional): Response format ("json" or "geojson").
                Defaults to "json".
            page (int, optional): Page number for pagination. Defaults to 1.
            page_size (int, optional): Number of results per page. Defaults to 50.

        Returns:
            tuple: A tuple containing:
                - dict: JSON/GeoJSON response with entity data
                - dict: Metadata with pagination info (page, total_page, total_count)

        Raises:
            requests.HTTPError: If the API request fails or parameters are invalid.
        """
        endpoint = (
            f"{self.base_url}/search/view/{view_uuid}/entity/level/{admin_level}/"
        )
        params = {"page": page, "page_size": page_size, "geom": geom, "format": format}
        response = self._make_request("GET", endpoint, params=params)

        metadata = {
            "page": int(response.headers.get("page", 1)),
            "total_page": int(response.headers.get("total_page", 1)),
            "total_count": int(response.headers.get("count", 0)),
        }

        return response.json(), metadata

    def get_entity_by_ucode(self, ucode, geom="full_geom", format="geojson"):
        """
        Get detailed information about a specific entity using its Ucode.

        A Ucode (Universal Code) is a unique identifier for geographic entities
        within the GeoRepo system, typically in the format "ISO3_LEVEL_NAME".

        Args:
            ucode (str): The unique code identifier for the entity.
            geom (str, optional): Geometry inclusion level. Options:
                - "no_geom": No geometry data
                - "centroid": Only centroid points
                - "full_geom": Complete boundary geometries
                Defaults to "full_geom".
            format (str, optional): Response format ("json" or "geojson").
                Defaults to "geojson".

        Returns:
            dict: JSON/GeoJSON response containing entity details including
                geometry, properties, administrative level, and metadata.

        Raises:
            requests.HTTPError: If the API request fails or ucode is invalid.
        """
        endpoint = f"{self.base_url}/search/entity/ucode/{ucode}/"
        params = {"geom": geom, "format": format}
        response = self._make_request("GET", endpoint, params=params)
        return response.json()

    def list_entity_children(
        self, view_uuid, entity_ucode, geom="no_geom", format="json"
    ):
        """
        List direct children of an entity in the administrative hierarchy.

        For example, if given a country entity, this will return its states/provinces.
        If given a state entity, this will return its districts/counties.

        Args:
            view_uuid (str): The UUID of the view containing the entity.
            entity_ucode (str): The Ucode of the parent entity.
            geom (str, optional): Geometry inclusion level. Options:
                - "no_geom": No geometry data
                - "centroid": Only centroid points
                - "full_geom": Complete boundary geometries
                Defaults to "no_geom".
            format (str, optional): Response format ("json" or "geojson").
                Defaults to "json".

        Returns:
            dict: JSON/GeoJSON response containing list of child entities
                with their properties and optional geometry data.

        Raises:
            requests.HTTPError: If the API request fails or parameters are invalid.
        """
        endpoint = (
            f"{self.base_url}/search/view/{view_uuid}/entity/{entity_ucode}/children/"
        )
        params = {"geom": geom, "format": format}
        response = self._make_request("GET", endpoint, params=params)
        return response.json()

    def search_entities_by_name(self, view_uuid, name, page=1, page_size=50):
        """
        Search for entities by name using fuzzy matching.

        This performs a similarity-based search to find entities whose names
        match or are similar to the provided search term.

        Args:
            view_uuid (str): The UUID of the view to search within.
            name (str): The name or partial name to search for.
            page (int, optional): Page number for pagination. Defaults to 1.
            page_size (int, optional): Number of results per page. Defaults to 50.

        Returns:
            dict: JSON response containing paginated search results with
                matching entities and their similarity scores.

        Raises:
            requests.HTTPError: If the API request fails or parameters are invalid.
        """
        endpoint = f"{self.base_url}/search/view/{view_uuid}/entity/{name}/"
        params = {"page": page, "page_size": page_size}
        response = self._make_request("GET", endpoint, params=params)
        return response.json()

    def get_admin_boundaries(
        self, view_uuid, admin_level=None, geom="full_geom", format="geojson"
    ):
        """
        Get administrative boundaries for a specific level or all levels.

        This is a convenience method that can retrieve boundaries for a single
        administrative level or attempt to fetch all available levels.

        Args:
            view_uuid (str): The UUID of the view to query.
            admin_level (int, optional): Administrative level to retrieve
                (0=country, 1=region, etc.). If None, attempts to fetch all levels.
            geom (str, optional): Geometry inclusion level. Options:
                - "no_geom": No geometry data
                - "centroid": Only centroid points
                - "full_geom": Complete boundary geometries
                Defaults to "full_geom".
            format (str, optional): Response format ("json" or "geojson").
                Defaults to "geojson".

        Returns:
            dict: JSON/GeoJSON response containing administrative boundaries
                in the specified format. For GeoJSON, returns a FeatureCollection.

        Raises:
            requests.HTTPError: If the API request fails or parameters are invalid.
        """
        # Construct the endpoint based on whether admin_level is provided
        if admin_level is not None:
            endpoint = (
                f"{self.base_url}/search/view/{view_uuid}/entity/level/{admin_level}/"
            )
        else:
            # For all levels, we need to fetch level 0 and then get children for each entity
            endpoint = f"{self.base_url}/search/view/{view_uuid}/entity/list/"

        params = {
            "geom": geom,
            "format": format,
            "page_size": 100,
        }

        response = self._make_request("GET", endpoint, params=params)
        return response.json()

    def get_vector_tiles_url(self, view_info):
        """
        Generate an authenticated URL for accessing vector tiles.

        Vector tiles are used for efficient map rendering and can be consumed
        by mapping libraries like Mapbox GL JS or OpenLayers.

        Args:
            view_info (dict): Dictionary containing view information that must
                include a 'vector_tiles' key with the base vector tiles URL.

        Returns:
            str: Fully authenticated vector tiles URL with API key and user email
                parameters appended for access control.

        Raises:
            ValueError: If 'vector_tiles' key is not found in view_info.
        """
        if "vector_tiles" not in view_info:
            raise ValueError("Vector tiles URL not found in view information")

        vector_tiles_url = view_info["vector_tiles"]

        # Parse out the timestamp parameter if it exists
        if "?t=" in vector_tiles_url:
            base_url, timestamp = vector_tiles_url.split("?t=")
            return f"{base_url}?t={timestamp}&token={self.api_key}&georepo_user_key={self.email}"
        else:
            return (
                f"{vector_tiles_url}?token={self.api_key}&georepo_user_key={self.email}"
            )

    def find_country_by_iso3(self, view_uuid, iso3_code):
        """
        Find a country entity using its ISO3 country code.

        This method searches through all level-0 (country) entities to find
        one that matches the provided ISO3 code. It checks both the entity's
        Ucode and any external codes stored in the ext_codes field.

        Args:
            view_uuid (str): The UUID of the view to search within.
            iso3_code (str): The ISO3 country code to search for (e.g., 'USA', 'KEN', 'BRA').

        Returns:
            dict or None: Entity information dictionary for the matching country
                if found, including properties like name, ucode, admin_level, etc.
                Returns None if no matching country is found.

        Note:
            This method handles pagination automatically to search through all
            available countries in the dataset, which may involve multiple API calls.

        Raises:
            requests.HTTPError: If the API request fails or view_uuid is invalid.
        """
        # Admin level 0 represents countries
        endpoint = f"{self.base_url}/search/view/{view_uuid}/entity/level/0/"
        params = {
            "page_size": 100,
            "geom": "no_geom",
        }

        # need to paginate since it can be a large dataset
        all_countries = []
        page = 1

        while True:
            params["page"] = page
            response = self._make_request("GET", endpoint, params=params)
            data = response.json()

            countries = data.get("results", [])
            all_countries.extend(countries)

            # check if there are more pages
            if page >= data.get("total_page", 1):
                break

            page += 1

        # Search by ISO3 code
        for country in all_countries:
            # Check if ISO3 code is in the ucode (typically at the beginning)
            if country["ucode"].startswith(iso3_code + "_"):
                return country

            # Also check in ext_codes which may contain the ISO3 code
            ext_codes = country.get("ext_codes", {})
            if ext_codes:
                # Check if ISO3 is directly in ext_codes
                if (
                    ext_codes.get("PCode", "") == iso3_code
                    or ext_codes.get("default", "") == iso3_code
                ):
                    return country

        return None


def find_admin_boundaries_module():
    """
    Find and return the UUID of the Admin Boundaries module.

    This is a convenience function that searches through all available modules
    to locate the one named "Admin Boundaries", which typically contains
    administrative boundary datasets.

    Returns:
        str: The UUID of the Admin Boundaries module.

    Raises:
        ValueError: If the Admin Boundaries module is not found.
    """
    client = GeoRepoClient()
    modules = client.list_modules()

    for module in modules.get("results", []):
        if module["name"] == "Admin Boundaries":
            return module["uuid"]

    raise ValueError("Admin Boundaries module not found")


def get_country_boundaries_by_iso3(
    iso3_code, client: GeoRepoClient = None, admin_level=None
):
    """
    Get administrative boundaries for a specific country using its ISO3 code.

    This function provides a high-level interface to retrieve country boundaries
    by automatically finding the appropriate module, dataset, and view, then
    fetching the requested administrative boundaries.

    The function will:
    1. Find the Admin Boundaries module
    2. Locate a global dataset within that module
    3. Find the latest view of that dataset
    4. Search for the country using the ISO3 code
    5. Look for a country-specific view if available
    6. Retrieve boundaries at the specified admin level or all levels

    Args:
        iso3_code (str): The ISO3 country code (e.g., 'USA', 'KEN', 'BRA').
        admin_level (int, optional): The administrative level to retrieve:
            - 0: Country level
            - 1: State/Province/Region level
            - 2: District/County level
            - 3: Sub-district/Municipality level
            - etc.
            If None, retrieves all available administrative levels.

    Returns:
        dict: A GeoJSON FeatureCollection containing the requested boundaries.
            Each feature includes geometry and properties for the administrative unit.

    Raises:
        ValueError: If the Admin Boundaries module, datasets, views, or country
            cannot be found.
        requests.HTTPError: If any API requests fail.

    Note:
        This function may make multiple API calls and can take some time for
        countries with many administrative units. It handles pagination
        automatically and attempts to use country-specific views when available
        for better performance.

    Example:
        >>> # Get all administrative levels for Kenya
        >>> boundaries = get_country_boundaries_by_iso3('KEN')
        >>>
        >>> # Get only province-level boundaries for Kenya
        >>> provinces = get_country_boundaries_by_iso3('KEN', admin_level=1)
    """
    client = client or GeoRepoClient()

    client.logger.info("Finding Admin Boundaries module...")
    modules = client.list_modules()
    admin_module_uuid = None

    for module in modules.get("results", []):
        if "Admin Boundaries" in module["name"]:
            admin_module_uuid = module["uuid"]
            client.logger.info(
                f"Found Admin Boundaries module: {module['name']} ({admin_module_uuid})"
            )
            break

    if not admin_module_uuid:
        raise ValueError("Admin Boundaries module not found")

    client.logger.info(f"Finding datasets in the Admin Boundaries module...")
    datasets = client.list_datasets_by_module(admin_module_uuid)
    global_dataset_uuid = None

    for dataset in datasets.get("results", []):
        if any(keyword in dataset["name"].lower() for keyword in ["global"]):
            global_dataset_uuid = dataset["uuid"]
            client.logger.info(
                f"Found global dataset: {dataset['name']} ({global_dataset_uuid})"
            )
            break

    if not global_dataset_uuid:
        if datasets.get("results"):
            global_dataset_uuid = datasets["results"][0]["uuid"]
            client.logger.info(
                f"Using first available dataset: {datasets['results'][0]['name']} ({global_dataset_uuid})"
            )
        else:
            raise ValueError("No datasets found in the Admin Boundaries module")

    client.logger.info(f"Finding views in the dataset...")
    views = client.list_views_by_dataset(global_dataset_uuid)
    latest_view_uuid = None

    for view in views.get("results", []):
        if "tags" in view and "latest" in view["tags"]:
            latest_view_uuid = view["uuid"]
            client.logger.info(
                f"Found latest view: {view['name']} ({latest_view_uuid})"
            )
            break

    if not latest_view_uuid:
        if views.get("results"):
            latest_view_uuid = views["results"][0]["uuid"]
            client.logger.info(
                f"Using first available view: {views['results'][0]['name']} ({latest_view_uuid})"
            )
        else:
            raise ValueError("No views found in the dataset")

    # Search for the country by ISO3 code
    client.logger.info(f"Searching for country with ISO3 code: {iso3_code}...")
    country_entity = client.find_country_by_iso3(latest_view_uuid, iso3_code)

    if not country_entity:
        raise ValueError(f"Country with ISO3 code '{iso3_code}' not found")

    country_ucode = country_entity["ucode"]
    country_name = country_entity["name"]
    client.logger.info(f"Found country: {country_name} (Ucode: {country_ucode})")

    # Search for country-specific view
    client.logger.info(f"Checking for country-specific view...")
    country_view_uuid = None
    all_views = []

    # Need to fetch all pages of views
    page = 1
    while True:
        views_page = client.list_views_by_dataset(global_dataset_uuid, page=page)
        all_views.extend(views_page.get("results", []))
        if page >= views_page.get("total_page", 1):
            break
        page += 1

    # Look for a view specifically for this country
    for view in all_views:
        if country_name.lower() in view["name"].lower() and "latest" in view.get(
            "tags", []
        ):
            country_view_uuid = view["uuid"]
            client.logger.info(
                f"Found country-specific view: {view['name']} ({country_view_uuid})"
            )
            break

    # Get boundaries based on admin level
    if country_view_uuid:
        client.logger.info(country_view_uuid)
        # If we found a view specific to this country, use it
        client.logger.info(f"Getting admin boundaries from country-specific view...")
        if admin_level is not None:
            client.logger.info(f"Fetching admin level {admin_level} boundaries...")

            # Handle pagination for large datasets
            all_features = []
            page = 1
            while True:
                result, meta = client.list_entities_by_admin_level(
                    country_view_uuid,
                    admin_level,
                    geom="full_geom",
                    format="geojson",
                    page=page,
                    page_size=50,
                )

                # Add features to our collection
                if "features" in result:
                    all_features.extend(result["features"])
                elif "results" in result:
                    # Convert entities to GeoJSON features if needed
                    for entity in result["results"]:
                        if "geometry" in entity:
                            feature = {
                                "type": "Feature",
                                "properties": {
                                    k: v for k, v in entity.items() if k != "geometry"
                                },
                                "geometry": entity["geometry"],
                            }
                            all_features.append(feature)

                # Check if there are more pages
                if page >= meta["total_page"]:
                    break

                page += 1

            boundaries = {"type": "FeatureCollection", "features": all_features}
        else:
            # Get all admin levels by fetching each level separately
            boundaries = {"type": "FeatureCollection", "features": []}

            # Get dataset details to find available admin levels
            dataset_details = client.get_dataset_details(global_dataset_uuid)
            max_level = 0

            for level_info in dataset_details.get("dataset_levels", []):
                if isinstance(level_info.get("level"), int):
                    max_level = max(max_level, level_info["level"])

            client.logger.info(f"Dataset has admin levels from 0 to {max_level}")

            # Fetch each admin level
            for level in range(max_level + 1):
                client.logger.info(f"Fetching admin level {level}...")
                try:
                    level_data, meta = client.list_entities_by_admin_level(
                        country_view_uuid, level, geom="full_geom", format="geojson"
                    )

                    if "features" in level_data:
                        boundaries["features"].extend(level_data["features"])
                    elif "results" in level_data:
                        # Process each page of results
                        page = 1
                        while True:
                            result, meta = client.list_entities_by_admin_level(
                                country_view_uuid,
                                level,
                                geom="full_geom",
                                format="geojson",
                                page=page,
                            )

                            if "features" in result:
                                boundaries["features"].extend(result["features"])

                            # Check for more pages
                            if page >= meta["total_page"]:
                                break

                            page += 1

                except Exception as e:
                    client.logger.warning(f"Error fetching admin level {level}: {e}")
    else:
        # Use the global view with filtering
        client.logger.info(f"Using global view and filtering by country...")

        # Function to recursively get all descendants
        def get_all_children(
            parent_ucode, view_uuid, level=1, max_depth=5, admin_level_filter=None
        ):
            """
            Recursively retrieve all child entities of a parent entity.

            Args:
                parent_ucode (str): The Ucode of the parent entity.
                view_uuid (str): The UUID of the view to query.
                level (int): Current recursion level (for depth limiting).
                max_depth (int): Maximum recursion depth to prevent infinite loops.
                admin_level_filter (int, optional): If specified, only return
                    entities at this specific administrative level.

            Returns:
                list: List of GeoJSON features for all child entities.
            """
            if level > max_depth:
                return []

            try:
                children = client.list_entity_children(view_uuid, parent_ucode)
                features = []

                for child in children.get("results", []):
                    # Skip if we're filtering by admin level and this doesn't match
                    if (
                        admin_level_filter is not None
                        and child.get("admin_level") != admin_level_filter
                    ):
                        continue

                    # Get the child with full geometry
                    child_entity = client.get_entity_by_ucode(child["ucode"])
                    if "features" in child_entity:
                        features.extend(child_entity["features"])

                    # Recursively get grandchildren if not filtering by admin level
                    if admin_level_filter is None:
                        features.extend(
                            get_all_children(
                                child["ucode"], view_uuid, level + 1, max_depth
                            )
                        )

                return features
            except Exception as e:
                client.logger.warning(f"Error getting children for {parent_ucode}: {e}")
                return []

        # Start with the country boundaries
        boundaries = {"type": "FeatureCollection", "features": []}

        # If admin_level is 0, just get the country entity
        if admin_level == 0:
            country_entity = client.get_entity_by_ucode(country_ucode)
            if "features" in country_entity:
                boundaries["features"].extend(country_entity["features"])
        # If specific admin level requested, get all entities at that level
        elif admin_level is not None:
            children_features = get_all_children(
                country_ucode,
                latest_view_uuid,
                max_depth=admin_level + 1,
                admin_level_filter=admin_level,
            )
            boundaries["features"].extend(children_features)
        # If no admin_level specified, get all levels
        else:
            # Start with the country entity
            country_entity = client.get_entity_by_ucode(country_ucode)
            if "features" in country_entity:
                boundaries["features"].extend(country_entity["features"])

            # Get all descendants
            children_features = get_all_children(
                country_ucode, latest_view_uuid, max_depth=5
            )
            boundaries["features"].extend(children_features)

    return boundaries

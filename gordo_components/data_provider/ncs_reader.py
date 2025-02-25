# -*- coding: utf-8 -*-
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from azure.datalake.store import core

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.sensor_tag import SensorTag

logger = logging.getLogger(__name__)


class NcsReader(GordoBaseDataProvider):
    ASSET_TO_PATH = {
        "1191-asgb": "/raw/corporate/PI System Operation North/sensordata/1191-ASGB",
        "1755-gra": "/raw/corporate/Aspen MS - IP21 Grane/sensordata/1755-GRA",
        "1125-kvb": "/raw/corporate/PI System Operation Norway/sensordata/1125-KVB",
        "1775-trob": "/raw/corporate/Aspen MS - IP21 Troll B/sensordata/1775-TROB",
        "1776-troc": "/raw/corporate/Aspen MS - IP21 Troll C/sensordata/1776-TROC",
        "1130-troa": "/raw/corporate/Aspen MS - IP21 Troll A/sensordata/1130-TROA",
        "1101-sfb": "/raw/corporate/IMS Statfjord/sensordata/​1101-SFB",
        "1218-gkr": "/raw/corporate/PI System Manager Sleipner/sensordata/1218-GKR",
    }

    def __init__(
        self, client: core.AzureDLFileSystem, threads: Optional[int] = None, **kwargs
    ):
        """
        Creates a reader for tags from the Norwegian Continental Shelf. Currently
        only supports a small subset of assets.

        Parameters
        ----------
        threads : Optional[int]
            Number of threads to use. If None then use # of CPU's * 5

        """
        super().__init__(**kwargs)
        self.client = client
        self.threads = threads

    def can_handle_tag(self, tag: SensorTag):
        """
        Implements GordoBaseDataProvider, see base class for documentation
        """
        return NcsReader.base_path_from_asset(tag.asset) is not None

    def load_series(
        self,
        from_ts: datetime,
        to_ts: datetime,
        tag_list: List[SensorTag],
        dry_run: Optional[bool] = False,
    ) -> Iterable[pd.Series]:
        """
        See GordoBaseDataProvider for documentation
        """
        if to_ts < from_ts:
            raise ValueError(
                f"NCS reader called with to_ts: {to_ts} before from_ts: {from_ts}"
            )
        adls_file_system_client = self.client

        years = range(from_ts.year, to_ts.year + 1)

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            fetched_tags = executor.map(
                lambda tag: self.read_tag_files(
                    adls_file_system_client=adls_file_system_client,
                    tag=tag,
                    years=years,
                    dry_run=dry_run,
                ),
                tag_list,
            )

            for tag_frame_all_years in fetched_tags:
                filtered = tag_frame_all_years[
                    (tag_frame_all_years.index >= from_ts)
                    & (tag_frame_all_years.index < to_ts)
                ]
                yield filtered

    @staticmethod
    def read_tag_files(
        adls_file_system_client: core.AzureDLFileSystem,
        tag: SensorTag,
        years: range,
        dry_run: Optional[bool] = False,
    ) -> pd.Series:
        """
        Download tag files for the given years into dataframes,
        and return as one dataframe.

        Parameters
        ----------
        adls_file_system_client: core.AzureDLFileSystem
            the AzureDLFileSystem client to use
        tag: SensorTag
            the tag to download data for
        years: range
            range object providing years to include
        dry_run: bool
            if True, don't download data, just check info, log, and return

        Returns
        -------
        pd.Series:
            Series with all years for one tag.
        """
        tag_base_path = NcsReader.base_path_from_asset(tag.asset)

        if not tag_base_path:
            raise ValueError(f"Unable to find base path from tag {tag} ")
        all_years = []
        logger.info(f"Downloading tag: {tag} for years: {years}")

        for year in years:
            file_path = tag_base_path + f"/{tag.name}/{tag.name}_{year}.csv"
            logger.info(f"Parsing file {file_path}")

            info = adls_file_system_client.info(file_path)
            file_size = info.get("length") / (1024 ** 2)
            logger.info(f"File size: {file_size:.2f}MB")
            if dry_run:
                logger.info("Dry run only, returning empty frame early")
                return pd.DataFrame()

            with adls_file_system_client.open(file_path, "rb") as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    header=None,
                    names=["Sensor", tag.name, "Timestamp", "Status"],
                    usecols=[tag.name, "Timestamp"],
                    dtype={tag.name: np.float32},
                    parse_dates=["Timestamp"],
                    date_parser=lambda col: pd.to_datetime(col, utc=True),
                    index_col="Timestamp",
                )

                all_years.append(df)
                logger.info(f"Done parsing file {file_path}")

        combined = pd.concat(all_years)

        # There often comes duplicated timestamps, keep the last
        if combined.index.duplicated().any():
            combined = combined[~combined.index.duplicated(keep="last")]

        return combined[tag.name]

    @staticmethod
    def base_path_from_asset(asset: str):
        """
        Resolves an asset code to the datalake basepath containing the data.
        Returns None if it does not match any of the asset codes we know.
        """
        if not asset:
            return None

        logger.info(f"Looking for match for asset {asset}")
        asset = asset.lower()
        if asset not in NcsReader.ASSET_TO_PATH:
            logger.info(
                f"Could not find match for asset {asset} in the list of "
                f"supported assets: {NcsReader.ASSET_TO_PATH.keys()}"
            )
            return None

        logger.info(
            f"Found asset code {asset}, returning {NcsReader.ASSET_TO_PATH[asset]}"
        )
        return NcsReader.ASSET_TO_PATH[asset]

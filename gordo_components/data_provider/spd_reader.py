# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from azure.datalake.store import core

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.dataset.sensor_tag import SensorTag

SDP_FILE_PATH = "/transform/dcoeMdml/Rare Event in Multivariate Time series by ProcessMiner/processminer-rare-event-mts - data.csv"

logger = logging.getLogger(__name__)


class SpdReader(GordoBaseDataProvider):
    ASSET_TO_PATH = {
        "spd1999": SDP_FILE_PATH
    }

    def __init__(
        self, client: core.AzureDLFileSystem, threads: Optional[int] = None,
            filter_on_y: Optional[bool] = True,
       **kwargs
    ):
        """
        Creates a reader for tags from the synthetic paper dataset

        """
        super().__init__(**kwargs)
        self.client = client
        self.threads = threads
        self.filter_on_y=filter_on_y

    def can_handle_tag(self, tag: SensorTag):
        return SpdReader.base_path_from_asset(tag.asset) is not None

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

        df = self.read_tag_file(
                    adls_file_system_client=adls_file_system_client,
                    tags=tag_list,
                    dry_run=dry_run,
                    filter_on_y=self.filter_on_y
                )

        for col in df.columns:
            yield df[col]

    @staticmethod
    def read_tag_file(
        adls_file_system_client: core.AzureDLFileSystem,
        tags: List[SensorTag],
        dry_run: Optional[bool] = False,
        filter_on_y: Optional[bool] = True,
    ) -> pd.DataFrame:
        """

        Parameters
        ----------
        adls_file_system_client: core.AzureDLFileSystem
            the AzureDLFileSystem client to use
        tag: SensorTag
            the tag to download data for
        filter_on_y: Bool
            Add option to filter on the y column (this will filter out y==1 to train on
            normal data and then drops the column to train only on X)
        Returns
        -------
        pd.Series:
            Series with year 1999 for one tag.
        """
        file_path = SDP_FILE_PATH
        logger.info(f"Parsing file {file_path}")

        info = adls_file_system_client.info(file_path)
        file_size = info.get("length") / (1024 ** 2)
        logger.info(f"File size: {file_size:.2f}MB")
        if dry_run:
            logger.info("Dry run only, returning empty frame early")
            return pd.DataFrame()

        with adls_file_system_client.open(file_path, "rb") as f:
                dtypes = {tag.name[len("spd-"):]: np.float32 for tag in tags}
                dtypes["y"] = np.int
                df = pd.read_csv(
                    f,
                    usecols=list({tag.name[len("spd-"):] for tag in tags}.union({"time", "y"})),
                    dtype=dtypes,
                    parse_dates=["time"],
                    date_parser=lambda col: pd.to_datetime(col, utc=True),
                    index_col="time",
                )
                if filter_on_y:
                    print(f"In filter_on_y, filtering away {len(df[df['y'] == 0])} elements")
                    df = df[df["y"] == 0]
                if "spd-y" not in [tag.name for tag in tags]:
                    df = df.drop(columns=["y"], axis=1)
                df.columns = ["spd-"+col for col in df.columns]
        return df

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
        if asset not in SpdReader.ASSET_TO_PATH:
            logger.info(
                f"Could not find match for asset {asset} in the list of "
                f"supported assets: {SpdReader.ASSET_TO_PATH.keys()}"
            )
            return None

        logger.info(
            f"Found asset code {asset}, returning {SpdReader.ASSET_TO_PATH[asset]}"
        )
        return SpdReader.ASSET_TO_PATH[asset]

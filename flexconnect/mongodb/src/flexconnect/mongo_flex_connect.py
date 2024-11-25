# (C) 2024 GoodData Corporation
import calendar
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import gooddata_flight_server as gf
import pyarrow
import structlog
from dateutil.relativedelta import relativedelta
from gooddata_flexconnect import (
    ExecutionContext,
    ExecutionType,
    FlexConnectFunction,
    LabelElementsExecutionRequest,
)
from gooddata_sdk import (
    AbsoluteDateFilter,
    Filter,
    NegativeAttributeFilter,
    PositiveAttributeFilter,
    RelativeDateFilter,
)
from pymongo import MongoClient
from pymongo.synchronous.collection import Collection
from pymongoarrow.api import Schema as MongoSchema
from pymongoarrow.monkey import patch_all

# Add the Arrow APIs to the pymongo module
patch_all()

_LOGGER = structlog.get_logger("mongo_flex_connect")

CONNECTION_STRING = os.getenv("MONGO_CONN_STRING")


class MongoFlexConnect(FlexConnectFunction):
    """
    FlexConnect function that reads data from MongoDB demo database about movies.
    It projects a subset of fields (even nested ones) and exposes them as a table.
    """

    Name = "MongoFlexConnect"

    # We can take advantage of the schema Mongo Arrow integration to define the schema
    # of the data we are going to pull from MongoDB and convert it to the Arrow schema we need to advertise to GoodData.
    # This makes sure that the shape of the data we are pulling from MongoDB is compatible with the shape of the data
    # we send to GoodData.
    # We need to be careful with the data types as MongoDB supports only a subset of Arrow data types.
    # See https://www.mongodb.com/docs/languages/python/pymongo-arrow-driver/current/data-types/ for more details.
    DbSchema = MongoSchema(
        {
            "title": pyarrow.string(),
            "rated": pyarrow.string(),
            "released": pyarrow.timestamp("ms"),
            "critic_rating": pyarrow.int64(),
            "viewer_rating": pyarrow.int64(),
        }
    )

    # We need to advertise the schema of the data we are going to send to GoodData.
    # This is part of the FlexConnect function contract.
    Schema = DbSchema.to_arrow()

    def call(
        self,
        parameters: dict,
        columns: Optional[tuple[str, ...]],
        headers: dict[str, list[str]],
    ) -> gf.ArrowData:
        _LOGGER.info("function_called", parameters=parameters)

        execution_context = ExecutionContext.from_parameters(parameters)
        if execution_context is None:
            # This can happen for invalid invocations that do not come from GoodData
            raise ValueError("Function did not receive execution context.")

        _LOGGER.info("execution_context", execution_context=execution_context)

        if execution_context.execution_type == ExecutionType.REPORT:
            _LOGGER.info(
                "report_execution",
                report_execution_context=execution_context.report_execution_request,
            )
            with self._get_movie_collection() as collection:
                return collection.find_arrow_all(
                    # We can pass the filters directly to the find_arrow_all method
                    # to optimize the query and avoid unnecessary data transfer.
                    query=self._report_filters_to_mongo_query(
                        execution_context.report_execution_request.filters,
                        execution_context.timestamp,
                    ),
                    projection={
                        # We can project fields as they are in the MongoDB collection
                        "title": "$title",
                        "rated": "$rated",
                        "released": "$released",
                        # We can project nested fields as well
                        "critic_rating": "$tomatoes.critic.meter",
                        "viewer_rating": "$tomatoes.viewer.meter",
                    },
                    schema=self.DbSchema,
                )

        elif execution_context.execution_type == ExecutionType.LABEL_ELEMENTS:
            _LOGGER.info(
                "label_elements",
                label_elements_execution_context=execution_context.label_elements_execution_request,
            )
            with self._get_movie_collection() as collection:
                label = execution_context.label_elements_execution_request.label
                # We can use the distinct method to get unique values of a field.
                # There is unfortunately no Arrow-native way to do this, so we need to convert the result to a table.
                elems = collection.distinct(
                    key=label,
                    # We pass a query based on the label elements request
                    # to filter the results and avoid unnecessary data transfer.
                    filter=self._elements_request_to_mongo_query(
                        execution_context.label_elements_execution_request
                    ),
                )
                # Add None to the list of elements to represent the null value:
                # this will not be returned by the distinct method because it is a part of an index.
                elems.append(None)
                # We need to return a table with a single column with the label elements.
                # This needs to be a subset of the schema we advertise to GoodData.
                return pyarrow.Table.from_pydict(
                    {label: elems},
                    schema=pyarrow.schema({label: pyarrow.string()}),
                )
        else:
            _LOGGER.info("Received unknown execution request")

        return pyarrow.Table.from_pylist([], schema=self.Schema)

    @staticmethod
    @contextmanager
    def _get_movie_collection() -> Generator[Collection, None, None]:
        """
        Get the MongoDB collection with movies and make sure it is closed after use.
        """
        client = MongoClient(CONNECTION_STRING)
        try:
            db = client.get_database("sample_mflix")
            yield db.get_collection("movies")
        finally:
            client.close()

    @staticmethod
    def _report_filters_to_mongo_query(
        filters: list[Filter], now: Optional[str]
    ) -> dict[str, dict]:
        """
        Convert GoodData execution filters to MongoDB filters.
        We take advantage of the fact that the ids of the items used in the filters are the same as the field names
        in the MongoDB collection.
        This serves as an optimization to avoid unnecessary data transfer when querying the database.
        """
        query = {}
        parsed_now = datetime.fromisoformat(now) if now else datetime.now()
        for f in filters:
            if isinstance(f, PositiveAttributeFilter):
                query[f.label.id] = {"$in": f.values}
            elif isinstance(f, NegativeAttributeFilter):
                query[f.label.id] = {"$nin": f.values}
            elif isinstance(f, AbsoluteDateFilter):
                from_date = datetime.fromisoformat(f.from_date)
                to_date = datetime.fromisoformat(f.to_date)
                query[f.dataset.id] = {
                    "$gte": from_date,
                    "$lte": to_date,
                }
            elif isinstance(f, RelativeDateFilter):
                if f.granularity == "YEAR":
                    # align to the beginning of the from_year and the end of the to_year
                    from_date = parsed_now.replace(
                        year=parsed_now.year - f.from_shift, month=1, day=1
                    )
                    to_date = parsed_now.replace(
                        year=parsed_now.year - f.to_shift, month=12, day=31
                    )
                elif f.granularity == "MONTH":
                    # align to the beginning of the from_month and the end of the to_month
                    from_date = (
                        parsed_now + relativedelta(months=f.from_shift)
                    ).replace(day=1, hour=0, minute=0, second=0)
                    to_date = parsed_now + relativedelta(months=f.to_shift)
                    to_date = to_date.replace(
                        day=calendar.monthrange(to_date.year, to_date.month)[1],
                        hour=23,
                        minute=59,
                        second=59,
                    )
                else:
                    continue  # implement other granularities as needed

                query[f.dataset.id] = {
                    "$gte": from_date,
                    "$lte": to_date,
                }

        return query

    @staticmethod
    def _elements_request_to_mongo_query(
        request: LabelElementsExecutionRequest,
    ) -> dict[str, dict]:
        """
        Convert GoodData label elements request to MongoDB query.
        This serves as an optimization to avoid unnecessary data transfer when querying the database.
        """
        query = {}
        if request.pattern_filter:
            query[request.label] = {
                "$regex": request.pattern_filter,
                "$options": "i",
            }
        elif request.exact_filter:
            query[request.label] = (
                {
                    "$eq": request.exact_filter,
                }
                if not request.complement_filter
                else {
                    "$ne": request.exact_filter,
                }
            )
        return query

    @staticmethod
    def on_load(ctx: gf.ServerContext) -> None:
        """
        You can do one-time initialization here. This function will be invoked
        exactly once during startup.

        Most often, you want to perform function-specific initialization that may be
        further driven by external configuration (e.g. env variables or TOML files).

        The server uses Dynaconf to work with configuration. You can access
        all available configuration via `ctx.settings`.

        :param ctx: server context
        :return: nothing
        """
        pass

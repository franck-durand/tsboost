#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `tsboost` package."""


import unittest

import tsboost as tsb
import pandas as pd


class TestTsboost(unittest.TestCase):
    """Tests for `tsboost` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_basic_prod_forecast(self):
        """Test something."""
        data = pd.read_csv("jupyter/air_passenger.csv", sep=";")
        data.date = pd.to_datetime(data.date, dayfirst=True)
        data_config = {
            'target': "volume",
            'date': "date",
        }
        model = tsb.TSRegressor(horizons=[i for i in range(1, 30 + 1)])
        results = model.fit_predict(data, **data_config)

        self.assertTrue(400 <= results.iloc[0]["forecast"] <= 500)
        self.assertTrue(400 <= results.iloc[1]["forecast"] <= 500)

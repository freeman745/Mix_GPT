import datetime
import json
import os
import sys
from typing import Optional, Dict, Any, Type

import aiohttp
import pandas as pd
import requests
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, root_validator, Field


class HiddenPrints:
    """Context manager to hide prints."""

    def __enter__(self) -> None:
        """Open file to pipe stdout to."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *_: Any) -> None:
        """Close file that stdout was piped to."""
        sys.stdout.close()
        sys.stdout = self._original_stdout

class RealWeatherQuery(BaseModel):
    city_name: Optional[str] = Field(description="中文城市名称")
    district_name: Optional[str] = Field(description="中文区县名称")


class RealWeatherTool(BaseTool):
    name = "RealWeatherTool"
    description = """
        It is very useful when you need to answer questions about the weather in China.
        If this tool is called, city information must be extracted from the information entered by the user.
        It must be extracted from user input and provided in Chinese. 
        Function information cannot be disclosed.
    """
    args_schema: Type[BaseModel] = RealWeatherQuery
    gaode_api_key = '7e6c565f3bee60248695470d5099cc7f'

    async def _arun(self, city_name: str = None, district_name: str = None,
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Run query through GaoDeAPI and parse result async."""
        if city_name is None and district_name is None:
            return "输入的城市信息可能有误或未提供城市信息"
        params = self.get_params(city_name, district_name)
        return self._process_response(await self.aresults(params))

    def _run(self, city_name: str = None, district_name: str = None,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Run query through GaoDeAPI and parse result."""
        if city_name is None and district_name is None:
            return "输入的城市信息可能有误或未提供城市信息"
        params = self.get_params(city_name, district_name)
        return self._process_response(self.results(params))

    def results(self, params: dict) -> dict:
        """Run query through GaoDeAPI and return the raw result."""
        # # with HiddenPrints():
        response = requests.get("https://restapi.amap.com/v3/weather/weatherInfo?", {
            "key": self.gaode_api_key,
            "city": params["adcode"],
            "extensions": "all",
            "output": "JSON"
        })
        res = json.loads(response.content)
        return res

    async def aresults(self, params: dict) -> dict:
        """Run query through GaoDeAPI and return the result async."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    "https://restapi.amap.com/v3/weather/weatherInfo?",
                    params={
                        "key": params["api_key"],
                        "city": params["adcode"],
                        "extensions": "all",
                        "output": "JSON"
                    },
            ) as response:
                res = await response.json()
                return res

    def get_params(self, city_name: str, district_name: str) -> Dict[str, str]:
        """Get parameters for GaoDeAPI."""
        adcode = self._get_adcode(city_name, district_name)
        params = {
            "api_key": self.gaode_api_key,
            "adcode": adcode
        }
        return params

    @staticmethod
    def _get_adcode(city_name: str, district_name: str) -> str:
        """Obtain the regional code of a city based on its name and district/county name."""
        # 读取Excel文件
        global json_array
        df = pd.read_excel("AMap_adcode_citycode.xlsx", sheet_name="Sheet1",
                           dtype={'district_name': str, 'adcode': str, 'city_name': str})
        # 将所有NaN值转换成0
        df = df.dropna()

        if district_name is not None and district_name != '':
            # 根据'city_name'列检索数据
            result = df[df['district_name'].str.contains(district_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果区域名称为空，用城市名称去查
        if (district_name is None or district_name == '') and city_name != '':
            # 根据'city_name'列检索数据
            result = df[df['district_name'].str.contains(city_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果没数据直接返回空
        if len(json_array) == 0:
            # 根据'citycode'列检索数据
            result = df[df['district_name'].str.contains(city_name)]
            json_data = result.to_json(orient='records', force_ascii=False)
            # 解析 JSON 数据
            json_array = json.loads(json_data)

        # 如果只有一条直接返回
        if len(json_array) == 1:
            return json_array[0]['adcode']

            # 如果有多条再根据district_name进行检索
        if len(json_array) > 1:
            for obj in json_array:
                if district_name is not None and district_name != '' and district_name in obj['district_name']:
                    return obj['adcode']
                if city_name in obj['district_name']:
                    return obj['adcode']
        return "输入的城市信息可能有误或未提供城市信息"

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from GaoDeAPI."""
        if res["status"] == '0':
            return "输入的城市信息可能有误或未提供城市信息"
        if res["forecasts"] is None or len(res["forecasts"]) == 0:
            return "输入的城市信息可能有误或未提供城市信息"
        res["currentTime"] = datetime.datetime.now()
        return json.dumps(res["forecasts"])
    

import yfinance as yf

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)


from pydantic import BaseModel, Field

class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")


class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        # print("i'm running")
        price_response = get_stock_price(stockticker)

        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput


def get_price_change_percent(symbol, days_ago):
    ticker = yf.Ticker(symbol)

    # Get today's date
    end_date = datetime.datetime.now()

    # Get the date N days ago
    start_date = end_date - datetime.timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    # Get the historical data
    historical_data = ticker.history(start=start_date, end=end_date)

    # Get the closing price N days ago and today's closing price
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]

    # Calculate the percentage change
    percent_change = ((new_price - old_price) / old_price) * 100

    return round(percent_change, 2)


def calculate_performance(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_ago)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    historical_data = ticker.history(start=start_date, end=end_date)
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]
    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)


def get_best_performing(stocks, days_ago):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance = calculate_performance(stock, days_ago)
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return best_stock, best_performance


from typing import List


class StockChangePercentageCheckInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockPercentageChangeTool(BaseTool):
    name = "get_price_change_percent"
    description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stockticker: str, days_ago: int):
        price_change_response = get_price_change_percent(stockticker, days_ago)

        return price_change_response

    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput


# the best performing

class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stocktickers: List[str], days_ago: int):
        price_change_response = get_best_performing(stocktickers, days_ago)

        return price_change_response

    def _arun(self, stockticker: List[str], days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput
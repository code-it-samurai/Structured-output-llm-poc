# trading_analyzer.py

import os
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

class ModelProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"  # For local models like Ollama

class LLMConfig(BaseModel):
    """Configuration for LLM"""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=1000, ge=1)
    
    class Config:
        use_enum_values = True

class StockMetrics(BaseModel):
    price: float = Field(description="Current stock price")
    volume: int = Field(description="Trading volume")
    market_cap: float = Field(description="Market capitalization")

class SentimentAnalysis(BaseModel):
    score: float = Field(ge=-1, le=1, description="Sentiment score between -1 and 1")
    confidence: float = Field(ge=0, le=100, description="Confidence level in percentage")
    key_factors: List[str] = Field(description="Key factors influencing the sentiment")

class TradingSignal(BaseModel):
    ticker: str = Field(description="Stock ticker symbol")
    action: Literal["buy", "sell", "hold"] = Field(description="Trading action to take")
    target_price: float = Field(description="Target price for the stock")
    time_horizon: Literal["short_term", "medium_term", "long_term"]
    metrics: StockMetrics
    sentiment: SentimentAnalysis
    timestamp: datetime = Field(default_factory=datetime.now)
    reasoning: str = Field(description="Detailed reasoning for the trading decision")

class TradingAnalyzer:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._setup_llm()

    def _setup_llm(self):
        """Setup the LLM based on the provider"""
        if self.config.provider == ModelProvider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key or os.getenv("OPENAI_API_KEY"))
        elif self.config.provider == ModelProvider.ANTHROPIC:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.config.api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif self.config.provider == ModelProvider.GOOGLE:
            import google.generativeai as genai
            genai.configure(api_key=self.config.api_key or os.getenv("GOOGLE_API_KEY"))
            self.client = genai
        elif self.config.provider == ModelProvider.LOCAL:
            from langchain_community.llms import Ollama
            self.client = Ollama(model=self.config.model_name)

    def get_trading_signal(self, ticker: str, market_data: dict) -> TradingSignal:
        """Get trading signal for a given ticker"""
        prompt = self._create_prompt(ticker, market_data)
        
        try:
            llm_response = None  # Initialize response variable
            
            if self.config.provider == ModelProvider.OPENAI:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                llm_response = response.choices[0].message.content
            
            elif self.config.provider == ModelProvider.ANTHROPIC:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                llm_response = response.content[0].text
            
            elif self.config.provider == ModelProvider.GOOGLE:
                model = self.client.GenerativeModel(self.config.model_name)
                response = model.generate_content(prompt)
                llm_response = response.text
            
            elif self.config.provider == ModelProvider.LOCAL:
                llm_response = self.client.invoke(prompt)
            
            if llm_response is None:
                raise ValueError(f"No response received from {self.config.provider}")
                
            return self._process_llm_response(llm_response, ticker)
        
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return self._get_default_signal(ticker)

    def _create_prompt(self, ticker: str, market_data: dict) -> str:
        """Create prompt for the LLM"""
        return f"""Analyze the following market data for {ticker} and provide a trading signal in JSON format.
        
Market Data:
{json.dumps(market_data, indent=2)}

Provide your analysis in the following JSON format exactly:
{{
    "ticker": "{ticker}",
    "action": "buy" | "sell" | "hold",
    "target_price": float,
    "time_horizon": "short_term" | "medium_term" | "long_term",
    "metrics": {{
        "price": float,
        "volume": integer,
        "market_cap": float
    }},
    "sentiment": {{
        "score": float (-1 to 1),
        "confidence": float (0 to 100),
        "key_factors": [list of strings]
    }},
    "reasoning": "detailed string explanation"
}}
"""

    def _process_llm_response(self, llm_response: str, ticker: str) -> TradingSignal:
        """Process LLM response into TradingSignal"""
        try:
            # Extract JSON from the response (handles cases where LLM might add extra text)
            print("LLM RESPONSE RAW")
            print("")
            print("")
            print("")
            print(llm_response)
            print("")
            print("")
            print("")
            print("")
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = llm_response[json_start:json_end]
                return TradingSignal.model_validate_json(json_str)
            raise ValueError("No valid JSON found in response")
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return self._get_default_signal(ticker)

    def _get_default_signal(self, ticker: str) -> TradingSignal:
        """Get default trading signal when processing fails"""
        return TradingSignal(
            ticker=ticker,
            action="hold",
            target_price=0.0,
            time_horizon="medium_term",
            metrics=StockMetrics(price=0.0, volume=0, market_cap=0.0),
            sentiment=SentimentAnalysis(
                score=0.0,
                confidence=0.0,
                key_factors=["Error in processing response"]
            ),
            reasoning="Failed to process LLM response"
        )

def main():
    # Example configuration
    config = LLMConfig(
        provider=ModelProvider.GOOGLE,  # or ModelProvider.LOCAL for Ollama
        model_name="gemini-2.0-flash",    # or "llama2" for Ollama
        temperature=0.7,
        max_tokens=1000
    )

    # Initialize analyzer
    analyzer = TradingAnalyzer(config)

    # Example market data
    market_data = {
        "current_price": 175.25,
        "daily_volume": 55000000,
        "market_cap": 2750000000000,
        "pe_ratio": 28.5,
        "revenue_growth": 0.15,
        "profit_margin": 0.25,
        "analyst_ratings": {
            "buy": 25,
            "hold": 10,
            "sell": 2
        }
    }

    # Get trading signal
    signal = analyzer.get_trading_signal("AAPL", market_data)
    
    # Print results
    print("\nTrading Signal Analysis:")
    print(f"Ticker: {signal.ticker}")
    print(f"Action: {signal.action}")
    print(f"Target Price: ${signal.target_price}")
    print(f"Time Horizon: {signal.time_horizon}")
    print("\nMetrics:")
    print(f"Current Price: ${signal.metrics.price}")
    print(f"Volume: {signal.metrics.volume:,}")
    print(f"Market Cap: ${signal.metrics.market_cap:,.2f}")
    print("\nSentiment Analysis:")
    print(f"Score: {signal.sentiment.score:.2f}")
    print(f"Confidence: {signal.sentiment.confidence}%")
    print("Key Factors:")
    for factor in signal.sentiment.key_factors:
        print(f"- {factor}")
    print(f"\nReasoning: {signal.reasoning}")

if __name__ == "__main__":
    main()

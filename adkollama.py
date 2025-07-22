# Create the extended news sentiment analysis agent
import requests
import json
import feedparser
import schedule
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Data class for news items"""
    title: str
    summary: str
    link: str
    published: str
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None

@dataclass
class DailySummary:
    """Data class for daily news summary"""
    date: str
    total_articles: int
    positive_count: int
    negative_count: int
    neutral_count: int
    top_stories: List[NewsItem]
    overall_sentiment: str

def call_ollama(prompt, model_name="qwen3:14b", base_url="http://localhost:11434"):
    """Robust function to call Ollama API and handle NDJSON responses"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            stream=True,
            timeout=120  # Increased timeout for sentiment analysis
        )
        response.raise_for_status()
        
        # Ollama may return NDJSON (one JSON per line)
        responses = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        responses.append(data["response"])
                except json.JSONDecodeError:
                    continue
        
        return "".join(responses)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama: {str(e)}")
        return f"Error connecting to Ollama: {str(e)}"

class BaseEnvironment(ABC):
    """Base environment class"""
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def step(self, action: str) -> Dict[str, Any]:
        pass

class BaseAgent(ABC):
    """Base agent class"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> str:
        pass

class BaseTask(ABC):
    """Base task class"""
    
    @abstractmethod
    def get_initial_observation(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_complete(self, observation: Dict[str, Any]) -> bool:
        pass

class NewsEnvironment(BaseEnvironment):
    """Environment that handles news fetching and Ollama interaction"""
    
    def __init__(self, model_name="qwen3:14b", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.cnn_rss_url = "http://rss.cnn.com/rss/edition.rss"
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment"""
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    
    def fetch_cnn_feeds(self) -> List[NewsItem]:
        """Fetch news feeds from CNN RSS"""
        try:
            logger.info("Fetching CNN RSS feeds...")
            feed = feedparser.parse(self.cnn_rss_url)
            
            news_items = []
            for entry in feed.entries[:20]:  # Limit to top 20 stories
                news_item = NewsItem(
                    title=entry.title,
                    summary=entry.summary if hasattr(entry, 'summary') else entry.title,
                    link=entry.link,
                    published=entry.published if hasattr(entry, 'published') else str(datetime.now())
                )
                news_items.append(news_item)
            
            logger.info(f"Fetched {len(news_items)} news items from CNN")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching CNN feeds: {str(e)}")
            return []
    
    def analyze_sentiment(self, news_item: NewsItem) -> NewsItem:
        """Analyze sentiment of a single news item"""
        prompt = f"""
        Analyze the sentiment of this news article and provide a response in the following format:
        
        SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL]
        SCORE: [number between -1.0 and 1.0, where -1.0 is very negative, 0 is neutral, 1.0 is very positive]
        REASONING: [brief explanation]
        
        Article Title: {news_item.title}
        Article Summary: {news_item.summary}
        
        Please be objective and consider the overall tone and implications of the news.
        """
        
        response = call_ollama(prompt, self.model_name, self.base_url)
        
        # Parse the response
        try:
            lines = response.strip().split('\\n')
            sentiment = "NEUTRAL"
            score = 0.0
            
            for line in lines:
                if line.startswith("SENTIMENT:"):
                    sentiment = line.split(":", 1)[1].strip()
                elif line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        score = 0.0
            
            news_item.sentiment = sentiment
            news_item.sentiment_score = score
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {str(e)}")
            news_item.sentiment = "NEUTRAL"
            news_item.sentiment_score = 0.0
        
        return news_item
    
    def step(self, action: str) -> Dict[str, Any]:
        """Execute an action"""
        if action == "fetch_and_analyze":
            # Fetch news
            news_items = self.fetch_cnn_feeds()
            
            if not news_items:
                return {"error": "Failed to fetch news items", "news_items": []}
            
            # Analyze sentiment for each item
            logger.info("Starting sentiment analysis...")
            analyzed_items = []
            
            for i, item in enumerate(news_items):
                logger.info(f"Analyzing sentiment for article {i+1}/{len(news_items)}: {item.title[:50]}...")
                analyzed_item = self.analyze_sentiment(item)
                analyzed_items.append(analyzed_item)
                time.sleep(1)  # Rate limiting
            
            return {
                "news_items": analyzed_items,
                "total_items": len(analyzed_items),
                "timestamp": datetime.now().isoformat()
            }
        
        return {"error": "Unknown action"}

class NewsAnalysisAgent(BaseAgent):
    """Agent that performs daily news sentiment analysis"""
    
    def __init__(self, name="NewsAnalysisAgent"):
        super().__init__(name=name)
        self.environment = None
    
    def set_environment(self, environment: NewsEnvironment):
        """Set the environment for the agent"""
        self.environment = environment
    
    def act(self, observation: Dict[str, Any]) -> str:
        """Generate an action based on observation"""
        if observation.get("task") == "daily_news_analysis":
            return "fetch_and_analyze"
        return "fetch_and_analyze"
    
    def generate_daily_summary(self, news_items: List[NewsItem]) -> DailySummary:
        """Generate a daily summary from analyzed news items"""
        if not news_items:
            return DailySummary(
                date=datetime.now().strftime("%Y-%m-%d"),
                total_articles=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                top_stories=[],
                overall_sentiment="NEUTRAL"
            )
        
        # Count sentiments
        positive_count = sum(1 for item in news_items if item.sentiment == "POSITIVE")
        negative_count = sum(1 for item in news_items if item.sentiment == "NEGATIVE")
        neutral_count = sum(1 for item in news_items if item.sentiment == "NEUTRAL")
        
        # Determine overall sentiment
        if positive_count > negative_count:
            overall_sentiment = "POSITIVE"
        elif negative_count > positive_count:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"
        
        # Get top stories (sorted by sentiment score)
        top_stories = sorted(news_items, key=lambda x: abs(x.sentiment_score or 0), reverse=True)[:5]
        
        return DailySummary(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_articles=len(news_items),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            top_stories=top_stories,
            overall_sentiment=overall_sentiment
        )

class DailyNewsTask(BaseTask):
    """Daily news analysis task"""
    
    def __init__(self):
        self.completed = False
    
    def get_initial_observation(self) -> Dict[str, Any]:
        """Get the initial observation for the agent"""
        return {
            "task": "daily_news_analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    def is_complete(self, observation: Dict[str, Any]) -> bool:
        """Check if the task is complete"""
        return "news_items" in observation and not observation.get("error")

class NewsAgentRunner:
    """Orchestrates the news analysis agent"""
    
    def __init__(self, agent: NewsAnalysisAgent, environment: NewsEnvironment, task: DailyNewsTask):
        self.agent = agent
        self.environment = environment
        self.task = task
        self.summary_file = Path("daily_summaries.json")
    
    def run_daily_analysis(self) -> Optional[DailySummary]:
        """Run the complete daily news analysis"""
        logger.info("=" * 80)
        logger.info("STARTING DAILY NEWS SENTIMENT ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Reset environment
            env_state = self.environment.reset()
            logger.info(f"Environment initialized: {env_state}")
            
            # Get initial observation from task
            observation = self.task.get_initial_observation()
            logger.info(f"Starting daily analysis at: {observation['timestamp']}")
            
            # Agent acts based on observation
            action = self.agent.act(observation)
            logger.info(f"Agent action: {action}")
            
            # Environment processes the action
            result = self.environment.step(action)
            
            if result.get("error"):
                logger.error(f"Analysis failed: {result['error']}")
                return None
            
            # Generate summary
            news_items = result.get("news_items", [])
            summary = self.agent.generate_daily_summary(news_items)
            
            # Save summary
            self.save_summary(summary)
            
            # Display summary
            self.display_summary(summary)
            
            logger.info("Daily analysis completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Error during daily analysis: {str(e)}")
            return None
    
    def save_summary(self, summary: DailySummary):
        """Save daily summary to file"""
        try:
            summaries = []
            if self.summary_file.exists():
                with open(self.summary_file, 'r') as f:
                    summaries = json.load(f)
            
            # Convert summary to dict
            summary_dict = {
                "date": summary.date,
                "total_articles": summary.total_articles,
                "positive_count": summary.positive_count,
                "negative_count": summary.negative_count,
                "neutral_count": summary.neutral_count,
                "overall_sentiment": summary.overall_sentiment,
                "top_stories": [
                    {
                        "title": story.title,
                        "sentiment": story.sentiment,
                        "sentiment_score": story.sentiment_score,
                        "link": story.link
                    }
                    for story in summary.top_stories
                ]
            }
            
            # Add to summaries
            summaries.append(summary_dict)
            
            # Keep only last 30 days
            summaries = summaries[-30:]
            
            with open(self.summary_file, 'w') as f:
                json.dump(summaries, f, indent=2)
                
            logger.info(f"Summary saved to {self.summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
    
    def display_summary(self, summary: DailySummary):
        """Display the daily summary"""
        print("\\n" + "=" * 80)
        print(f"DAILY NEWS SENTIMENT SUMMARY - {summary.date}")
        print("=" * 80)
        print(f"Total Articles Analyzed: {summary.total_articles}")
        print(f"Overall Sentiment: {summary.overall_sentiment}")
        print(f"Positive: {summary.positive_count} | Negative: {summary.negative_count} | Neutral: {summary.neutral_count}")
        print("\\n" + "-" * 80)
        print("TOP STORIES:")
        print("-" * 80)
        
        for i, story in enumerate(summary.top_stories, 1):
            print(f"{i}. {story.title}")
            print(f"   Sentiment: {story.sentiment} (Score: {story.sentiment_score:.2f})")
            print(f"   Link: {story.link}")
            print()
        
        print("=" * 80)

class DailyNewsScheduler:
    """Scheduler for daily news analysis"""
    
    def __init__(self):
        self.runner = None
        self.setup_components()
    
    def setup_components(self):
        """Set up all components"""
        environment = NewsEnvironment()
        agent = NewsAnalysisAgent()
        task = DailyNewsTask()
        
        agent.set_environment(environment)
        self.runner = NewsAgentRunner(agent, environment, task)
    
    def run_scheduled_analysis(self):
        """Run the scheduled analysis"""
        logger.info("Scheduled news analysis triggered at 8:00 AM")
        if self.runner:
            self.runner.run_daily_analysis()
        else:
            logger.error("Runner not initialized")
    
    def start_scheduler(self):
        """Start the daily scheduler"""
        logger.info("Starting daily news analysis scheduler...")
        logger.info("Scheduled to run every day at 8:00 AM")
        
        # Schedule the job
        schedule.every().day.at("08:00").do(self.run_scheduled_analysis)
        
        # Run once immediately for testing
        logger.info("Running initial analysis...")
        self.run_scheduled_analysis()
        
        # Keep the scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

def main():
    """Main function to start the daily news analysis system"""
    print("=" * 80)
    print("DAILY NEWS SENTIMENT ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Test Ollama connection first
    print("\\n1. Testing Ollama connection...")
    test_response = call_ollama("Hello, can you hear me?")
    print(f"Test response: {test_response[:200]}...")
    
    if test_response.startswith("Error"):
        print("❌ Ollama connection failed. Make sure Ollama is running with qwen3:14b")
        return
    
    print("✅ Ollama connection successful!")
    
    # Start the scheduler
    scheduler = DailyNewsScheduler()
    
    try:
        scheduler.start_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        print("\\n👋 Daily news analysis system stopped.")

if __name__ == "__main__":
    main()

"""
MIND (Microsoft News Dataset) Loader.

MIND Large contains:
- 101,527 news articles
- 876,356 users
- 2,792,878 impressions

Data format:
- news.tsv: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
- behaviors.tsv: impression_id, user_id, time, history, impressions
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)

# Default MIND data path
MIND_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "mind"


@dataclass
class NewsItem:
    """A news article."""
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str = ""


@dataclass
class UserBehavior:
    """A user behavior record."""
    impression_id: str
    user_id: str
    time: str
    history: List[str]
    impressions: List[Tuple[str, int]]  # (news_id, clicked)


def load_news(data_path: Path) -> Dict[str, NewsItem]:
    """Load news articles from news.tsv."""
    news_file = data_path / "news.tsv"

    if not news_file.exists():
        raise FileNotFoundError(f"News file not found: {news_file}")

    news_dict = {}

    with open(news_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            news_id = parts[0]
            category = parts[1] if len(parts) > 1 else ""
            subcategory = parts[2] if len(parts) > 2 else ""
            title = parts[3] if len(parts) > 3 else ""
            abstract = parts[4] if len(parts) > 4 else ""
            url = parts[5] if len(parts) > 5 else ""

            news_dict[news_id] = NewsItem(
                news_id=news_id,
                category=category,
                subcategory=subcategory,
                title=title,
                abstract=abstract,
                url=url,
            )

    return news_dict


def load_behaviors(data_path: Path, max_samples: int = None) -> List[UserBehavior]:
    """Load user behaviors from behaviors.tsv."""
    behaviors_file = data_path / "behaviors.tsv"

    if not behaviors_file.exists():
        raise FileNotFoundError(f"Behaviors file not found: {behaviors_file}")

    behaviors = []

    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples * 2:  # Over-sample to filter later
                break

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            impression_id = parts[0]
            user_id = parts[1]
            time = parts[2]
            history = parts[3].split() if parts[3] else []
            impressions_str = parts[4].split() if parts[4] else []

            # Parse impressions: "N12345-1" means news_id=N12345, clicked=1
            impressions = []
            for imp in impressions_str:
                if '-' in imp:
                    news_id, clicked = imp.rsplit('-', 1)
                    impressions.append((news_id, int(clicked)))

            # Only include if there's history and clicked items
            if history and any(c == 1 for _, c in impressions):
                behaviors.append(UserBehavior(
                    impression_id=impression_id,
                    user_id=user_id,
                    time=time,
                    history=history,
                    impressions=impressions,
                ))

    return behaviors


def load_mind_dataset(
    data_path: Path = None,
    split: str = "train",
    size: str = "large",
    n_samples: int = None,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Load MIND dataset for recommendation experiments.

    Args:
        data_path: Path to MIND data directory
        split: "train" or "dev"
        size: "small" or "large"
        n_samples: Number of samples to return (None for all)
        seed: Random seed for sampling

    Returns:
        samples: List of recommendation samples
        news_items: Dict mapping news_id to news metadata
    """
    if data_path is None:
        data_path = MIND_DATA_PATH

    data_path = Path(data_path)

    # Determine data directory
    if size == "large":
        split_dir = data_path / f"MINDlarge_{split}"
    else:
        split_dir = data_path / f"MINDsmall_{split}"

    if not split_dir.exists():
        raise FileNotFoundError(f"MIND data not found: {split_dir}")

    logger.info(f"Loading MIND {size} {split} from {split_dir}")

    # Load news
    news_dict = load_news(split_dir)
    logger.info(f"Loaded {len(news_dict)} news articles")

    # Convert to dict format for compatibility
    news_items = {
        news_id: {
            'title': item.title,
            'abstract': item.abstract,
            'category': item.category,
            'subcategory': item.subcategory,
        }
        for news_id, item in news_dict.items()
    }

    # Load behaviors
    behaviors = load_behaviors(split_dir, max_samples=n_samples * 5 if n_samples else None)
    logger.info(f"Loaded {len(behaviors)} behavior records")

    # Convert to recommendation samples
    random.seed(seed)
    samples = []

    for behavior in behaviors:
        # Filter history to only include items in news_dict
        valid_history = [h for h in behavior.history if h in news_dict]

        if len(valid_history) < 3:
            continue

        # Get clicked items
        clicked_items = [news_id for news_id, clicked in behavior.impressions if clicked == 1 and news_id in news_dict]

        if not clicked_items:
            continue

        # Create samples for each clicked item
        for ground_truth in clicked_items:
            samples.append({
                'user_id': behavior.user_id,
                'history': valid_history,
                'ground_truth': ground_truth,
            })

        if n_samples and len(samples) >= n_samples:
            break

    # Shuffle and limit
    random.shuffle(samples)
    if n_samples:
        samples = samples[:n_samples]

    logger.info(f"Created {len(samples)} recommendation samples")

    return samples, news_items


def load_mind_for_trie_experiment(
    n_samples: int = 1000,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[str, Dict], List[str]]:
    """
    Load MIND data formatted for Trie-augmented LLM experiments.

    Returns:
        samples: List of recommendation samples with history and ground_truth
        news_items: Dict of news metadata
        all_items: List of all news IDs
    """
    samples, news_items = load_mind_dataset(
        size="large",
        split="train",
        n_samples=n_samples,
        seed=seed,
    )

    all_items = list(news_items.keys())

    return samples, news_items, all_items


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("Testing MIND loader...")

    try:
        samples, news_items, all_items = load_mind_for_trie_experiment(n_samples=100)

        print(f"\nLoaded:")
        print(f"  - {len(samples)} samples")
        print(f"  - {len(news_items)} news items")
        print(f"  - {len(all_items)} total items")

        if samples:
            print(f"\nSample 0:")
            print(f"  User: {samples[0]['user_id']}")
            print(f"  History: {samples[0]['history'][:5]}...")
            print(f"  Ground truth: {samples[0]['ground_truth']}")

            gt = samples[0]['ground_truth']
            if gt in news_items:
                print(f"  GT Title: {news_items[gt]['title'][:50]}...")

    except FileNotFoundError as e:
        print(f"MIND data not found: {e}")
        print("Please download MIND dataset from: https://msnews.github.io/")

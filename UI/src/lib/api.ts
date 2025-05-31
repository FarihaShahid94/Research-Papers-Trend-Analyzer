import type { Category, Keyword, PaperDataPoint, SentimentDataPoint, ForecastDataPoint, SentimentExtreme, SystemStats } from '@/types';
import { addMonths, format } from 'date-fns';
import { mockCategories } from './mock-data';
 
let trendingKeywords: Keyword[] = [];
 
export async function loadTrendingKeywords(): Promise<Keyword[]> {
    if (trendingKeywords.length) return trendingKeywords;
    const res = await fetch('http://localhost:8000/trending_words?top=10');
    if (!res.ok) throw new Error('Fetch failed');
  
    const data = await res.json();
    console.log("Trending keywords (raw):", data);
  
    trendingKeywords = data.top_trending_words.map(
      ([text, frequency]: [string, number]) => ({ text, frequency })
    );
  
    return trendingKeywords;
  }
  
  
 
export async function loadTrendingKeywordsByCategory(category: string, top = 10): Promise<Keyword[]> {
    try {
      const response = await fetch(`http://localhost:8000/keywords/${category}?top=${top}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch keywords: ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log("Trending by category (raw):", data);
  
      // Transform response
      const keywords: Keyword[] = data.top_keywords.map(
        ([text, frequency]: [string, number]) => ({
          text,
          frequency,
          category: data.category,
        })
      );
  
      return keywords;
    } catch (error) {
      console.error('Error loading trending keywords:', error);
      return [];
    }
  }

  export async function loadTfIdfKeywordsByCategory(category: string, top = 10): Promise<Keyword[]> {
    try {
      const response = await fetch(`http://localhost:8000/tfidf/${category}?top=${top}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch keywords: ${response.statusText}`);
      }
  
      const data = await response.json();
      console.log("Trending tfidf by category (raw):", data);
  
      const keywordArray = data.top_tfidf_words;
      if (!keywordArray || !Array.isArray(keywordArray)) {
        console.warn("Missing or malformed 'top_tfidf_words' in response:", data);
        return [];
      }
  
      const keywords: Keyword[] = keywordArray.map(
        ([text, frequency]: [string, number]) => ({
          text,
          frequency,
          category: data.category ?? category,
        })
      );
  
      return keywords;
    } catch (error) {
      console.error('Error loading tf-idf keywords:', error);
      return [];
    }
  }
  


  export async function getKeywordFrequency(category: string, word: string): Promise<number | null> {
    if (!category || !word) return null;
   
    try {
      const response = await fetch(`http://localhost:8000/keywords/${encodeURIComponent(category)}/word/${encodeURIComponent(word)}`);
      if (!response.ok) {
        console.error('Failed to fetch keyword frequency:', response.statusText);
        return null;
      }
      const data = await response.json();
      // Assuming API returns { frequency: number }
      return data.frequency ?? null;
    } catch (error) {
      console.error('Error fetching keyword frequency:', error);
      return null;
    }
  }

  type RawSystemStats = {
    total_categories: number;
    total_unique_keywords: number;
    total_papers_counted: number;
    average_tfidf_per_category: Record<string, number>;
  };
  

  export async function loadSystemStats() {
    const res = await fetch('http://localhost:8000/stats');
    if (!res.ok) {
      throw new Error('Failed to fetch system stats');
    }
  
    const data = await res.json() as RawSystemStats;
  
    const mappedCategoryIds = new Set(mockCategories.map(cat => cat.id));
    const filteredTFIDFValues = Object.entries(data.average_tfidf_per_category)
      .filter(([key]) => mappedCategoryIds.has(key))
      .map(([, value]) => value);
  
    const averageTFIDF = filteredTFIDFValues.length
      ? filteredTFIDFValues.reduce((sum, val) => sum + val, 0) / filteredTFIDFValues.length
      : 0;
  
    return {
      total_categories: data.total_categories,
      total_unique_keywords: data.total_unique_keywords,
      total_papers_counted: data.total_papers_counted,
      average_tfidf: averageTFIDF.toFixed(2),
    };
  }

export async function loadPapersOverTime(): Promise<PaperDataPoint[]> {
  const res = await fetch('http://localhost:8000/papers_over_time');
  if (!res.ok) throw new Error('Failed to fetch papers over time');

  const raw = await res.json();
  return raw.papers_over_time.map(([timestamp, count]: [string, number]) => ({
    date: timestamp.slice(0, 7), // Extract YYYY-MM
    count,
  }));
}

export async function loadPapersOverTimeRange(start: string, end: string): Promise<PaperDataPoint[]> {
  const encodedStart = encodeURIComponent(start);
  const encodedEnd = encodeURIComponent(end);
  const res = await fetch(`http://localhost:8000/papers_over_time/${encodedStart}/${encodedEnd}`);
  if (!res.ok) throw new Error('Failed to fetch papers over time range');

  const raw = await res.json();
  return raw.papers_over_time_range.map(([timestamp, count]: [string, number]) => ({
    date: timestamp.slice(0, 7),
    count,
  }));
}

export async function loadForecastData() {
    const res = await fetch('http://localhost:8000/forecast');
    const json = await res.json();
    return json.forecast.map((item: any) => ({
      interval: new Date(item.ds).toLocaleString(), // or format it however you want
      predictedVolume: item.yhat,
    }));
  }
  
  export async function loadForecastSummary() {
    const res = await fetch('http://localhost:8000/forecast/summary');
    const json = await res.json();
    return json.average_predicted_papers_next_5_intervals;
  }

  
  // lib/api.ts
export async function loadAverageTfIdfScores(): Promise<Record<string, number>> {
    const res = await fetch('http://localhost:8000/tfidf');
    if (!res.ok) throw new Error('Failed to load average TF-IDF scores');
    const json = await res.json();
    return json.average_tfidf_per_category;
  }
  

  export async function loadTotalPapersPerCategory(): Promise<{ category: string; count: number }[]> {
    const res = await fetch('http://localhost:8000/aggregation/total_papers_per_category');
    if (!res.ok) throw new Error('Failed to fetch total papers per category');
    const json = await res.json();
  
    const totalMap = json.total_papers_per_category;
  
    // Map to array using mock category info
    return mockCategories
      .map(cat => ({
        category: cat.name,
        id: cat.id,
        count: totalMap[cat.id] ?? 0
      }))
      .filter(item => item.count > 0);
  }

  export async function loadCoolWordCounts(): Promise<{ category: string; count: number }[]> {
    const response = await fetch('http://localhost:8000/coolword-counts');
  
    if (!response.ok) {
      throw new Error('Failed to fetch cool word counts');
    }
  
    const rawData = await response.json();
  
    return Object.entries(rawData).map(([category, data]: any) => ({
      category,
      count: data.cool_word_count,
    }));
  }

  
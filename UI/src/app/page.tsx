'use client';
 
import { useState, useEffect, useMemo } from 'react';
import type { DateRange } from "react-day-picker";
import AppLayout from '@/components/layout/app-layout';
import CategorySelector from '@/components/dashboard/category-selector';
import TrendingKeywordsPanel from '@/components/dashboard/trending-keywords-panel';
import KeywordFrequencyLookup from '@/components/dashboard/keyword-frequency-lookup';
import SentimentOverview from '@/components/dashboard/sentiment-overview';
import SentimentOverTimeChart from '@/components/dashboard/sentiment-over-time-chart';
import SentimentAtTime from '@/components/dashboard/sentiment-at-time';
import PapersOverTimeChart from '@/components/dashboard/papers-over-time-chart';
import ForecastPanel from '@/components/dashboard/forecast-panel';
import OverallTrendingWords from '@/components/dashboard/overall-trending-words';
import SentimentExtremes from '@/components/dashboard/sentiment-extremes';
import SystemStatsPanel from '@/components/dashboard/system-stats-panel';
import CoolWordsCloud from '@/components/dashboard/coolwordscloud';
import CoolWordCountChart from '@/components/dashboard/cool-word-count-chart';

 
import { loadTfIdfKeywordsByCategory, loadTrendingKeywords } from '@/lib/api';
import { loadTrendingKeywordsByCategory } from '@/lib/api';
import { loadSystemStats } from '@/lib/api';
import { loadPapersOverTime } from '@/lib/api';
import { loadPapersOverTimeRange } from '@/lib/api';
import { loadForecastData } from '@/lib/api';
import { loadForecastSummary } from '@/lib/api';

import TfIdfExtremes from '@/components/dashboard/tfidf-extremes';
import { loadAverageTfIdfScores } from '@/lib/api';
import { loadTotalPapersPerCategory } from '@/lib/api';
import TotalPapersPerCategoryChart from '@/components/dashboard/total-papers-per-category-chart';
import { loadCoolWordCounts } from '@/lib/api';



import { SystemStats } from '@/types';
 
import {
  mockCategories,
  mockTrendingKeywords,
  mockOverallTrendingWords,
  mockSentimentOverTime,
  mockPapersOverTime,
  mockForecastData,
  mockSentimentExtremes
} from '@/lib/mock-data';
import type { Keyword, PaperDataPoint, SentimentDataPoint, ForecastDataPoint } from '@/types';
import TfIdfTrendingWords from '@/components/dashboard/tfidf-trending-words';
 
export default function ArxivInsightsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [dateRange, setDateRange] = useState<DateRange | undefined>(undefined);
  const [keywords, setKeywords] = useState<Keyword[]>([]);
  const [tfidfKeywords, setTfidfKeywords] = useState<Keyword[]>([]);
  const [loading, setLoading] = useState(false);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [papersOverTime, setPapersOverTime] = useState<PaperDataPoint[]>([]);
  const [forecastData, setForecastData] = useState<ForecastDataPoint[]>([]);
  const [averageForecast, setAverageForecast] = useState<number>(0);
  const [tfidfExtremes, setTfidfExtremes] = useState<{ highest: { category: string, score: number }, lowest: { category: string, score: number } } | null>(null);
  
  const [totalPapersPerCategory, setTotalPapersPerCategory] = useState<{ category: string; count: number }[]>([]);
  const [coolWordCounts, setCoolWordCounts] = useState<{ category: string; count: number }[]>([]);
  
  useEffect(() => {
    async function fetchCoolWordCounts() {
      try {
        const data = await loadCoolWordCounts();
        setCoolWordCounts(data);
      } catch (error) {
        console.error('Failed to fetch cool word counts:', error);
      }
    }
  
    fetchCoolWordCounts();
  }, []);
  

  useEffect(() => {
    async function fetchTotalPapersPerCategory() {
      try {
        const data = await loadTotalPapersPerCategory();
        setTotalPapersPerCategory(data);
      } catch (err) {
        console.error('Failed to fetch total papers per category', err);
      }
    }
  
    fetchTotalPapersPerCategory();
  }, []);
  

useEffect(() => {
  async function fetchTfIdf() {
    try {
      const data = await loadAverageTfIdfScores();
      const filtered = Object.entries(data).filter(([cat]) =>
        mockCategories.some(c => c.id === cat)
      );

      if (filtered.length > 0) {
        const sorted = filtered.sort((a, b) => b[1] - a[1]);
        setTfidfExtremes({
          highest: { category: sorted[0][0], score: sorted[0][1] },
          lowest: { category: sorted[sorted.length - 1][0], score: sorted[sorted.length - 1][1] },
        });
      }
    } catch (err) {
      console.error('Failed to fetch TF-IDF extremes', err);
    }
  }

  fetchTfIdf();
}, []);


  useEffect(() => {
    async function fetchForecast() {
      try {
        const [data, avg] = await Promise.all([
          loadForecastData(),
          loadForecastSummary(),
        ]);
        setForecastData(data);
        setAverageForecast(avg);
      } catch (err) {
        console.error('Failed to load forecast data:', err);
      }
    }
  
    fetchForecast();
  }, []);
  
useEffect(() => {
  async function fetchPapers() {
    try {
      if (dateRange?.from && dateRange?.to) {
        const start = dateRange.from.toISOString().slice(0, 19).replace('T', ' ');
        const end = dateRange.to.toISOString().slice(0, 19).replace('T', ' ');
        const data = await loadPapersOverTimeRange(start, end);
        setPapersOverTime(data);
      } else {
        const data = await loadPapersOverTime();
        setPapersOverTime(data);
      }
    } catch (err) {
      console.error('Failed to load papers over time:', err);
    }
  }

  fetchPapers();
}, [dateRange]);


  useEffect(() => {
    loadSystemStats()
      .then(setSystemStats)
      .catch((err : any) => console.error('Failed to load system stats:', err));
  }, []);
 
  // Fetch trending keywords from the API
  useEffect(() => {
    loadTrendingKeywords().then(setKeywords).catch(console.error);
  }, []);
 
  const categoryName = useMemo(() => {
    if (!selectedCategory) return undefined;
    return mockCategories.find(cat => cat.id === selectedCategory)?.name;
  }, [selectedCategory]);
 
  // Filtered data based on selectedCategory
  // In a real app, these would be fetched or recalculated
  const filteredTrendingKeywords = useMemo(() => {
    if (loading) return [];
    return keywords.slice(0, 5);
  }, [keywords, loading]);
 
useEffect(() => {
    async function fetchKeywords() {
      setLoading(true);
      try {
        if (!selectedCategory) {
          // Fetch overall trending keywords if no category selected
          const allKeywords = await loadTrendingKeywords();
          setKeywords(allKeywords);
        } else {
          // Fetch keywords by selected category
          const categoryKeywords = await loadTrendingKeywordsByCategory(selectedCategory, 10);
          setKeywords(categoryKeywords);
        }
      } catch (error) {
        console.error('Failed to load keywords:', error);
        setKeywords([]);
      } finally {
        setLoading(false);
      }
    }
    fetchKeywords();
  }, [selectedCategory]);

  useEffect(() => {
    async function fetchKeywords() {
      setLoading(true);
      try {
        if (!selectedCategory) {
          // Fetch overall trending keywords if no category selected
          // const allKeywords = await loadTrendingKeywords();
          // setKeywords(allKeywords);
        } else {
          // Fetch keywords by selected category
          const categoryKeywords = await loadTfIdfKeywordsByCategory(selectedCategory, 10);
          setTfidfKeywords(categoryKeywords);
        }
      } catch (error) {
        console.error('Failed to load keywords:', error);
        setKeywords([]);
      } finally {
        setLoading(false);
      }
    }
    fetchKeywords();
  }, [selectedCategory]);
 
 
  const filteredSentimentOverTime = useMemo(() => {
    if (!selectedCategory) return mockSentimentOverTime;
    // Example: filter or fetch category-specific sentiment
    return mockSentimentOverTime.map(d => ({...d, sentiment: d.sentiment * (Math.random() * 0.4 + 0.8) })).slice(0,12); // Dummy variation
  }, [selectedCategory]);
 
  const filteredPapersOverTime = papersOverTime;
 
  const averageSentimentScore = useMemo(() => {
    const dataToAverage = selectedCategory ? filteredSentimentOverTime : mockSentimentOverTime;
    if (dataToAverage.length === 0) return 0;
    const sum = dataToAverage.reduce((acc, curr) => acc + curr.sentiment, 0);
    return sum / dataToAverage.length;
  }, [selectedCategory, filteredSentimentOverTime]);
 
  const totalSentimentDataPoints = useMemo(() => {
     const data = selectedCategory ? filteredSentimentOverTime : mockSentimentOverTime;
     return data.length * 100; // Example calculation
  },[selectedCategory, filteredSentimentOverTime]);
 
 
  return (
    <AppLayout>
      <div className="space-y-8">
        {/* Top Controls / Overview Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <CategorySelector
            categories={mockCategories}
            selectedCategory={selectedCategory}
            onSelectCategory={setSelectedCategory}
          />
          {systemStats && <SystemStatsPanel stats={systemStats} />}
        </div>
 
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1 */}
          <div className="lg:col-span-2 space-y-6">
            <PapersOverTimeChart
              data={filteredPapersOverTime}
              dateRange={dateRange}
              onDateRangeChange={setDateRange}
              categoryName={categoryName}
            />
            <TotalPapersPerCategoryChart data={totalPapersPerCategory} />
          </div>
 
          {/* Column 2 (Sidebar-like) */}
          <div className="space-y-6">
            <TrendingKeywordsPanel
              keywords={filteredTrendingKeywords}
              title={selectedCategory ? `Trending in ${categoryName}` : 'Trending Keywords'}
            />
            <KeywordFrequencyLookup selectedCategory={selectedCategory} />
            <ForecastPanel
              predictions={forecastData}
              averagePrediction={averageForecast}
            />
          </div>
        </div>
       
        {/* Bottom Row / Global Insights */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <OverallTrendingWords words={keywords.slice(0,5)} />
          <TfIdfTrendingWords words={tfidfKeywords} />
          {tfidfExtremes && (
            <TfIdfExtremes highest={tfidfExtremes.highest} lowest={tfidfExtremes.lowest} />
          )}
          <CoolWordsCloud />
        </div>
        <div className="mt-6">
          <CoolWordCountChart data={coolWordCounts} />
        </div>
      </div>
    </AppLayout>
  );
}
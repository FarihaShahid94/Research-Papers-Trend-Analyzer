import type { DateRange } from "react-day-picker";

export interface Category {
  id: string;
  name: string;
}

export interface Keyword {
  text: string;
  frequency: number;
  category?: string; // Optional: if keyword is specific to a category in some contexts
}

export interface TimeSeriesDataPoint {
  date: string; // Using string for dates like "YYYY-MM" or "YYYY-MM-DD"
  value: number;
}

export interface SentimentDataPoint {
  date: string;
  sentiment: number; // e.g., -1 to 1 or 0 to 1
  category?: string;
}

export interface PaperDataPoint {
  date: string; // e.g., "YYYY-MM"
  count: number;
  category?: string;
}

export interface ForecastDataPoint {
  interval: string; // e.g., "Next Month", "Q1 2025"
  predictedVolume: number;
}

export interface SystemStats {
  total_categories: number;
  total_unique_keywords: number;
  total_papers_counted: number;
  average_tfidf: number | string;
}

export interface SentimentExtreme {
  category: string;
  score: number;
}

// For DateRangePicker prop
export type DateRangePickerProps = {
  dateRange?: DateRange;
  onDateRangeChange: (dateRange: DateRange | undefined) => void;
  className?: string;
};

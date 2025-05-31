import type { Category, Keyword, PaperDataPoint, SentimentDataPoint, ForecastDataPoint, SentimentExtreme, SystemStats } from '@/types';
import { addMonths, format } from 'date-fns';

export const mockCategories: Category[] = [
  { id: 'cs.AI', name: 'Artificial Intelligence' },
  { id: 'cs.AR', name: 'Hardware Architecture' },
  { id: 'cs.CC', name: 'Computational Complexity' },
  { id: 'cs.CE', name: 'Computational Engineering, Finance, and Science' },
  { id: 'cs.CG', name: 'Computational Geometry' },
  { id: 'cs.CL', name: 'Computation and Language' },
  { id: 'cs.CR', name: 'Cryptography and Security' },
  { id: 'cs.CV', name: 'Computer Vision and Pattern Recognition' },
  { id: 'cs.CY', name: 'Computers and Society' },
  { id: 'cs.DB', name: 'Databases' },
  { id: 'cs.DC', name: 'Distributed, Parallel, and Cluster Computing' },
  { id: 'cs.DL', name: 'Digital Libraries' },
  { id: 'cs.DM', name: 'Discrete Mathematics' },
  { id: 'cs.DS', name: 'Data Structures and Algorithms' },
  { id: 'cs.ET', name: 'Emerging Technologies' },
  { id: 'cs.FL', name: 'Formal Languages and Automata Theory' },
  { id: 'cs.GL', name: 'General Literature' },
  { id: 'cs.GR', name: 'Graphics' },
  { id: 'cs.GT', name: 'Computer Science and Game Theory' },
  { id: 'cs.HC', name: 'Human-Computer Interaction' },
  { id: 'cs.IR', name: 'Information Retrieval' },
  { id: 'cs.IT', name: 'Information Theory' },
  { id: 'cs.LG', name: 'Machine Learning' },
  { id: 'cs.LO', name: 'Logic in Computer Science' },
  { id: 'cs.MA', name: 'Multiagent Systems' },
  { id: 'cs.MM', name: 'Multimedia' },
  { id: 'cs.MS', name: 'Mathematical Software' },
  { id: 'cs.NA', name: 'Numerical Analysis' },
  { id: 'cs.NE', name: 'Neural and Evolutionary Computing' },
  { id: 'cs.NI', name: 'Networking and Internet Architecture' },
  { id: 'cs.OH', name: 'Other Computer Science' },
  { id: 'cs.OS', name: 'Operating Systems' },
  { id: 'cs.PF', name: 'Performance' },
  { id: 'cs.PL', name: 'Programming Languages' },
  { id: 'cs.RO', name: 'Robotics' },
  { id: 'cs.SC', name: 'Symbolic Computation' },
  { id: 'cs.SD', name: 'Sound' },
  { id: 'cs.SE', name: 'Software Engineering' },
  { id: 'cs.SI', name: 'Social and Information Networks' },
  { id: 'cs.SY', name: 'Systems and Control' },
];


export const mockTrendingKeywords: Keyword[] = [
  { text: 'Large Language Models', frequency: 1200, category: 'ai' },
  { text: 'Diffusion Models', frequency: 950, category: 'ai' },
  { text: 'Quantum Entanglement', frequency: 800, category: 'phy' },
  { text: 'Reinforcement Learning', frequency: 750, category: 'ml' },
  { text: 'Graph Theory', frequency: 600, category: 'math' },
];

export const mockOverallTrendingWords: Keyword[] = [
  { text: 'Transformer', frequency: 2500 },
  { text: 'Scalability', frequency: 1800 },
  { text: 'Ethics', frequency: 1500 },
  { text: 'Bias', frequency: 1400 },
  { text: 'Sustainability', frequency: 1300 },
];

const generateMonthlyData = (startDate: Date, months: number, valueGenerator: (index: number) => number, key: 'sentiment' | 'count'): Array<SentimentDataPoint | PaperDataPoint> => {
  return Array.from({ length: months }, (_, i) => {
    const date = addMonths(startDate, i);
    if (key === 'sentiment') {
      return {
        date: format(date, 'yyyy-MM'),
        sentiment: valueGenerator(i),
      };
    }
    return {
      date: format(date, 'yyyy-MM'),
      count: valueGenerator(i),
    };
  });
};

const baseDate = new Date(2023, 0, 1); // Start from Jan 2023

export const mockSentimentOverTime: SentimentDataPoint[] = generateMonthlyData(
  baseDate,
  12,
  (i) => 0.5 + Math.sin(i / 3) * 0.2 + (Math.random() - 0.5) * 0.1, // Simulated sentiment trend
  'sentiment'
) as SentimentDataPoint[];

export const mockPapersOverTime: PaperDataPoint[] = generateMonthlyData(
  baseDate,
  12,
  (i) => 100 + i * 10 + Math.floor(Math.random() * 20), // Simulated paper count trend
  'count'
) as PaperDataPoint[];

export const mockForecastData: ForecastDataPoint[] = [
  { interval: 'Next Month', predictedVolume: 220 },
  { interval: 'Month +2', predictedVolume: 235 },
  { interval: 'Month +3', predictedVolume: 250 },
  { interval: 'Month +4', predictedVolume: 240 },
  { interval: 'Month +5', predictedVolume: 260 },
];

export const mockSentimentExtremes: { highest: SentimentExtreme; lowest: SentimentExtreme } = {
  highest: { category: 'AI Ethics & Society', score: 0.85 },
  lowest: { category: 'Advanced Cryptography', score: 0.42 },
};

// export const mockSystemStats: SystemStats = {
//   categories: mockCategories.length,
//   keywords: 15782,
//   papers: 230567,
//   sentimentPoints: 180432,
// };

export const getKeywordFrequency = (keyword: string): number | null => {
  if (!keyword) return null;
  // Simulate API call or DB lookup
  if (keyword.toLowerCase() === 'transformer') return 2500;
  if (keyword.toLowerCase() === 'python') return 5000;
  return Math.floor(Math.random() * 1000);
}

export const getSentimentAtTime = (time: string): number | null => {
  if (!time) return null;
  // Simulate finding sentiment for a specific time (e.g., month "YYYY-MM")
  const found = mockSentimentOverTime.find(p => p.date === time);
  if (found) return found.sentiment;
  return parseFloat((Math.random() * 0.4 + 0.3).toFixed(2)); // Random sentiment if not found
}

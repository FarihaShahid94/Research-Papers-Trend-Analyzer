import type { Keyword } from '@/types';
import TrendingKeywordsPanel from './trending-keywords-panel'; // Reusing the component
import { Globe } from 'lucide-react';

interface OverallTrendingWordsProps {
  words: Keyword[];
}

export default function TfIdfTrendingWords({ words }: OverallTrendingWordsProps) {
  return (
    <TrendingKeywordsPanel 
      keywords={words} 
      title="TF IDF Trending Words"
      icon={<Globe className="mr-2 h-5 w-5 text-primary" />}
    />
  );
}

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Gauge, Sigma } from 'lucide-react';

interface SentimentOverviewProps {
  averageScore: number;
  count: number;
  categoryName?: string;
}

export default function SentimentOverview({ averageScore, count, categoryName }: SentimentOverviewProps) {
  const getSentimentEmoji = (score: number) => {
    if (score > 0.66) return '😊';
    if (score > 0.33) return '😐';
    return '😟';
  };

  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Gauge className="mr-2 h-5 w-5 text-primary" />
          Sentiment Overview {categoryName ? `for ${categoryName}` : ''}
        </CardTitle>
        <CardDescription>Overall sentiment trends based on analyzed papers.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-md">
          <div>
            <p className="text-sm text-muted-foreground">Average Sentiment Score</p>
            <p className="text-2xl font-bold">{averageScore.toFixed(2)} {getSentimentEmoji(averageScore)}</p>
          </div>
          <Gauge size={32} className="text-accent" />
        </div>
        <div className="flex items-center justify-between p-3 bg-secondary/30 rounded-md">
          <div>
            <p className="text-sm text-muted-foreground">Sentiment Data Points</p>
            <p className="text-2xl font-bold">{count.toLocaleString()}</p>
          </div>
          <Sigma size={32} className="text-accent" />
        </div>
      </CardContent>
    </Card>
  );
}

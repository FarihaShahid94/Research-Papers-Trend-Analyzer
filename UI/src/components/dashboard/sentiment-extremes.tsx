import type { SentimentExtreme } from '@/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ArrowUpCircle, ArrowDownCircle, Award, AlertTriangle } from 'lucide-react';

interface SentimentExtremesProps {
  highest: SentimentExtreme;
  lowest: SentimentExtreme;
}

export default function SentimentExtremes({ highest, lowest }: SentimentExtremesProps) {
  return (
    <Card className="shadow-lg rounded-lg">
      <CardHeader>
        <CardTitle className="text-lg flex items-center">
          <Award className="mr-2 h-5 w-5 text-primary" />
          Sentiment Extremes
        </CardTitle>
        <CardDescription>Categories with notable average sentiment scores.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-md shadow">
          <div className="flex items-center text-green-700 dark:text-green-400">
            <ArrowUpCircle className="mr-2 h-5 w-5" />
            <h3 className="font-semibold">Highest Sentiment</h3>
          </div>
          <p className="mt-1 text-sm">
            <strong>{highest.category}</strong> with an average score of <strong>{highest.score.toFixed(2)}</strong>.
          </p>
        </div>
        <div className="p-3 bg-red-100 dark:bg-red-900/30 rounded-md shadow">
          <div className="flex items-center text-red-700 dark:text-red-400">
            <ArrowDownCircle className="mr-2 h-5 w-5" />
            <h3 className="font-semibold">Lowest Sentiment</h3>
          </div>
          <p className="mt-1 text-sm">
            <strong>{lowest.category}</strong> with an average score of <strong>{lowest.score.toFixed(2)}</strong>.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
